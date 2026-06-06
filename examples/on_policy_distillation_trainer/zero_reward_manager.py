# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import os
import threading
import uuid
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

_EXTERNAL_REWARD_SCORE_PATH = os.environ.get(
    "VERIFY_SYSTEM_REWARD_SCORE_PATH",
    "/root/paddlejob/gpfsspace/liuyang_etr_orm/rl_new_verl/verl/verl/utils/reward_score",
)
_FORMAT_CHECK = None
_REQUEST_VS_JUDGE = None


def _load_verify_helpers():
    global _FORMAT_CHECK, _REQUEST_VS_JUDGE
    if _FORMAT_CHECK is not None and _REQUEST_VS_JUDGE is not None:
        return _FORMAT_CHECK, _REQUEST_VS_JUDGE

    if not os.path.isdir(_EXTERNAL_REWARD_SCORE_PATH):
        raise FileNotFoundError(f"VerifySystem reward_score path not found: {_EXTERNAL_REWARD_SCORE_PATH}")

    import verl.utils.reward_score as reward_score_pkg

    if _EXTERNAL_REWARD_SCORE_PATH not in reward_score_pkg.__path__:
        reward_score_pkg.__path__.append(_EXTERNAL_REWARD_SCORE_PATH)

    from verl.utils.reward_score.verify_system import format_check, request_vs_judge

    _FORMAT_CHECK = format_check
    _REQUEST_VS_JUDGE = request_vs_judge
    return _FORMAT_CHECK, _REQUEST_VS_JUDGE


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _as_list(value: Any) -> list[Any]:
    value = _to_python(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _parse_verifier(value: Any) -> list[dict[str, Any]]:
    value = _to_python(value)
    if isinstance(value, str):
        value = json.loads(value)
        if isinstance(value, str):
            value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError(f"verifier must be a list or JSON-encoded list, got {type(value).__name__}")
    return value


def _safe_token_len(tokenizer, text: str | None) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text)


def _extract_extra_fields(value: Any) -> dict[str, Any]:
    value = _to_python(value)
    return value if isinstance(value, dict) else {}


class ZeroRewardManager(RewardManagerBase):
    """Reward manager for OPD runs that intentionally ignore task rewards."""

    def __init__(self, config, tokenizer, compute_score=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)

    async def run_single(self, data: DataProto) -> dict:
        return {"reward_score": 0.0, "reward_extra_info": {"acc": 0.0}}


class OPDVerifierRewardManager(RewardManagerBase):
    """Verifier reward manager for logging OPD training reward/accuracy curves."""

    def __init__(self, config, tokenizer, compute_score=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.max_resp_len = config.reward.get("reward_kwargs", {}).get("max_resp_len", None)
        self.log_folder = os.environ.get("ROLLOUT_DIR", None)
        if self.log_folder:
            os.makedirs(self.log_folder, exist_ok=True)
        self.format_check, self.request_vs_judge = _load_verify_helpers()

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        loop = asyncio.get_running_loop()

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_length_int = int(valid_response_length.detach().cpu().item())
        valid_response_ids = response_ids[:valid_response_length_int]

        response_str = await loop.run_in_executor(None, lambda: self.tokenizer.decode(valid_response_ids))
        model_output = response_str
        response_for_check = response_str.replace("<|im_end|>", "").replace("<|endoftext|>", "")

        src = _as_list(data_item.non_tensor_batch["src"])
        category = str(_to_python(data_item.non_tensor_batch.get("category", "")))
        verifier_config = _parse_verifier(data_item.non_tensor_batch["verifier"])
        verifier_name = verifier_config[0].get("name", "") if verifier_config else ""

        format_follow_flag, think_content, answer_content = self.format_check(response_for_check)
        response_for_judge = answer_content if format_follow_flag else "None_Response"

        result = await loop.run_in_executor(
            None,
            lambda: self.request_vs_judge(src[0], response_for_judge, verifier_config, model_output),
        )
        if result is None:
            score = 0.0
        else:
            score = float(result["reward"][0])

        extra_fields = _extract_extra_fields(data_item.non_tensor_batch.get("tool_extra_fields", {}))
        param_version = extra_fields.get("global_steps") or extra_fields.get("max_global_steps") or -1
        log_id = str(uuid.uuid4())

        reward_extra_info = {
            "acc": score,
            "score": score,
            "format_follow": int(bool(format_follow_flag)),
            "valid_response_length": valid_response_length_int,
            "cot_length": _safe_token_len(self.tokenizer, think_content),
            "answer_length": _safe_token_len(self.tokenizer, answer_content),
            "overlong_ratio": int(valid_response_length_int >= self.max_resp_len) if self.max_resp_len else 0,
            "verifier_name": verifier_name,
            "category": category,
            "log_id": log_id,
        }

        self._maybe_write_log(
            {
                "log_id": log_id,
                "param_version": param_version,
                "category": category,
                "query": src[0],
                "response": response_for_judge,
                "model_output": model_output,
                "reward": score,
                "score": score,
                "format_follow": int(bool(format_follow_flag)),
                "valid_response_length": valid_response_length_int,
                "verifier_name": verifier_name,
            }
        )

        return {"reward_score": score, "reward_extra_info": reward_extra_info}

    def _maybe_write_log(self, log_entry: dict[str, Any]) -> None:
        if not self.log_folder:
            return

        def _write_task():
            param_version = log_entry.get("param_version", -1)
            output_file_path = os.path.join(self.log_folder, f"param_version_{param_version}.jsonl")
            with open(output_file_path, "a", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False, default=str)
                f.write("\n")

        threading.Thread(target=_write_task, daemon=True).start()
