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

"""自然语言人类反馈（Natural Feedback）的奖励函数。
相较于原版verl，需要修改call_back部分，即返回

该文件同时提供两套入口：
- NaturalFeedbackRewardManager：给 PPO/GRPO 的 reward_manager 使用（AbstractRewardManager 接口：__call__）。
- NaturalFeedbackRewardLoopManager：给 experimental reward_loop 使用（RewardManagerBase 接口：run_single）。

奖励定义（按你的公式）：
R =
    -1                          if output format invalid
     0                          if outcome incorrect (l_hat != l)
     1 + lambda * R_process      if outcome correct   (l_hat == l)

其中 R_process ∈ {0, 1}，由 critique 与 human_critique 的相似度是否超过阈值决定。

注意：为了兼容 verl 的验证指标聚合（process_validation_metrics），reward_extra_info 必须输出“扁平字段 + 数值列表”。

F1 打分方式（用于过程奖励 r_process）：
- token: 纯 token set 的 F1（本地计算，最快）
- llm_core: 使用 LLM 基于“核心论点（Core Human Critiques）”计算 F1（论文推荐）
- llm_all: 使用 LLM 基于“所有论点（All Human Critiques）”计算 F1（对比实验）

你可以在本文件顶部配置：LLM_URL / LLM_API_KEY / LLM_MODEL_NAME 以及 F1_METHOD。
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Tuple

import torch

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.workers.reward_manager.abstract import AbstractRewardManager


# =========================
# 0) User-configurable settings
# =========================

# F1_METHOD 可选："token" | "llm_core" | "llm_all"
F1_METHOD: str = os.environ.get("NATURAL_FEEDBACK_F1_METHOD", "llm_all")

# 强制仅使用 LLM 计算 F1（禁用 token F1 作为主路径）。
# 注意：这不会影响 token F1 函数本身是否存在，仅影响 _compute_f1 的选择与回退逻辑。
FORCE_LLM_F1: bool = os.environ.get("NATURAL_FEEDBACK_FORCE_LLM_F1", "1").lower() in ("1", "true", "t")


# =========================
# 1) LLM 配置（支持 OpenAI-compatible 接口）
# =========================

# 注意：请通过环境变量设置敏感信息，避免硬编码
LLM_URL: str = os.environ.get("NATURAL_FEEDBACK_LLM_URL", "")
LLM_API_KEY: str = os.environ.get("NATURAL_FEEDBACK_LLM_API_KEY", "")
LLM_MODEL_NAME: str = os.environ.get("NATURAL_FEEDBACK_LLM_MODEL_NAME", "")

# LLM 调用参数
LLM_TEMPERATURE: float = float(os.environ.get("NATURAL_FEEDBACK_LLM_TEMPERATURE", "0.01"))
LLM_MAX_TOKENS: int = int(os.environ.get("NATURAL_FEEDBACK_LLM_MAX_TOKENS", "2048"))
LLM_REQUEST_TIMEOUT: float = float(
    os.environ.get("NATURAL_FEEDBACK_LLM_REQUEST_TIMEOUT", os.environ.get("NATURAL_FEEDBACK_LLM_TIMEOUT_S", "120.0"))
)

# 并发与重试
LLM_MAX_RETRIES: int = int(os.environ.get("NATURAL_FEEDBACK_LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY: float = float(os.environ.get("NATURAL_FEEDBACK_LLM_RETRY_BASE_DELAY", "1.0"))
LLM_RETRY_JITTER: float = float(os.environ.get("NATURAL_FEEDBACK_LLM_RETRY_JITTER", "0.2"))

# 并发上限（信号量）
LLM_MAX_CONCURRENT_REQUESTS: int = int(os.environ.get("NATURAL_FEEDBACK_LLM_MAX_CONCURRENT_REQUESTS", "80"))

# 最小间隔（全局节流，=0 表示关闭）
LLM_MIN_INTERVAL_S: float = float(os.environ.get("NATURAL_FEEDBACK_LLM_MIN_INTERVAL_S", "0"))

# Reward 计算的样本级并发（旧 NaturalFeedbackRewardManager 路径使用）
# 说明：LLM 调用仍会受 _llm_inflight_sema（LLM_MAX_CONCURRENT_REQUESTS）限制。
REWARD_ITEM_MAX_WORKERS: int = int(os.environ.get("NATURAL_FEEDBACK_REWARD_ITEM_MAX_WORKERS", "80"))

# 缓存配置
LLM_CACHE_SIZE: int = int(os.environ.get("NATURAL_FEEDBACK_LLM_CACHE_SIZE", "10000"))
LLM_CACHE_TTL: float = float(os.environ.get("NATURAL_FEEDBACK_LLM_CACHE_TTL", "86400"))

# 回退机制（当 FORCE_LLM_F1=1 时，回退会被禁用）
LLM_FALLBACK_TO_TOKEN: bool = os.environ.get("NATURAL_FEEDBACK_LLM_FALLBACK_TO_TOKEN", "1").lower() in (
    "1",
    "true",
    "t",
)

# 调试与监控
LLM_DEBUG: bool = os.environ.get("NATURAL_FEEDBACK_LLM_DEBUG", "0").lower() in ("1", "true", "t")
LLM_DEBUG_LOG_INTERVAL: int = int(os.environ.get("NATURAL_FEEDBACK_LLM_DEBUG_LOG_INTERVAL", "100"))
LLM_ERROR_LOG_INTERVAL: int = int(os.environ.get("NATURAL_FEEDBACK_LLM_ERROR_LOG_INTERVAL", "100"))

_llm_success_log_counter = 0
_llm_error_log_counter = 0


# =========================
# 2) Prompt templates
# =========================

PROMPT_F1_LLM_CORE = """I will provide you with a generated evaluation content and a reference evaluation content. Your task is to analyze the similarity between the <Generated Evaluation Content> and the <Reference Evaluation Content> by calculating F1 scores based on their key arguments.

Core Principle: Focus exclusively on \"Key Arguments\" - decisive reasons that are powerful enough to justify the final choice on their own. Identify these core justifications, not minor points.

## Part 1: First check
First check if the generated critique repeats the same point across multiple times. If yes, directly output without conducting part 2:
<thinking>
Put here how the generated critique repeats points.
</thinking>
<scores>
<critique_f1>0</critique_f1>
<critique_precision>0</critique_precision>
<critique_recall>0</critique_recall>
</scores>
## Part 2: Steps for F1 Score Calculation
1. Count Reference Key Arguments (N_ref):
Check if the reference identifies a fatal error (critical factual error, harmful statement, or fundamental misunderstanding).
- If yes: Only this fatal error counts. Set N_ref = 1.
- If no: Count all unique Key Arguments (decisive reasons that could justify the choice by themselves). Set N_ref to this count.

2. Count Generated Key Arguments (N_gen):
- Identify all unique Key Arguments in the generated evaluation.
- Set N_gen to this count.

3. Count True Positives (TP):
- Initialize TP = 0.
- For each reference key argument, search for a match in generated key arguments.
- Matching Rule: Both semantic meaning and stance (which response and positive/negative) must align.
- Example: \"Response A is more detailed\" only matches with similar praise of Response A, not Response B.
- For fatal errors: Generated must identify the same error in the same response.
- Each generated argument can only match once.
- Increment TP by 1 for each valid match.

4. Calculate Scores:
- Precision_critique: TP / N_gen (0 if N_gen = 0)
- Recall_critique: TP / N_ref (0 if N_ref = 0)
- CritiqueScore: 2 * (Precision * Recall) / (Precision + Recall) (0 if sum = 0)

Output Format (rounded to 4 decimal places):
<thinking>
Put the thinking process here.
</thinking>
<scores>
<critique_f1>CritiqueScore</critique_f1>
<critique_precision>Precision_critique</critique_precision>
<critique_recall>Recall_critique</critique_recall>
</scores>

<Generated Evaluation Content>
{critiques}
</Generated Evaluation Content>
<Reference Evaluation Content>
{reference_critiques}
</Reference Evaluation Content>
"""

PROMPT_F1_LLM_ALL = """I will provide you with a generated evaluation content and a reference evaluation content. Your task is to analyze the similarity between the <Generated Evaluation Content> and the <Reference Evaluation Content> by calculating F1 scores based on their all arguments.

## Part 1: First check
First check if the generated critique repeats the same point across multiple times. If yes, directly output without conducting part 2:
<thinking>
Put here how the generated critique repeats points.
</thinking>
<scores>
<critique_f1>0</critique_f1>
<critique_precision>0</critique_precision>
<critique_recall>0</critique_recall>
</scores>
## Part 2: Steps for F1 Score Calculation
1. Count Reference All Arguments (N_ref):
- Check if the reference identifies a fatal error (critical factual error, harmful statement, or fundamental misunderstanding).
-- If yes: Only this fatal error counts. Set N_ref = 1.
-- If no: Count all unique Arguments (decisive reasons that could justify the choice by themselves). Set N_ref to this count.

2. Count Generated All Arguments (N_gen):
- Identify all unique Arguments in the generated evaluation.
- Set N_gen to this count.

3. Count True Positives (TP):
- Initialize TP = 0.
- For each reference argument, search for a match in generated arguments.
- Matching Rule: Both semantic meaning and stance (which response and positive/negative) must align.
-- Example: \"Response A is more detailed\" only matches with similar praise of Response A, not Response B.
-- For fatal errors: Generated must identify the same error in the same response.
- Each generated argument can only match once.
- Increment TP by 1 for each valid match.

4. Calculate Scores:
- Precision_critique: TP / N_gen (0 if N_gen = 0)
- Recall_critique: TP / N_ref (0 if N_ref = 0)
- CritiqueScore: 2 * (Precision * Recall) / (Precision + Recall) (0 if sum = 0)

Output Format (rounded to 4 decimal places):
<thinking>
Put the thinking process here.
</thinking>
<scores>
<critique_f1>CritiqueScore</critique_f1>
<critique_precision>Precision_critique</critique_precision>
<critique_recall>Recall_critique</critique_recall>
</scores>

<Generated Evaluation Content>
{critiques}
</Generated Evaluation Content>
<Reference Evaluation Content>
{reference_critiques}
</Reference Evaluation Content>
"""


# =========================
# 3) Parsing helpers
# =========================


def _extract_xml_tag(text: str, tag: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_choice_from_tag(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\[\[\s*([AB])\s*\]\]", text)
    return m.group(1) if m else None


def _normalize_choice(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'", "`"}:
        s = s[1:-1].strip()
    s = s.upper()
    if s in {"A", "0"}:
        return "A"
    if s in {"B", "1"}:
        return "B"
    m = re.search(r"\b([AB])\b", s)
    return m.group(1) if m else None


# =========================
# 4) Token F1 (kept for optional fallback / debugging)
# =========================


def _f1_token(human: str, gen: str) -> float:
    set_a = set(human.split())
    set_b = set(gen.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    if inter == 0:
        return 0.0
    p = inter / len(set_a)
    r = inter / len(set_b)
    return 2 * p * r / (p + r)


# =========================
# 5) LLM F1 parsing
# =========================


_FLOAT_RE = r"[+-]?(?:\d+\.\d+|\d+\.?|\.\d+)(?:[eE][+-]?\d+)?"


def _parse_llm_f1_scores(text: str) -> Tuple[float, float, float]:
    blocks = re.findall(r"<scores>([\s\S]*?)</scores>", text)
    search_text = blocks[-1] if blocks else text

    def _get(tag: str) -> Optional[float]:
        m = re.search(rf"<{tag}>\s*({_FLOAT_RE})\s*</{tag}>", search_text)
        return float(m.group(1)) if m else None

    f1 = _get("critique_f1")
    p = _get("critique_precision")
    r = _get("critique_recall")
    if f1 is None or p is None or r is None:
        raise ValueError("Failed to parse llm f1/precision/recall from response")

    f1 = max(0.0, min(1.0, float(f1)))
    p = max(0.0, min(1.0, float(p)))
    r = max(0.0, min(1.0, float(r)))
    return f1, p, r


# =========================
# 6) LLM call utilities (cache + rate limit + concurrency)
# =========================


_llm_cache_lock = threading.Lock()
_llm_cache: "OrderedDict[str, Tuple[float, Tuple[float, float, float]]]" = OrderedDict()

_llm_last_request_lock = threading.Lock()
_llm_last_request_ts = 0.0

_llm_inflight_sema = threading.Semaphore(LLM_MAX_CONCURRENT_REQUESTS)


def _cache_get(key: str) -> Optional[Tuple[float, float, float]]:
    now = time.time()
    with _llm_cache_lock:
        item = _llm_cache.get(key)
        if item is None:
            return None
        ts, val = item
        if (now - ts) > LLM_CACHE_TTL:
            _llm_cache.pop(key, None)
            return None
        _llm_cache.move_to_end(key)
        return val


def _cache_put(key: str, val: Tuple[float, float, float]) -> None:
    now = time.time()
    with _llm_cache_lock:
        _llm_cache[key] = (now, val)
        _llm_cache.move_to_end(key)
        while len(_llm_cache) > LLM_CACHE_SIZE:
            _llm_cache.popitem(last=False)


def _rate_limit_sleep() -> None:
    global _llm_last_request_ts
    if LLM_MIN_INTERVAL_S <= 0:
        return
    with _llm_last_request_lock:
        now = time.time()
        delta = now - _llm_last_request_ts
        if delta < LLM_MIN_INTERVAL_S:
            time.sleep(LLM_MIN_INTERVAL_S - delta)
        _llm_last_request_ts = time.time()


def _make_cache_key(method: str, reference_critiques: str, critiques: str) -> str:
    h = hashlib.sha256()
    h.update(method.encode("utf-8"))
    h.update(b"\n")
    h.update(reference_critiques.encode("utf-8", errors="ignore"))
    h.update(b"\n\n")
    h.update(critiques.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _extract_llm_content(data: Any) -> str:
    try:
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return ""


def _call_llm_f1(prompt: str, cache_key: Optional[str] = None) -> Tuple[float, float, float]:
    global _llm_success_log_counter

    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            if LLM_DEBUG:
                _llm_success_log_counter += 1
                if LLM_DEBUG_LOG_INTERVAL <= 1 or (_llm_success_log_counter % LLM_DEBUG_LOG_INTERVAL == 0):
                    print(f"[LLM F1 CACHE HIT] key={cache_key[:8]}...")
            return cached

    if not LLM_URL or not LLM_MODEL_NAME:
        raise ValueError("LLM_URL/LLM_MODEL_NAME not set")

    import requests

    url = (LLM_URL or "").strip()
    url = url.lstrip("= ").strip()
    url = url.rstrip("/")

    if url.endswith("/v1/chat/completions"):
        pass
    elif url.endswith("/v1"):
        url = url + "/chat/completions"
    else:
        url = url + "/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }

    last_err: Optional[Exception] = None

    with _llm_inflight_sema:
        for attempt in range(LLM_MAX_RETRIES):
            try:
                _rate_limit_sleep()
                if LLM_DEBUG and attempt == 0:
                    print(
                        f"[LLM F1 REQUEST] model={LLM_MODEL_NAME} url={url} "
                        f"temp={LLM_TEMPERATURE} max_tokens={LLM_MAX_TOKENS} key={str(cache_key)[:8]}..."
                    )
                resp = requests.post(url, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT)

                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}")

                resp.raise_for_status()
                data = resp.json()
                content = _extract_llm_content(data)
                try:
                    scores = _parse_llm_f1_scores(content)
                except Exception as pe:
                    if LLM_DEBUG:
                        global _llm_error_log_counter
                        _llm_error_log_counter += 1
                        if LLM_ERROR_LOG_INTERVAL <= 1 or (_llm_error_log_counter % LLM_ERROR_LOG_INTERVAL == 0):
                            snippet = content.replace("\n", " ")[:600]
                            print(
                                f"[LLM F1 PARSE FAIL] ({_llm_error_log_counter}) {type(pe).__name__}: {str(pe)[:120]} | "
                                f"content_snippet={snippet}"
                            )
                    raise

                if cache_key:
                    _cache_put(cache_key, scores)

                if LLM_DEBUG:
                    _llm_success_log_counter += 1
                    if LLM_DEBUG_LOG_INTERVAL <= 1 or (_llm_success_log_counter % LLM_DEBUG_LOG_INTERVAL == 0):
                        f1v, pv, rv = scores
                        print(f"[LLM F1 OK] key={str(cache_key)[:8]}... f1={f1v:.4f} p={pv:.4f} r={rv:.4f}")

                return scores
            except Exception as e:
                last_err = e
                if LLM_DEBUG:
                    print(
                        f"[LLM F1 RETRY] attempt={attempt+1}/{LLM_MAX_RETRIES} err={type(e).__name__}: {str(e)[:200]}"
                    )

                if attempt >= LLM_MAX_RETRIES - 1:
                    break

                delay = LLM_RETRY_BASE_DELAY * (2**attempt) + random.random() * LLM_RETRY_JITTER
                if LLM_DEBUG:
                    print(f"[LLM F1 RETRY SLEEP] {delay:.2f}s")
                time.sleep(delay)

    assert last_err is not None
    raise last_err


# =========================
# 7) F1 public helpers
# =========================


def f1_llm_core(reference_critiques: str, critiques: str) -> Tuple[float, float, float]:
    prompt = PROMPT_F1_LLM_CORE.format(critiques=critiques, reference_critiques=reference_critiques)
    cache_key = _make_cache_key("llm_core", reference_critiques=reference_critiques, critiques=critiques)
    return _call_llm_f1(prompt, cache_key=cache_key)


def f1_llm_all(reference_critiques: str, critiques: str) -> Tuple[float, float, float]:
    prompt = PROMPT_F1_LLM_ALL.format(critiques=critiques, reference_critiques=reference_critiques)
    cache_key = _make_cache_key("llm_all", reference_critiques=reference_critiques, critiques=critiques)
    return _call_llm_f1(prompt, cache_key=cache_key)


def _compute_f1(human_critique: str, critique: str) -> Tuple[float, float, float]:
    method = (F1_METHOD or "token").strip().lower()

    if method == "token":
        if FORCE_LLM_F1:
            raise ValueError("F1_METHOD=token is disabled when NATURAL_FEEDBACK_FORCE_LLM_F1=1")
        f1 = _f1_token(human_critique, critique)
        return f1, 0.0, 0.0

    try:
        if method == "llm_core":
            return f1_llm_core(reference_critiques=human_critique, critiques=critique)
        elif method == "llm_all":
            return f1_llm_all(reference_critiques=human_critique, critiques=critique)
    except Exception as e:
        if LLM_DEBUG:
            global _llm_error_log_counter
            _llm_error_log_counter += 1
            if LLM_ERROR_LOG_INTERVAL <= 1 or (_llm_error_log_counter % LLM_ERROR_LOG_INTERVAL == 0):
                print(f"[LLM F1 Error] ({_llm_error_log_counter}) {type(e).__name__}: {str(e)[:200]}")

        if LLM_FALLBACK_TO_TOKEN and not FORCE_LLM_F1:
            f1 = _f1_token(human_critique, critique)
            return f1, 0.0, 0.0

        if FORCE_LLM_F1:
            print(f"[LLM F1 Failed] {type(e).__name__}: {str(e)[:200]}, returning f1=0")
            return 0.0, 0.0, 0.0

        raise

    raise ValueError(f"Unsupported F1_METHOD: {F1_METHOD}")


# =========================
# 8) Batch reward entry (for BatchRewardManager)
# =========================
#
# 使用方式：
# - reward_manager.source=register
# - reward_manager.name=batch
# - custom_reward_function.path=.../natural_feedback_reward.py
# - custom_reward_function.name=compute_score_batch
#
# 说明：输入是 list，输出是 list[dict]，每个 dict 至少包含 score。
# 该函数内部会对需要的样本并发调用外部 LLM 计算 F1。
# =========================


def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[dict],
    enable_process_reward: bool = True,
    process_reward_lambda: float = 0.5,
    process_similarity_threshold: float = 0.5,
    **kwargs,
) -> List[dict]:
    n = len(solution_strs)
    results: List[dict] = [{"score": 0.0} for _ in range(n)]

    tasks: list[tuple[int, str, str]] = []  # (idx, human_critique, critique)
    parsed: list[tuple[bool, float, float]] = []  # (format_ok, r_format, r_outcome)
    pred_list: list[str] = ["N/A"] * n
    gt_list: list[str] = ["N/A"] * n

    for i in range(n):
        gen_text = solution_strs[i]

        critique = _extract_xml_tag(gen_text, "critics")
        choice_tag = _extract_xml_tag(gen_text, "choice")
        model_choice = _normalize_choice(_extract_choice_from_tag(choice_tag) or _extract_choice_from_tag(gen_text))
        format_ok = (critique is not None) and (model_choice is not None)

        gt_choice_raw = None
        gt_obj = ground_truths[i]
        if isinstance(gt_obj, str):
            gt_choice_raw = gt_obj
        elif isinstance(gt_obj, dict):
            gt_choice_raw = gt_obj.get("ground_truth_choice", None)
            if gt_choice_raw is None:
                gt_choice_raw = gt_obj.get("ground_truth", None)
        if gt_choice_raw is None:
            gt_choice_raw = (extra_infos[i] or {}).get("ground_truth_choice", None)
        gt_choice = _normalize_choice(gt_choice_raw)

        pred_list[i] = model_choice if model_choice is not None else "N/A"
        gt_list[i] = gt_choice if gt_choice is not None else "N/A"

        if not format_ok:
            r_format = -1.0
            r_outcome = 0.0
        else:
            r_format = 0.0
            r_outcome = 1.0 if (model_choice == gt_choice) else 0.0

        parsed.append((format_ok, r_format, r_outcome))

        if enable_process_reward and format_ok and r_outcome == 1.0:
            human_critique = (extra_infos[i] or {}).get("human_critique", "")
            if not human_critique and isinstance(ground_truths[i], dict):
                human_critique = ground_truths[i].get("human_critique", "")
            if human_critique and critique:
                tasks.append((i, human_critique, critique))

    f1_results: dict[int, tuple[float, float, float]] = {}
    if tasks:
        max_workers = max(1, int(LLM_MAX_CONCURRENT_REQUESTS))
        t0 = time.time()
        done = 0
        total = len(tasks)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(_compute_f1, hc, c): idx for (idx, hc, c) in tasks}
            for fut in as_completed(fut_map):
                idx = fut_map[fut]
                try:
                    f1_results[idx] = fut.result()
                except Exception:
                    if LLM_FALLBACK_TO_TOKEN and not FORCE_LLM_F1:
                        for (j, hc, c) in tasks:
                            if j == idx:
                                f1_results[idx] = (_f1_token(hc, c), 0.0, 0.0)
                                break
                    else:
                        f1_results[idx] = (0.0, 0.0, 0.0)

                done += 1
                if LLM_DEBUG and (done % 10 == 0 or done == total):
                    elapsed = time.time() - t0
                    print(f"[LLM F1 PROGRESS] done={done}/{total} elapsed_s={elapsed:.2f}")

    for i in range(n):
        format_ok, r_format, r_outcome = parsed[i]

        r_process = 0.0
        f1_score = 0.0
        p_score = 0.0
        r_score = 0.0

        if enable_process_reward and format_ok and r_outcome == 1.0:
            if i in f1_results:
                f1_score, p_score, r_score = f1_results[i]
                r_process = 1.0 if f1_score > process_similarity_threshold else 0.0

        if not format_ok:
            score = -1.0
        else:
            if r_outcome == 0.0:
                score = 0.0
            else:
                score = 1.0 + process_reward_lambda * r_process

        if r_outcome == 1.0:
            p_process0_given_outcome1 = 1.0 if r_process == 0.0 else 0.0
        else:
            # 验证阶段 process_validation_metrics 会对该字段做 np.mean，None 会触发 TypeError
            # 用 NaN 占位，后续聚合逻辑可按需过滤 NaN
            p_process0_given_outcome1 = float("nan")

        results[i] = {
            "score": float(score),
            "r_format": float(r_format),
            "r_outcome": float(r_outcome),
            "r_process": float(r_process),
            "f1": float(f1_score),
            "p": float(p_score),
            "r": float(r_score),
            "p_process0_given_outcome1": p_process0_given_outcome1,
            "format_ok": 1.0 if format_ok else 0.0,
            "pred": pred_list[i],
            "gt": gt_list[i],
        }

    return results


# =========================
# 9) (Legacy) RewardManager entry
# =========================
#
# 旧路径：reward_manager.source=importlib & name=NaturalFeedbackRewardManager
# 当前你已切到 BatchRewardManager + compute_score_batch，下面类保留供兼容。
# =========================


class NaturalFeedbackRewardManager(AbstractRewardManager):
    """给 PPO/GRPO 使用的 RewardManager（AbstractRewardManager 接口）。"""

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: Any,
        reward_fn_key: str,
        enable_process_reward: bool = True,
        process_reward_lambda: float = 0.5,
        process_similarity_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(tokenizer, num_examine, compute_score, reward_fn_key, **kwargs)
        self.tokenizer = tokenizer
        self.enable_process_reward = enable_process_reward
        self.lam = process_reward_lambda
        self.threshold = process_similarity_threshold

    def _process_one(
        self,
        i: int,
        response_ids: torch.Tensor,
        valid_response_lengths: torch.Tensor,
        nb: dict,
    ):
        valid_len = int(valid_response_lengths[i].item())
        valid_response_ids = response_ids[i][:valid_len]
        gen_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        critique = _extract_xml_tag(gen_text, "critics")
        choice_tag = _extract_xml_tag(gen_text, "choice")
        model_choice = _normalize_choice(_extract_choice_from_tag(choice_tag))
        format_ok = (critique is not None) and (model_choice is not None)

        gt_choice = _normalize_choice(nb.get("ground_truth_choice", [None])[i])

        if not format_ok:
            r_format = -1.0
            r_outcome = 0.0
            r_process = 0.0
            f1_score = 0.0
            p_score = 0.0
            r_score = 0.0
            reward = -1.0
        else:
            r_format = 0.0
            r_outcome = 1.0 if (model_choice == gt_choice) else 0.0

            r_process = 0.0
            f1_score = 0.0
            p_score = 0.0
            r_score = 0.0

            if self.enable_process_reward and r_outcome == 1.0:
                human_critique = nb.get("human_critique", [""])[i]
                if human_critique and critique:
                    try:
                        f1_score, p_score, r_score = _compute_f1(human_critique, critique)
                    except Exception:
                        if LLM_FALLBACK_TO_TOKEN and not FORCE_LLM_F1:
                            f1_score = _f1_token(human_critique, critique)
                            p_score = 0.0
                            r_score = 0.0
                        else:
                            f1_score = 0.0
                            p_score = 0.0
                            r_score = 0.0
                    r_process = 1.0 if f1_score > self.threshold else 0.0

            if r_outcome == 0.0:
                reward = 0.0
            else:
                reward = 1.0 + self.lam * r_process

        if r_outcome == 1.0:
            p_process0_given_outcome1 = 1.0 if r_process == 0.0 else 0.0
        else:
            p_process0_given_outcome1 = None

        return (
            float(reward),
            float(r_format),
            float(r_outcome),
            float(r_process),
            float(f1_score),
            float(p_score),
            float(r_score),
            p_process0_given_outcome1,
            1.0 if format_ok else 0.0,
            model_choice if model_choice is not None else "N/A",
            gt_choice if gt_choice is not None else "N/A",
        )

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        nb = data.non_tensor_batch
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        prompt_len = data.batch["prompts"].shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        n = int(response_ids.shape[0])
        rewards: List[float] = [0.0] * n
        r_format_list: List[float] = [0.0] * n
        r_outcome_list: List[float] = [0.0] * n
        r_process_list: List[float] = [0.0] * n
        f1_list: List[float] = [0.0] * n
        p_list: List[float] = [0.0] * n
        r_list: List[float] = [0.0] * n
        p_process0_given_outcome1_list: List[Optional[float]] = [None] * n
        format_ok_list: List[float] = [0.0] * n
        pred_list: List[str] = ["N/A"] * n
        gt_list: List[str] = ["N/A"] * n

        item_workers = REWARD_ITEM_MAX_WORKERS
        if item_workers <= 0:
            item_workers = max(1, int(LLM_MAX_CONCURRENT_REQUESTS))
        item_workers = min(item_workers, n)

        if item_workers <= 1:
            for i in range(n):
                (
                    rewards[i],
                    r_format_list[i],
                    r_outcome_list[i],
                    r_process_list[i],
                    f1_list[i],
                    p_list[i],
                    r_list[i],
                    p_process0_given_outcome1_list[i],
                    format_ok_list[i],
                    pred_list[i],
                    gt_list[i],
                ) = self._process_one(i, response_ids, valid_response_lengths, nb)
        else:
            t0 = time.time()
            done = 0
            with ThreadPoolExecutor(max_workers=item_workers) as ex:
                futures = {
                    ex.submit(self._process_one, i, response_ids, valid_response_lengths, nb): i for i in range(n)
                }
                for fut in as_completed(futures):
                    i = futures[fut]
                    (
                        rewards[i],
                        r_format_list[i],
                        r_outcome_list[i],
                        r_process_list[i],
                        f1_list[i],
                        p_list[i],
                        r_list[i],
                        p_process0_given_outcome1_list[i],
                        format_ok_list[i],
                        pred_list[i],
                        gt_list[i],
                    ) = fut.result()
                    done += 1
                    if done == 1:
                        print(
                            f"[REWARD ITEM START] n={n} item_workers={item_workers} "
                            f"llm_inflight={LLM_MAX_CONCURRENT_REQUESTS} min_interval_s={LLM_MIN_INTERVAL_S}"
                        )
                    if LLM_DEBUG and (done % 10 == 0 or done == n):
                        elapsed = time.time() - t0
                        print(f"[REWARD ITEM PROGRESS] done={done}/{n} elapsed_s={elapsed:.2f}")

        reward_tensor = torch.tensor(rewards, dtype=torch.float32)

        if return_dict:
            reward_extra_info = {
                "r_format": r_format_list,
                "r_outcome": r_outcome_list,
                "r_process": r_process_list,
                "f1": f1_list,
                "p": p_list,
                "r": r_list,
                "p_process0_given_outcome1": p_process0_given_outcome1_list,
                "format_ok": format_ok_list,
                "pred": pred_list,
                "gt": gt_list,
            }
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor


# =========================
# 10) RewardLoop adapter (experimental)
# =========================


class NaturalFeedbackRewardLoopManager(RewardManagerBase):
    """给 experimental reward_loop 使用的适配器。

    RewardLoopWorker 期望 reward manager 提供：async run_single(DataProto) -> dict。
    内部复用 NaturalFeedbackRewardManager 的规则计算。
    """

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
        self._ppo_rm = NaturalFeedbackRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=self.compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )

    async def run_single(self, data: DataProto):
        out = self._ppo_rm(data, return_dict=True)
        reward_tensor = out["reward_tensor"]
        reward_score = float(reward_tensor.reshape(-1)[0].item())

        extra = out.get("reward_extra_info", {})

        def _get0(key, default):
            v = extra.get(key, default)
            if isinstance(v, list) and v:
                return v[0]
            return default

        reward_extra_info = {
            "r_format": float(_get0("r_format", 0.0)),
            "r_outcome": float(_get0("r_outcome", 0.0)),
            "r_process": float(_get0("r_process", 0.0)),
            "f1": float(_get0("f1", 0.0)),
            "p": float(_get0("p", 0.0)),
            "r": float(_get0("r", 0.0)),
            "format_ok": float(_get0("format_ok", 0.0)),
            "pred": _get0("pred", "N/A"),
            "gt": _get0("gt", "N/A"),
        }

        p0 = _get0("p_process0_given_outcome1", None)
        reward_extra_info["p_process0_given_outcome1"] = float("nan") if p0 is None else float(p0)

        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}
