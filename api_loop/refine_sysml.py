#!/usr/bin/env python3
"""Iteratively refine SysMLv2 models with provider-backed LLM APIs and SysIDE."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from time import perf_counter, sleep

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency for provider selection
    OpenAI = None  # type: ignore[assignment]

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - optional dependency for provider selection
    Anthropic = None  # type: ignore[assignment]

# Paths relative to this file so the script works from anywhere inside the repo.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MISTRAL_LARGE_MODEL = "mistral-large-latest"
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "deepseek_reasoner", "mistral_large"),
        default="openai",
        help="LLM API provider to call.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the requirements prompt (.json or .txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "runs",
        help="Directory for generated SysML files and logs.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help="Model identifier for the selected provider.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="Number of refinement attempts to perform.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the model (omit to use API default).",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=50000,
        help="Stop once roughly this many LLM tokens have been consumed.",
    )
    parser.add_argument(
        "--example",
        type=Path,
        help="Optional path to an example SysML snippet to include in the prompt.",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        required=True,
        help="Virtual environment root that owns the syside CLI (must contain bin/python).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM and compiler calls (useful for smoke tests).",
    )
    parser.add_argument(
        "--syside-timeout-seconds",
        type=int,
        default=60,
        help="Timeout for each syside subprocess call.",
    )
    parser.add_argument(
        "--syside-validate-with",
        choices=("format", "check"),
        default="format",
        help="Validation command used to assess generated SysML.",
    )
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=8,
        help="How many times to retry failed API calls per iteration.",
    )
    parser.add_argument(
        "--api-retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base delay before retrying failed API calls.",
    )
    parser.add_argument(
        "--api-retry-max-backoff-seconds",
        type=float,
        default=30.0,
        help="Maximum retry backoff delay for failed API calls.",
    )
    parser.add_argument(
        "--api-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-request timeout for API calls.",
    )
    parser.add_argument(
        "--anthropic-max-output-tokens",
        type=int,
        default=8192,
        help="Max output tokens for Anthropic calls (required by Anthropic API).",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default=DEFAULT_DEEPSEEK_BASE_URL,
        help="Base URL for DeepSeek OpenAI-compatible API.",
    )
    parser.add_argument(
        "--mistral-base-url",
        default=DEFAULT_MISTRAL_BASE_URL,
        help="Base URL for Mistral OpenAI-compatible API.",
    )
    parser.add_argument(
        "--resume-source-dir",
        type=Path,
        help=(
            "Optional previous run directory containing iteration_XX files. "
            "When set, generation resumes from --resume-from-iteration + 1."
        ),
    )
    parser.add_argument(
        "--resume-from-iteration",
        type=int,
        default=0,
        help="Iteration number in --resume-source-dir to resume from (e.g., 8).",
    )
    parser.add_argument(
        "--max-additional-prompts",
        type=int,
        default=0,
        help=(
            "Additional iterations to run when resuming. "
            "If 0, --max-iters is used as the additional count."
        ),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_markdown(path: Path) -> str:
    """Lightweight markdown-to-text helper (strip fences/headings only)."""
    raw = path.read_text(encoding="utf-8").strip()
    lines: List[str] = []
    inside_fence = False
    for line in raw.splitlines():
        if line.strip().startswith("```"):
            inside_fence = not inside_fence
            continue
        if inside_fence:
            lines.append(line)
            continue
        if line.lstrip().startswith("#"):
            lines.append(line.lstrip("# ").strip())
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def load_user_input(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for candidate_key in ("prompt", "requirements", "input", "description"):
            value = data.get(candidate_key)
            if isinstance(value, str) and value.strip():
                base = value.strip()
                break
        else:
            base = json.dumps(data, indent=2, ensure_ascii=False)
        extras = data.get("extra_context") or data.get("context")
        if isinstance(extras, str) and extras.strip():
            base = f"{base}\n\nAdditional context:\n{extras.strip()}"
        return base
    if suffix == ".md":
        return read_markdown(path)
    return path.read_text(encoding="utf-8").strip()


def load_example_snippet(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def extract_text_from_response(response) -> str:
    text_chunks: List[str] = []
    maybe_text = getattr(response, "output_text", None)
    if isinstance(maybe_text, str) and maybe_text.strip():
        return maybe_text.strip()
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if isinstance(text, str):
                text_chunks.append(text)
    return "\n".join(text_chunks).strip()


def extract_text_from_anthropic_response(response) -> str:
    text_chunks: List[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            text_chunks.append(text)
    return "\n".join(text_chunks).strip()


def extract_text_from_openai_chat_completion_response(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")


def sanitize_candidate_text(text: str) -> str:
    """Strip markdown fences/prose noise and return raw SysML text."""
    cleaned = text.strip()
    if "```" in cleaned:
        lines: List[str] = []
        inside_fence = False
        for line in cleaned.splitlines():
            if line.strip().startswith("```"):
                inside_fence = not inside_fence
                continue
            if inside_fence:
                lines.append(line)
        if lines:
            cleaned = "\n".join(lines).strip()
    for marker in ("package ", "package '", 'package "'):
        idx = cleaned.find(marker)
        if idx > 0:
            cleaned = cleaned[idx:].strip()
            break
    return cleaned


def compact_compiler_feedback(stdout: str, stderr: str, max_lines: int = 80, max_chars: int = 4000) -> str:
    merged = "\n".join(part for part in [stdout, stderr] if part).strip()
    if not merged:
        return ""
    merged = ANSI_ESCAPE_RE.sub("", merged)
    lines = [line.rstrip() for line in merged.splitlines() if line.strip()]
    if len(lines) > max_lines:
        head = lines[: max_lines // 2]
        tail = lines[-(max_lines // 2) :]
        lines = head + ["... (truncated compiler output) ..."] + tail
    compacted = "\n".join(lines).strip()
    if len(compacted) > max_chars:
        compacted = compacted[:max_chars].rstrip() + "\n... (truncated compiler output) ..."
    return compacted


def is_infrastructure_compiler_failure(stdout: str, stderr: str) -> bool:
    """Detect environment/runtime failures that should not be sent to the model."""
    text = f"{stdout}\n{stderr}"
    indicators = (
        "Traceback (most recent call last):",
        "ModuleNotFoundError:",
        "ImportError:",
        "No module named",
        "PermissionError:",
        "FileNotFoundError:",
    )
    return any(marker in text for marker in indicators)


def build_prompt(
    spec_text: str,
    iteration: int,
    previous_candidate: Optional[str],
    compiler_feedback: Optional[str],
    example_text: Optional[str],
) -> str:
    sections = [
        "You are generating SysML v2 text in a compile-fix loop.",
        textwrap.dedent(
            f"""
            REQUIREMENTS:
            {spec_text.strip()}
            """
        ).strip(),
        "Goal: output a SysML v2 model that validates and satisfies the requirements.",
        "Rules: output only SysML text with no prose or markdown; prioritize fixing "
        "validator errors first; if a previous candidate is provided, apply minimal "
        "edits and preserve valid parts.",
        "Use a conservative subset that is robust with syside: `package`, "
        "`private import ScalarValues::*`, `enum def` with `enum` members, `part def`, "
        "`part`, `attribute`, `port def`, `port`, `item def`, and `connect A to B;`.",
        "Prefer simple declarations: `attribute name : Type;` or "
        "`attribute name : Type = Value;`. Keep a single package and minimal structure.",
        "Avoid constructs that frequently fail validation: `block`, `property`, "
        "`enumeration`, `signal def`, `event def`, `state machine`, `stateMachine`, "
        "`transition def`, `operation def`, `precondition`, `postcondition`, `when`, "
        "`connect A -> B`, and quoted pseudo-code constraints.",
    ]
    if previous_candidate:
        sections.append(
            textwrap.dedent(
                f"""
                PREVIOUS ATTEMPT (iteration {iteration - 1}):
                {previous_candidate.strip()}
                """
            ).strip()
        )
    if example_text:
        sections.append(
            textwrap.dedent(
                f"""
                REFERENCE EXAMPLE SYSML SNIPPET:
                {example_text.strip()}
                """
            ).strip()
        )
    if compiler_feedback:
        sections.append(
            textwrap.dedent(
                f"""
                SYSIDE COMPILER FEEDBACK TO ADDRESS:
                {compiler_feedback.strip()}
                """
            ).strip()
        )
        sections.append(
            "Revise the earlier model to resolve the diagnostics without undoing "
            "correct structure."
        )
    sections.append("Return only the updated SysMLv2 model.")
    return "\n\n".join(section for section in sections if section)


def call_model(
    client,
    provider: str,
    prompt: str,
    model: str,
    temperature: Optional[float],
    api_max_retries: int,
    api_retry_backoff_seconds: float,
    api_retry_max_backoff_seconds: float,
    api_timeout_seconds: float,
    anthropic_max_output_tokens: int,
    deepseek_base_url: str,
    mistral_base_url: str,
) -> Tuple[str, Dict[str, int], Dict[str, object]]:
    if client is None:
        return (
            "# Dry run placeholder SysMLv2 model",
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            {},
        )
    last_exc: Optional[Exception] = None
    response = None
    for attempt in range(1, api_max_retries + 2):
        try:
            if provider == "openai":
                request_kwargs: Dict[str, object] = {
                    "model": model,
                    "input": prompt,
                    "timeout": api_timeout_seconds,
                }
                if temperature is not None:
                    request_kwargs["temperature"] = temperature
                response = client.responses.create(**request_kwargs)
            elif provider == "anthropic":
                request_kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": anthropic_max_output_tokens,
                    "timeout": api_timeout_seconds,
                }
                if temperature is not None:
                    request_kwargs["temperature"] = temperature
                response = client.messages.create(**request_kwargs)
            elif provider == "deepseek_reasoner":
                request_kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "timeout": api_timeout_seconds,
                }
                if temperature is not None:
                    request_kwargs["temperature"] = temperature
                response = client.chat.completions.create(**request_kwargs)
            elif provider == "mistral_large":
                request_kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "timeout": api_timeout_seconds,
                }
                if temperature is not None:
                    request_kwargs["temperature"] = temperature
                response = client.chat.completions.create(**request_kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            break
        except Exception as exc:
            last_exc = exc
            if attempt > api_max_retries:
                break
            backoff = min(
                api_retry_max_backoff_seconds,
                api_retry_backoff_seconds * (2 ** (attempt - 1)),
            )
            delay = backoff + random.uniform(0.0, 0.5)
            print(
                f"[api:{provider}] attempt {attempt}/{api_max_retries + 1} failed ({exc}); "
                f"retrying in {delay:.2f}s..."
            )
            sleep(delay)
    if response is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{provider} API call failed without an exception.")

    response_payload: Dict[str, object] = {}
    if hasattr(response, "model_dump"):
        response_payload = response.model_dump()
    elif hasattr(response, "to_dict"):
        response_payload = response.to_dict()

    if provider == "openai":
        response_text = sanitize_candidate_text(extract_text_from_response(response))
        usage = getattr(response, "usage", None)

        def usage_value(*names: str) -> int:
            for name in names:
                value = getattr(usage, name, None) if usage else None
                if value is not None:
                    return int(value)
            if usage and hasattr(usage, "model_dump"):
                dump = usage.model_dump()
                for name in names:
                    if name in dump:
                        return int(dump[name])
            return 0

        token_stats = {
            "input_tokens": usage_value("input_tokens", "prompt_tokens"),
            "output_tokens": usage_value("output_tokens", "completion_tokens"),
            "total_tokens": usage_value("total_tokens"),
        }
        if not token_stats["total_tokens"]:
            token_stats["total_tokens"] = (
                token_stats["input_tokens"] + token_stats["output_tokens"]
            )
    elif provider == "anthropic":
        response_text = sanitize_candidate_text(extract_text_from_anthropic_response(response))
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        token_stats = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    elif provider in {"deepseek_reasoner", "mistral_large"}:
        response_text = sanitize_candidate_text(
            extract_text_from_openai_chat_completion_response(response)
        )
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens
        token_stats = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return response_text, token_stats, response_payload


def resolve_python_executable(venv_root: Optional[Path]) -> Path:
    if not venv_root:
        raise ValueError("--venv is required so syside runs inside the correct environment.")
    venv_root = venv_root.resolve()
    python_path = (venv_root / "bin" / "python")
    if not python_path.exists():
        raise FileNotFoundError(
            f"Could not find python at {python_path}. "
            "Ensure --venv points to a valid virtual environment containing syside."
        )
    return python_path


def resolve_syside_command(python_path: Path, venv_root: Path) -> List[str]:
    """Prefer the venv's syside CLI; fall back to python -m syside."""
    venv_root = venv_root.resolve()
    syside_cli = venv_root / "bin" / "syside"
    if syside_cli.exists():
        return [str(syside_cli)]
    return [str(python_path), "-m", "syside"]


def run_syside_check(
    python_path: Path,
    venv_root: Optional[Path],
    model_path: Path,
    timeout_seconds: int,
    validate_with: str,
) -> subprocess.CompletedProcess:
    relative_target = model_path.name
    if validate_with == "check":
        subcmd = ["check", relative_target]
    else:
        subcmd = ["format", relative_target]
    cmd = resolve_syside_command(python_path, venv_root.resolve()) + subcmd
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=model_path.parent,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "").rstrip()
        timeout_note = (
            f"[timeout] syside {validate_with} exceeded {timeout_seconds} seconds."
        )
        stderr = f"{stderr}\n{timeout_note}" if stderr else timeout_note
        return subprocess.CompletedProcess(
            cmd,
            returncode=124,
            stdout=stdout,
            stderr=stderr,
        )


def assert_syside_available(python_path: Path, venv_root: Path, timeout_seconds: int) -> None:
    """Fail fast if syside is not available in the chosen interpreter/venv."""
    venv_root = venv_root.resolve()
    cmd = resolve_syside_command(python_path, venv_root)
    try:
        probe = subprocess.run(
            cmd + ["--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "syside availability probe timed out. "
            f"Interpreter: {python_path}\n"
            f"Command tried: {' '.join(cmd)} --version\n"
            f"Timeout: {timeout_seconds}s\n"
            f"stdout:\n{(exc.stdout or '').strip()}\n"
            f"stderr:\n{(exc.stderr or '').strip()}"
        ) from exc
    if probe.returncode != 0:
        raise RuntimeError(
            "syside is not available in the selected venv/interpreter. "
            f"Interpreter: {python_path}\n"
            f"Command tried: {' '.join(cmd)}\n"
            f"stdout:\n{probe.stdout}\n"
            f"stderr:\n{probe.stderr}"
        )


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    timestamp_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    ensure_dir(timestamp_dir)
    run_start_time = utc_now()
    run_start_wall = perf_counter()

    model_name = args.model
    if args.provider == "anthropic" and model_name == DEFAULT_OPENAI_MODEL:
        model_name = DEFAULT_ANTHROPIC_MODEL
        print(
            f"[config] provider=anthropic and model not set; defaulting to {model_name}"
        )
    if args.provider == "deepseek_reasoner" and model_name == DEFAULT_OPENAI_MODEL:
        model_name = DEFAULT_DEEPSEEK_REASONER_MODEL
        print(
            f"[config] provider=deepseek_reasoner and model not set; defaulting to {model_name}"
        )
    if args.provider == "mistral_large" and model_name == DEFAULT_OPENAI_MODEL:
        model_name = DEFAULT_MISTRAL_LARGE_MODEL
        print(
            f"[config] provider=mistral_large and model not set; defaulting to {model_name}"
        )

    spec_text = load_user_input(args.input)
    example_text = load_example_snippet(args.example)
    client = None
    if not args.dry_run:
        if args.provider == "openai":
            if OpenAI is None:
                raise RuntimeError(
                    "OpenAI provider selected but `openai` package is not installed. "
                    "Install with `pip install openai`."
                )
            client = OpenAI()
        elif args.provider == "anthropic":
            if Anthropic is None:
                raise RuntimeError(
                    "Anthropic provider selected but `anthropic` package is not installed. "
                    "Install with `pip install anthropic`."
                )
            client = Anthropic()
        elif args.provider == "deepseek_reasoner":
            if OpenAI is None:
                raise RuntimeError(
                    "DeepSeek provider selected but `openai` package is not installed. "
                    "Install with `pip install openai`."
                )
            deepseek_api_key = (
                os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("SILICONFLOW_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            if not deepseek_api_key:
                raise RuntimeError(
                    "DeepSeek provider selected but no API key found. "
                    "Set DEEPSEEK_API_KEY (preferred), or SILICONFLOW_API_KEY."
                )
            client = OpenAI(
                api_key=deepseek_api_key,
                base_url=args.deepseek_base_url,
            )
        elif args.provider == "mistral_large":
            if OpenAI is None:
                raise RuntimeError(
                    "Mistral provider selected but `openai` package is not installed. "
                    "Install with `pip install openai`."
                )
            mistral_api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not mistral_api_key:
                raise RuntimeError(
                    "Mistral provider selected but no API key found. "
                    "Set MISTRAL_API_KEY."
                )
            client = OpenAI(
                api_key=mistral_api_key,
                base_url=args.mistral_base_url,
            )
        else:
            raise ValueError(f"Unsupported provider: {args.provider}")

    python_exe = None if args.dry_run else resolve_python_executable(args.venv)
    if not args.dry_run and python_exe is not None:
        assert_syside_available(python_exe, args.venv, args.syside_timeout_seconds)

    run_log: List[Dict[str, object]] = []
    previous_candidate: Optional[str] = None
    compiler_feedback: Optional[str] = None
    tokens_consumed = 0
    start_iteration = 1
    end_iteration = args.max_iters

    if args.resume_source_dir is not None:
        if args.resume_from_iteration < 1:
            raise ValueError("--resume-from-iteration must be >= 1 when resuming.")
        source_dir = args.resume_source_dir.resolve()
        resume_iter = args.resume_from_iteration
        source_sysml = source_dir / f"iteration_{resume_iter:02d}.sysml"
        if not source_sysml.exists():
            raise FileNotFoundError(
                f"Resume source missing expected file: {source_sysml}"
            )
        previous_candidate = source_sysml.read_text(encoding="utf-8")
        additional = (
            args.max_additional_prompts
            if args.max_additional_prompts > 0
            else args.max_iters
        )
        start_iteration = resume_iter + 1
        end_iteration = resume_iter + additional
        print(
            f"[resume] source={source_dir} "
            f"resume_from={resume_iter} additional={additional} "
            f"target_end_iteration={end_iteration}"
        )
        if not args.dry_run and python_exe is not None:
            seed_result = run_syside_check(
                python_exe,
                args.venv,
                source_sysml,
                args.syside_timeout_seconds,
                args.syside_validate_with,
            )
            seed_stdout = seed_result.stdout.strip()
            seed_stderr = seed_result.stderr.strip()
            if seed_result.returncode != 0 and is_infrastructure_compiler_failure(
                seed_stdout, seed_stderr
            ):
                raise RuntimeError(
                    "Infrastructure error while validating resume source iteration; "
                    "refusing to continue.\n"
                    f"stdout:\n{seed_stdout}\n"
                    f"stderr:\n{seed_stderr}"
                )
            compiler_feedback = compact_compiler_feedback(seed_stdout, seed_stderr)
            print(
                f"[resume] seeded compiler feedback from "
                f"{source_sysml.name} (return code {seed_result.returncode})"
            )

    for iteration in range(start_iteration, end_iteration + 1):
        if args.max_total_tokens and tokens_consumed >= args.max_total_tokens:
            print(
                f"[stop] Token budget of {args.max_total_tokens} exhausted "
                f"(~{tokens_consumed} used)."
            )
            break
        print(f"[iter {iteration}] generating proposal...")
        iteration_start_time = utc_now()
        iteration_wall_start = perf_counter()
        prompt = build_prompt(
            spec_text=spec_text,
            iteration=iteration,
            previous_candidate=previous_candidate,
            compiler_feedback=compiler_feedback,
            example_text=example_text,
        )
        (
            candidate_text,
            token_usage,
            raw_response,
        ) = call_model(
            client,
            args.provider,
            prompt,
            model_name,
            args.temperature,
            args.api_max_retries,
            args.api_retry_backoff_seconds,
            args.api_retry_max_backoff_seconds,
            args.api_timeout_seconds,
            args.anthropic_max_output_tokens,
            args.deepseek_base_url,
            args.mistral_base_url,
        )
        prompt_path = timestamp_dir / f"iteration_{iteration:02d}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        sysml_path = timestamp_dir / f"iteration_{iteration:02d}.sysml"
        sysml_path.write_text(candidate_text, encoding="utf-8")
        response_path = timestamp_dir / f"iteration_{iteration:02d}_response.json"
        response_path.write_text(json.dumps(raw_response, indent=2), encoding="utf-8")
        tokens_consumed += token_usage.get("total_tokens", 0)
        previous_candidate = candidate_text

        compile_stdout = ""
        compile_stderr = ""
        success = False
        return_code: Optional[int] = None
        if args.dry_run:
            compile_stdout = "[dry-run] Skipping syside check."
            success = True
        else:
            print(
                f"[iter {iteration}] running 'python -m syside {args.syside_validate_with} "
                f"{sysml_path.name}' "
                f"via {python_exe}..."
            )
            result = run_syside_check(
                python_exe,
                args.venv,
                sysml_path,
                args.syside_timeout_seconds,
                args.syside_validate_with,
            )
            compile_stdout = result.stdout.strip()
            compile_stderr = result.stderr.strip()
            return_code = result.returncode
            success = result.returncode == 0
            if not success and is_infrastructure_compiler_failure(
                compile_stdout, compile_stderr
            ):
                raise RuntimeError(
                    "Infrastructure error while invoking syside; refusing to continue "
                    "or send traceback text back to the model.\n"
                    f"stdout:\n{compile_stdout}\n"
                    f"stderr:\n{compile_stderr}"
                )
            compiler_feedback = compact_compiler_feedback(compile_stdout, compile_stderr)
            print(f"[iter {iteration}] syside return code: {result.returncode}")
            if success:
                print(f"[iter {iteration}] Validation passed.")
            else:
                print(f"[iter {iteration}] Validation NOT passed (continuing).")

        iteration_end_time = utc_now()
        run_log.append(
            {
                "iteration": iteration,
                "iteration_start": iso_utc(iteration_start_time),
                "iteration_end": iso_utc(iteration_end_time),
                "iteration_duration_seconds": perf_counter() - iteration_wall_start,
                "sysml_path": str(sysml_path),
                "prompt_path": str(prompt_path),
                "response_path": str(response_path),
                "success": success,
                "compiler_stdout": compile_stdout,
                "compiler_stderr": compile_stderr,
                "return_code": return_code,
                "tokens_used_this_iter": token_usage,
                "tokens_used_total": tokens_consumed,
                "provider": args.provider,
                "model": model_name,
            }
        )

        if success:
            break

    summary_path = timestamp_dir / "run_log.json"
    summary_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    run_meta = {
        "run_start": iso_utc(run_start_time),
        "run_end": iso_utc(utc_now()),
        "run_duration_seconds": perf_counter() - run_start_wall,
        "iterations_completed": len(run_log),
        "tokens_used_total": tokens_consumed,
        "provider": args.provider,
        "model": model_name,
    }
    (timestamp_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"[done] run details saved to {summary_path}")


if __name__ == "__main__":
    main()
