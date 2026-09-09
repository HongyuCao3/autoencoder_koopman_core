# Deliberately does not import .chat_model / .selfchat / .screening here:
# those pull in torch/transformers at import time, which offline tooling
# (e.g. tests/test_prompt_bank.py) and the prompt bank itself should not
# require. Import the submodule directly for those, e.g.
# `from persona_drift.screening import run_screening`.
#
# 2026-09-09: the three re-exports below are now LAZY (PEP 562), for the same
# reason the imports above were left out. `from .analysis import ...` pulled
# pandas at package-import time, so `import persona_drift.sequor_bank` was
# impossible in the `constraint` line's vLLM environment
# (docs/experiments/constraint_retention_plan.md section 8), which has vLLM and
# nothing else on purpose. The names resolve exactly as before on first
# attribute access -- tests/test_package_lazy_reexports.py pins that, including
# that they are the same objects as the submodules' own.
import importlib

_LAZY = {
    "analyze_screening": ".analysis",
    "PromptEntry": ".prompt_bank",
    "load_prompt_bank": ".prompt_bank",
    "score_response": ".prompt_bank",
    "select_screening_prompts": ".prompt_bank",
    "build_reminder_text": ".reminder",
}

__all__ = [
    "analyze_screening",
    "PromptEntry",
    "load_prompt_bank",
    "score_response",
    "select_screening_prompts",
    "build_reminder_text",
]


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])
