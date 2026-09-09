"""Regression test for making persona_drift/__init__.py's re-exports lazy.

The change exists so `import persona_drift.sequor_bank` works in an
environment without pandas (the `constraint` line's vLLM env, plan section 8).
Everything else about the package's surface must be unchanged: same names,
same objects, same `__all__`, and `from persona_drift import X` still working
for every X that worked before.
"""

from __future__ import annotations

import importlib

import persona_drift

EAGER_NAMES = [
    "analyze_screening",
    "PromptEntry",
    "load_prompt_bank",
    "score_response",
    "select_screening_prompts",
    "build_reminder_text",
]


def test_all_is_unchanged():
    assert persona_drift.__all__ == EAGER_NAMES


def test_every_reexport_resolves_to_its_submodule_object():
    for name, module_name in (
        ("analyze_screening", "persona_drift.analysis"),
        ("PromptEntry", "persona_drift.prompt_bank"),
        ("load_prompt_bank", "persona_drift.prompt_bank"),
        ("score_response", "persona_drift.prompt_bank"),
        ("select_screening_prompts", "persona_drift.prompt_bank"),
        ("build_reminder_text", "persona_drift.reminder"),
    ):
        assert getattr(persona_drift, name) is getattr(importlib.import_module(module_name), name)


def test_from_import_still_works():
    from persona_drift import analyze_screening, build_reminder_text  # noqa: F401


def test_unknown_attribute_still_raises_attribute_error():
    try:
        persona_drift.no_such_name
    except AttributeError as exc:
        assert "no_such_name" in str(exc)
    else:
        raise AssertionError("expected AttributeError")


def test_dir_lists_the_reexports():
    assert set(EAGER_NAMES) <= set(dir(persona_drift))


def test_importing_a_light_submodule_does_not_drag_in_pandas():
    """The point of the change. `sys.modules` is checked rather than the import
    working, because pandas IS installed in the test environment -- the failure
    being prevented is the package importing it at all."""

    import subprocess
    import sys

    probe = (
        "import sys; import persona_drift.sequor_bank; "
        "print('pandas' in sys.modules, 'torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"
