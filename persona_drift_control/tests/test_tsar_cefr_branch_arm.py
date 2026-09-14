"""Tests for the depth-4 counterfactual branch arm (GPU-2a).

THE FIRST SUBMISSION DIED ON AN ATTRIBUTE NAME. Job 15896362 burned 2m21s of
A100 time, loaded vLLM, and then raised `'TsarItem' object has no attribute
'source_text'` -- the field is `original`. The --dry-run at the time returned
before it touched the bank, so it passed. A check that passes exactly when it
is not needed is not a check; these tests, and the widened dry run, close that
path without a GPU.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_tsar_cefr_branch_arm.py"
sys.path.insert(0, str(ROOT / "src"))

from persona_drift import tsar_cefr_actions as A          # noqa: E402
from persona_drift import tsar_cefr_bank as bank          # noqa: E402


def load_script():
    spec = importlib.util.spec_from_file_location("branch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arm = load_script()


def an_item(text_id="07-a2", target="A2"):
    """A real TsarItem, so a renamed field fails here instead of on the GPU."""
    return bank.TsarItem(text_id=text_id, source_id=text_id.rpartition("-")[0],
                         original="A source paragraph.", target_cefr=target,
                         reference="A simpler paragraph.")


def test_roots_are_built_from_fields_TsarItem_actually_has():
    roots = arm.build_roots([an_item()], lambda text: 128)
    assert roots[0]["text_out"] == "A source paragraph."
    assert roots[0]["node_id"] == "07-a2|"
    assert roots[0]["depth"] == 0 and roots[0]["parent_id"] is None


def test_root_building_touches_every_field_the_real_run_needs():
    """Names the contract: if TsarItem loses one of these, this test says so."""
    for field in ("text_id", "source_id", "original", "target_cefr"):
        assert hasattr(an_item(), field), field


def test_the_census_matches_the_geometry_it_claims():
    census = arm.tree_census(n_items=199, depth=4)
    assert census["nodes_by_depth"] == {1: 796, 2: 3184, 3: 12736, 4: 50944}
    assert census["n_generations"] == 199 * 340 == 67660
    assert census["n_readout_rows"] == 67660 + 199


def test_a_shallower_tree_is_strictly_cheaper_and_still_well_formed():
    assert arm.tree_census(199, 3)["n_generations"] == 16716
    assert arm.tree_census(199, 3)["n_generations"] < arm.tree_census(199, 4)["n_generations"]


def test_named_exclusion_is_enforced_in_canonical_mode():
    items = [an_item("51-a2"), an_item("51-b1", "B1"), an_item("07-a2")]
    kept, dropped = arm.select_items(items, None, "canonical")
    assert dropped == ["51-a2"]
    assert {i.text_id for i in kept} == {"51-b1", "07-a2"}


def test_canonical_mode_refuses_a_split_missing_the_excluded_item():
    """This arm must stand on the same 199 items as the gate that admitted
    GPU-1; a split without `51-a2` is a different item set, not a free pass."""
    with pytest.raises(SystemExit) as err:
        arm.select_items([an_item("07-a2")], None, "canonical")
    assert "same 199 items" in str(err.value)


def test_the_prompt_renders_for_every_signed_action():
    root = arm.build_roots([an_item()], lambda text: 128)[0]
    for action in A.ACTION_NAMES:
        message = A.user_message(root["text_out"], action, root["target_cefr"])
        assert root["text_out"] in message and message.strip()
