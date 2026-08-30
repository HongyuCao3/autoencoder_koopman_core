import numpy as np

from persona_drift.modeling.dataset import (
    ReducedStateConfig,
    build_identification_dataset,
    build_reduced_state_pairs,
    group_by_trajectory,
    split_by_system_prompt_id,
)


def _row(trajectory_id, system_prompt_id, turn, y_probe, u_remind):
    return {
        "trajectory_id": trajectory_id,
        "system_prompt_id": system_prompt_id,
        "turn": turn,
        "y_probe": y_probe,
        "u_remind": u_remind,
    }


def test_group_by_trajectory_sorts_by_turn():
    rows = [
        _row("t1", "p1", 2, 0.5, 0),
        _row("t1", "p1", 1, 0.9, 0),
        _row("t2", "p1", 1, 0.7, 1),
    ]
    groups = group_by_trajectory(rows)
    assert set(groups) == {"t1", "t2"}
    assert [r["turn"] for r in groups["t1"]] == [1, 2]


def test_split_by_system_prompt_id_is_deterministic_and_disjoint():
    rows = [_row(f"t{i}", f"p{i}", 1, 0.5, 0) for i in range(10)]
    split_a = split_by_system_prompt_id(rows, train_frac=0.7, val_frac=0.15, seed=0)
    split_b = split_by_system_prompt_id(rows, train_frac=0.7, val_frac=0.15, seed=0)
    assert [r["trajectory_id"] for r in split_a["train"]] == [
        r["trajectory_id"] for r in split_b["train"]
    ]
    all_ids = set()
    for part in split_a.values():
        ids = {r["system_prompt_id"] for r in part}
        assert not (ids & all_ids)  # no prompt appears in two splits
        all_ids |= ids
    assert all_ids == {f"p{i}" for i in range(10)}


def test_build_reduced_state_pairs_nu1_mu0_matches_hand_computation():
    traj = [_row("t1", "p1", turn, y, u) for turn, (y, u) in enumerate([(1.0, 0), (0.8, 1), (0.6, 0), (0.9, 1)])]
    pairs = build_reduced_state_pairs(traj, ReducedStateConfig(nu=1, mu=0))
    assert len(pairs) == 3  # turns 0,1,2 each pair with the next turn
    assert np.allclose(pairs[0]["z"], [1.0])
    assert np.allclose(pairs[0]["v"], [0])
    assert pairs[0]["y"] == 1.0
    assert np.allclose(pairs[0]["z_next"], [0.8])
    assert np.allclose(pairs[2]["z"], [0.6])
    assert np.allclose(pairs[2]["z_next"], [0.9])


def test_build_reduced_state_pairs_nu1_mu1_includes_past_input():
    traj = [_row("t1", "p1", turn, y, u) for turn, (y, u) in enumerate([(1.0, 0), (0.8, 1), (0.6, 0), (0.9, 1)])]
    pairs = build_reduced_state_pairs(traj, ReducedStateConfig(nu=1, mu=1))
    # start = max(nu-1, mu) = 1, so the first usable t is turn index 1
    assert len(pairs) == 2
    # z_1 = [y_1, v_0] = [0.8, 0]; v_1 = 1; z_2 = [y_2, v_1] = [0.6, 1]
    assert np.allclose(pairs[0]["z"], [0.8, 0])
    assert np.allclose(pairs[0]["v"], [1])
    assert np.allclose(pairs[0]["z_next"], [0.6, 1])


def test_build_reduced_state_pairs_drops_nan_scored_turns():
    traj = [_row("t1", "p1", turn, y, u) for turn, (y, u) in enumerate([(1.0, 0), (float("nan"), 1), (0.6, 0)])]
    pairs = build_reduced_state_pairs(traj, ReducedStateConfig(nu=1, mu=0))
    assert pairs == []  # every pair here touches the NaN turn


def test_build_identification_dataset_does_not_leak_across_trajectories():
    traj_a = [_row("a", "p1", t, 1.0, 0) for t in range(3)]
    traj_b = [_row("b", "p1", t, 0.0, 1) for t in range(3)]
    dataset = build_identification_dataset(traj_a + traj_b, ReducedStateConfig(nu=1, mu=0))
    assert dataset["Z"].shape[0] == 4  # 2 pairs per trajectory x 2 trajectories
    # every z in the dataset is either all-from-a's constant 1.0 or all-from-b's 0.0
    assert set(dataset["Z"].flatten().tolist()) == {1.0, 0.0}
