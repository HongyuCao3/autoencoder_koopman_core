#!/usr/bin/env python3
"""Task-scoped Hydra entrypoint for the ARX / richer_abs_sign ridge fit
(same fitting code as scripts/fit_koopman_defense_model.py, which keeps
working unchanged for the already-validated Phase A-I sbatch pipeline --
this is an additive entrypoint for new fits, not a replacement).

Parameters come from conf/task/<name>.yaml's `fit` block instead of argparse
flags, so switching `task=defense`/`task=persona_drift`/`task=sycophancy`
switches the *entire* parameter set at once and it's impossible to
accidentally mix, say, defense's validated nu=1,mu=2 into a persona-drift
fit by forgetting to pass one flag. Hydra also writes the fully-resolved
config (including whatever was overridden on the command line) to
<run_dir>/.hydra/{config.yaml,overrides.yaml} automatically -- no extra
logging code needed.

Usage:
    python scripts/fit_koopman_hydra.py                          # task=defense (default)
    python scripts/fit_koopman_hydra.py task.fit.mu=1
    python scripts/fit_koopman_hydra.py task=persona_drift task.fit.rows_path=outputs/foo/trajectories.jsonl

CPU-only, pure numpy/pandas -- no GPU needed. Run directly (no sbatch).
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from persona_drift.modeling.content_similarity import (  # noqa: E402
    annotate_similarity,
    fit_tfidf_corpus,
    reference_texts_excluding_ids,
)
from persona_drift.modeling.dataset import (  # noqa: E402
    ReducedStateConfig,
    build_identification_dataset,
    load_trajectories,
    split_by_system_prompt_id,
)
from persona_drift.modeling.evaluate import one_step_error, rollout_output_error  # noqa: E402
from persona_drift.modeling.koopman import (  # noqa: E402
    KoopmanSurrogate,
    abs_sign_extra_features,
    no_extra_features,
)

CONFIG_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "conf")


def _fit_and_evaluate(name, extra_features_fn, train_rows, held_out_rows, config, ridge, y_col):
    train_dataset = build_identification_dataset(train_rows, config, y_col=y_col)
    model = KoopmanSurrogate(extra_features_fn=extra_features_fn, ridge=ridge).fit(train_dataset)
    return {
        "name": name,
        "A": model.A.tolist(),
        "B": model.B.tolist(),
        "b": model.b.tolist(),
        "C": model.C.tolist(),
        "train_one_step_mse": one_step_error(model, train_dataset),
        "held_out_rollout_mse": rollout_output_error(model, held_out_rows, config, y_col=y_col),
    }, model


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="fit_koopman")
def main(cfg: DictConfig) -> None:
    fit = cfg.task.fit
    out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
    rows = load_trajectories(fit.rows_path)

    split = split_by_system_prompt_id(
        rows,
        train_frac=1.0 - fit.held_out_frac,
        val_frac=0.0,
        seed=fit.split_seed,
        split_col=fit.split_col,
    )
    train_rows = split["train"]
    held_out_rows = split["test"]
    n_train = len({r[fit.split_col] for r in train_rows})
    held_out_ids = sorted({r[fit.split_col] for r in held_out_rows})

    aux_cols = list(fit.aux_cols)
    reference_texts = None
    if "attack_similarity" in aux_cols:
        reference_texts = reference_texts_excluding_ids(
            train_rows, exclude_ids=set(held_out_ids), text_col="attacker_query", id_col=fit.split_col
        )
        corpus = fit_tfidf_corpus(reference_texts)
        train_rows = annotate_similarity(train_rows, "attacker_query", corpus, out_col="attack_similarity")
        held_out_rows = annotate_similarity(held_out_rows, "attacker_query", corpus, out_col="attack_similarity")

    state_config = ReducedStateConfig(
        nu=fit.nu, mu=fit.mu, aux_cols=tuple(aux_cols), contemporaneous_v=fit.contemporaneous_v
    )

    arx_report, arx_model = _fit_and_evaluate(
        "arx", no_extra_features, train_rows, held_out_rows, state_config, fit.ridge, fit.y_col
    )
    richer_report, _ = _fit_and_evaluate(
        "richer_abs_sign", abs_sign_extra_features, train_rows, held_out_rows, state_config, fit.ridge, fit.y_col
    )
    controllability = arx_model.controllability(fit.controllability_horizon)

    report = {
        "task": cfg.task.name,
        "config": OmegaConf.to_container(fit, resolve=True),
        "n_train": n_train,
        "n_held_out": len(held_out_ids),
        "held_out_ids": held_out_ids,
        "content_reference_texts": reference_texts,
        "arx": arx_report,
        "richer_abs_sign": richer_report,
        "controllability_arx": controllability,
    }
    out_path = out_dir / "koopman_fit_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"task={cfg.task.name} n_train={n_train} n_held_out={len(held_out_ids)}")
    print(f"ARX held_out_rollout_mse={arx_report['held_out_rollout_mse']:.6f}")
    print(f"richer_abs_sign held_out_rollout_mse={richer_report['held_out_rollout_mse']:.6f}")
    print(f"controllability_rank={controllability['controllability_rank']} (state_dim={arx_model.state_dim})")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
