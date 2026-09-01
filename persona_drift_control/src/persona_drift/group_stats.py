"""Shared groupby-summary shapes used by both analysis_adversarial.py's
y_safety_by_category/y_safety_by_turn and analysis_helpfulness.py's
y_help_by_category/y_help_by_turn diagnostics -- identical pandas idiom in
both, only the value/group column names differ."""

from __future__ import annotations

from typing import Any

import pandas as pd


def summary_by_category(df: pd.DataFrame, value_col: str, category_col: str = "category") -> dict[str, dict[str, float]]:
    return {
        category: {"mean": float(g[value_col].mean()), "sd": float(g[value_col].std())}
        for category, g in df.groupby(category_col)
    }


def summary_by_turn(df: pd.DataFrame, value_col: str, turn_col: str = "turn") -> dict[int, dict[str, Any]]:
    return {
        int(turn): {"mean": float(g[value_col].mean()), "n": int(g[value_col].notna().sum())}
        for turn, g in df.groupby(turn_col)
    }
