"""
KPI Generator Module
Generates Key Performance Indicators from dataset + AI suggestions
"""

import logging
import numpy as np
import pandas as pd
import streamlit as st
from utils.config import AppConfig
from utils.helpers import is_id_column, format_number

logger = logging.getLogger("ai_dashboard.kpi_generator")


class KPIGenerator:
    """Generates KPI metrics combining AI suggestions with rule-based logic."""

    COLORS = ["purple", "blue", "green", "orange", "red", "pink", "purple", "blue"]
    ICONS = {
        "sum": "💰", "mean": "📊", "count": "🔢",
        "max": "⬆️", "min": "⬇️", "default": "📈"
    }

    def generate(self, df: pd.DataFrame, profile: dict, ai_analysis: dict) -> list:
        """
        Generate final KPI list.
        
        Returns list of KPI dicts:
          label, value, formatted_value, sub_label, color, icon
        """
        # Start with AI-suggested KPIs
        ai_kpis = ai_analysis.get("kpis", [])
        kpis = []

        valid_cols = set(df.columns)
        id_cols = set(profile.get("id_cols", []))
        num_cols = profile.get("numerical_cols", [])

        # Process AI-suggested KPIs
        for kpi_def in ai_kpis:
            col = kpi_def.get("column", "")
            if col not in valid_cols or col in id_cols:
                continue
            if col not in df.select_dtypes(include="number").columns:
                continue

            kpi = self._compute_kpi(df, kpi_def)
            if kpi:
                kpis.append(kpi)

        # Fill gaps with rule-based KPIs
        existing_cols = {k["column"] for k in kpis}
        remaining_cols = [c for c in num_cols if c not in existing_cols and c not in id_cols]

        for col in remaining_cols:
            if len(kpis) >= AppConfig.MAX_KPIS:
                break

            kpi_def = self._infer_kpi_type(col)
            kpi = self._compute_kpi(df, {**kpi_def, "column": col})
            if kpi:
                kpis.append(kpi)

        # Always add row count
        if len(kpis) < AppConfig.MAX_KPIS:
            kpis.append({
                "label": "Total Records",
                "column": "_count",
                "value": len(df),
                "formatted_value": format_number(len(df)),
                "sub_label": "rows in dataset",
                "color": self.COLORS[len(kpis) % len(self.COLORS)],
                "icon": "🗃️",
                "aggregation": "count",
            })

        # Assign colors and icons
        for i, kpi in enumerate(kpis):
            kpi["color"] = self.COLORS[i % len(self.COLORS)]
            if "icon" not in kpi:
                agg = kpi.get("aggregation", "default")
                kpi["icon"] = self.ICONS.get(agg, self.ICONS["default"])

        logger.info(f"Generated {len(kpis)} KPIs")
        return kpis[:AppConfig.MAX_KPIS]

    # ── Private Methods ────────────────────────────────────────

    def _compute_kpi(self, df: pd.DataFrame, kpi_def: dict) -> dict | None:
        """Compute a single KPI value."""
        col = kpi_def.get("column", "")
        label = kpi_def.get("label", col.replace("_", " ").title())
        agg = kpi_def.get("aggregation", "sum")
        fmt = kpi_def.get("format", "number")

        if col not in df.columns:
            return None

        series = df[col].dropna()
        if len(series) == 0:
            return None

        try:
            if agg == "sum":
                value = float(series.sum())
                sub = f"sum of {len(series):,} values"
            elif agg == "mean":
                value = float(series.mean())
                sub = f"average across {len(series):,} records"
            elif agg == "count":
                value = int(series.count())
                sub = f"non-null records"
            elif agg == "max":
                value = float(series.max())
                sub = f"maximum value"
            elif agg == "min":
                value = float(series.min())
                sub = f"minimum value"
            elif agg == "median":
                value = float(series.median())
                sub = f"median value"
            else:
                value = float(series.sum())
                sub = ""
        except Exception as e:
            logger.warning(f"Failed to compute KPI for {col}: {e}")
            return None

        # Format value
        if fmt == "currency":
            formatted = f"${format_number(value)}"
        elif fmt == "percent":
            formatted = f"{value:.1f}%"
        else:
            formatted = format_number(value)

        return {
            "label": label,
            "column": col,
            "value": value,
            "formatted_value": formatted,
            "sub_label": sub,
            "aggregation": agg,
            "format": fmt,
        }

    def _infer_kpi_type(self, col: str) -> dict:
        """Infer KPI aggregation from column name."""
        col_lower = col.lower()
        currency_keywords = ["price", "revenue", "sales", "amount", "cost", "profit", "income", "spend"]
        rate_keywords = ["rate", "ratio", "pct", "percent", "score"]
        count_keywords = ["count", "qty", "quantity", "num", "number"]

        if any(k in col_lower for k in currency_keywords):
            return {"aggregation": "sum", "format": "currency"}
        elif any(k in col_lower for k in rate_keywords):
            return {"aggregation": "mean", "format": "percent"}
        elif any(k in col_lower for k in count_keywords):
            return {"aggregation": "sum", "format": "number"}
        else:
            return {"aggregation": "mean", "format": "number"}
