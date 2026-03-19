"""
KPI Renderer Component
Renders KPI cards using native Streamlit columns + inline-styled HTML.
Avoids relying on external CSS classes which Streamlit sometimes strips.
"""

import logging
import pandas as pd
import streamlit as st
from utils.helpers import format_number

logger = logging.getLogger("ai_dashboard.kpis")

ACCENT_COLORS = [
    "#6366f1",  # purple
    "#06b6d4",  # cyan
    "#10b981",  # green
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#ec4899",  # pink
    "#8b5cf6",  # violet
    "#14b8a6",  # teal
]


class KPIRenderer:
    """Renders KPI metric cards using Streamlit columns + inline styles."""

    def render(self, kpis: list, df: pd.DataFrame):
        """Render KPI cards in rows of 4."""
        if not kpis:
            st.warning("No KPIs generated.")
            return

        # Recompute live values from the (possibly filtered) dataframe
        computed = []
        for kpi in kpis:
            display_val = self._compute_value(kpi, df)
            computed.append({**kpi, "display_value": display_val})

        # Render in rows of 4
        chunk_size = 4
        for row_start in range(0, len(computed), chunk_size):
            row_kpis = computed[row_start: row_start + chunk_size]
            cols = st.columns(len(row_kpis))
            for col_widget, kpi, idx in zip(cols, row_kpis, range(row_start, row_start + len(row_kpis))):
                with col_widget:
                    self._render_card(kpi, idx)

    # ── Private ────────────────────────────────────────────────

    def _compute_value(self, kpi: dict, df: pd.DataFrame) -> str:
        """Compute display value from filtered dataframe."""
        col = kpi.get("column", "")
        agg = kpi.get("aggregation", "sum")
        fmt = kpi.get("format", "number")

        if col == "_count":
            return format_number(len(df))

        if col and col in df.columns:
            try:
                series = df[col].dropna()
                ops = {
                    "sum":    lambda s: float(s.sum()),
                    "mean":   lambda s: float(s.mean()),
                    "count":  lambda s: int(s.count()),
                    "max":    lambda s: float(s.max()),
                    "min":    lambda s: float(s.min()),
                    "median": lambda s: float(s.median()),
                }
                raw = ops.get(agg, ops["sum"])(series)
                if fmt == "currency":
                    return f"${format_number(raw)}"
                elif fmt == "percent":
                    return f"{raw:.1f}%"
                else:
                    return format_number(raw)
            except Exception:
                pass

        return kpi.get("formatted_value", "N/A")

    def _render_card(self, kpi: dict, index: int):
        """Render one KPI card with fully inline CSS (no external class deps)."""
        accent = ACCENT_COLORS[index % len(ACCENT_COLORS)]
        icon   = kpi.get("icon", "📊")
        label  = kpi.get("label", "Metric")
        value  = kpi.get("display_value", "N/A")
        sub    = kpi.get("sub_label", "")

        st.markdown(f"""
<div style="
    background: linear-gradient(135deg,#16161f 0%,#1a1a27 100%);
    border: 1px solid #2a2a3a;
    border-left: 4px solid {accent};
    border-radius: 12px;
    padding: 1rem 1rem 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
    position: relative;
    min-height: 108px;
    box-sizing: border-box;
">
  <span style="
    position:absolute;top:0.85rem;right:0.9rem;
    font-size:1.35rem;opacity:0.32;
  ">{icon}</span>
  <div style="
    font-size:0.67rem;font-weight:600;
    color:#8888aa;text-transform:uppercase;
    letter-spacing:0.09em;margin-bottom:0.3rem;
  ">{label}</div>
  <div style="
    font-size:1.5rem;font-weight:700;
    color:#f0f0ff;
    font-family:'JetBrains Mono',ui-monospace,monospace;
    line-height:1.15;margin-bottom:0.25rem;
  ">{value}</div>
  <div style="font-size:0.67rem;color:#555570;">{sub}</div>
</div>
""", unsafe_allow_html=True)