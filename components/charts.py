"""
Charts Renderer Component
Renders Plotly charts with explicit heights and card-style containers
"""

import logging
import pandas as pd
import streamlit as st

logger = logging.getLogger("ai_dashboard.charts")

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


class ChartRenderer:
    """Renders Plotly charts with card wrappers and explicit sizing."""

    def render(self, charts: list, df: pd.DataFrame, profile: dict):
        """Render all charts — first chart full width, rest in 2-col grid."""
        if not charts:
            st.warning("No charts generated.")
            return

        # First chart — full width, taller
        self._render_card(charts[0], wide=True)

        # Remaining in 2-column grid
        remaining = charts[1:]
        for i in range(0, len(remaining), 2):
            left, right = st.columns(2, gap="medium")
            with left:
                self._render_card(remaining[i], wide=False)
            if i + 1 < len(remaining):
                with right:
                    self._render_card(remaining[i + 1], wide=False)

    # ── Private ────────────────────────────────────────────────

    def _render_card(self, chart: dict, wide: bool = False):
        """Wrap a chart in a styled card and render it."""
        fig = chart.get("figure")
        if fig is None:
            return

        title = chart.get("title", "")
        description = chart.get("description", "")
        chart_type = chart.get("type", "")

        # Card header
        type_icon = {
            "bar": "📊", "line": "📈", "pie": "🥧",
            "scatter": "🔵", "histogram": "📉", "heatmap": "🌡️",
        }.get(chart_type, "📊")

        st.markdown(f"""
<div style="
    background:linear-gradient(135deg,#16161f 0%,#1a1a27 100%);
    border:1px solid #2a2a3a;
    border-radius:12px;
    padding:1rem 1.2rem 0.6rem 1.2rem;
    margin-bottom:1rem;
">
  <div style="display:flex;align-items:center;gap:0.4rem;margin-bottom:0.15rem;">
    <span style="font-size:0.9rem;">{type_icon}</span>
    <span style="font-size:0.92rem;font-weight:600;color:#f0f0ff;">{title}</span>
  </div>
  {"" if not description else f'<div style="font-size:0.75rem;color:#555570;margin-bottom:0.3rem;">{description}</div>'}
</div>
""", unsafe_allow_html=True)

        # Force explicit height via layout update
        try:
            height = 460 if wide else 400
            fig.update_layout(height=height, margin=dict(l=50, r=30, t=30, b=50))
        except Exception:
            pass

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )