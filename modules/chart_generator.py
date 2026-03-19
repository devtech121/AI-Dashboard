"""
Chart Generator Module
Generates Plotly charts based on data profile and AI suggestions
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.chart_generator")

# Explicit chart heights — critical for Streamlit column rendering
CHART_HEIGHT = 420
CHART_HEIGHT_WIDE = 460

# Shared Plotly layout defaults
LAYOUT_DEFAULTS = dict(
    template=AppConfig.CHART_TEMPLATE,
    font_family=AppConfig.CHART_FONT_FAMILY,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,22,31,0.8)",
    height=CHART_HEIGHT,
    margin=dict(l=50, r=30, t=55, b=50),
    title_font_size=15,
    title_font_color="#f0f0ff",
    legend=dict(
        bgcolor="rgba(22,22,31,0.8)",
        bordercolor="rgba(42,42,58,1)",
        borderwidth=1,
        font=dict(size=11),
    ),
    colorway=AppConfig.CHART_COLOR_SEQUENCE,
    xaxis=dict(
        gridcolor="rgba(42,42,58,0.6)",
        tickfont=dict(size=11, color="#8888aa"),
        title_font=dict(size=12, color="#8888aa"),
        linecolor="rgba(42,42,58,0.8)",
    ),
    yaxis=dict(
        gridcolor="rgba(42,42,58,0.6)",
        tickfont=dict(size=11, color="#8888aa"),
        title_font=dict(size=12, color="#8888aa"),
        linecolor="rgba(42,42,58,0.8)",
    ),
)


class ChartGenerator:
    """Generates Plotly chart figures from AI suggestions + rules."""

    @st.cache_data(show_spinner=False)
    def generate(_self, df: pd.DataFrame, profile: dict, ai_analysis: dict) -> list:
        """
        Generate Plotly figures.
        
        Returns list of chart dicts:
          title, description, figure
        """
        ai_charts = ai_analysis.get("charts", [])
        valid_cols = set(df.columns)
        generated = []

        # Process AI-suggested charts
        for chart_def in ai_charts:
            if len(generated) >= AppConfig.MAX_CHARTS:
                break

            x_col = chart_def.get("x")
            y_col = chart_def.get("y")
            chart_type = chart_def.get("type", "bar")

            if x_col and x_col not in valid_cols:
                continue
            if y_col and y_col not in valid_cols:
                continue

            # Override chart type using rules
            chart_type = _self._enforce_chart_rules(df, chart_type, x_col, y_col, profile)

            fig = _self._build_figure(df, chart_type, x_col, y_col, chart_def)
            if fig is not None:
                generated.append({
                    "title": chart_def.get("title", f"{x_col} chart"),
                    "description": chart_def.get("description", ""),
                    "figure": fig,
                    "type": chart_type,
                })

        # Fill with rule-based charts if needed
        if len(generated) < AppConfig.MIN_CHARTS:
            rule_charts = _self._rule_based_charts(df, profile, existing=generated)
            generated.extend(rule_charts[:AppConfig.MAX_CHARTS - len(generated)])

        # Add correlation heatmap if enough numerical columns
        if len(generated) < AppConfig.MAX_CHARTS and len(profile.get("numerical_cols", [])) >= 3:
            heatmap = _self._build_correlation_heatmap(df, profile)
            if heatmap:
                generated.append(heatmap)

        logger.info(f"Generated {len(generated)} charts")
        return generated[:AppConfig.MAX_CHARTS]

    # ── Figure Builders ────────────────────────────────────────

    def _build_figure(self, df, chart_type, x_col, y_col, chart_def):
        """Dispatch to the appropriate chart builder."""
        try:
            builders = {
                "bar": self._build_bar,
                "line": self._build_line,
                "pie": self._build_pie,
                "scatter": self._build_scatter,
                "histogram": self._build_histogram,
            }
            builder = builders.get(chart_type, self._build_bar)
            return builder(df, x_col, y_col, chart_def)
        except Exception as e:
            logger.warning(f"Failed to build {chart_type} chart for {x_col}/{y_col}: {e}")
            return None

    def _build_bar(self, df, x_col, y_col, chart_def):
        title = chart_def.get("title", f"{y_col} by {x_col}")

        if y_col and y_col in df.columns:
            agg_df = (
                df.groupby(x_col)[y_col]
                .sum()
                .reset_index()
                .sort_values(y_col, ascending=False)
                .head(20)
            )
            fig = px.bar(
                agg_df, x=x_col, y=y_col,
                title=title,
                color_discrete_sequence=AppConfig.CHART_COLOR_SEQUENCE,
                labels={
                    x_col: x_col.replace("_", " ").title(),
                    y_col: y_col.replace("_", " ").title()
                }
            )
        else:
            vc = df[x_col].value_counts().head(15).reset_index()
            vc.columns = [x_col, "count"]
            fig = px.bar(
                vc, x=x_col, y="count", title=title,
                color_discrete_sequence=AppConfig.CHART_COLOR_SEQUENCE,
            )

        fig.update_layout(**LAYOUT_DEFAULTS)
        fig.update_traces(marker_line_width=0, marker_color="#6366f1")
        return fig

    def _build_line(self, df, x_col, y_col, chart_def):
        title = chart_def.get("title", f"{y_col} over {x_col}")

        plot_df = df[[x_col, y_col]].dropna().copy() if y_col else df[[x_col]].dropna().copy()
        try:
            plot_df[x_col] = pd.to_datetime(plot_df[x_col])
            plot_df = plot_df.sort_values(x_col)
            if y_col:
                plot_df = plot_df.groupby(x_col)[y_col].mean().reset_index()
        except Exception:
            plot_df = plot_df.sort_values(x_col)

        if y_col:
            fig = px.line(
                plot_df, x=x_col, y=y_col, title=title,
                labels={
                    x_col: x_col.replace("_", " ").title(),
                    y_col: y_col.replace("_", " ").title()
                }
            )
            fig.update_traces(
                line_color="#6366f1",
                line_width=2.5,
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.08)"
            )
        else:
            vc = plot_df[x_col].value_counts().sort_index().reset_index()
            vc.columns = [x_col, "count"]
            fig = px.line(vc, x=x_col, y="count", title=title)
            fig.update_traces(line_color="#6366f1", line_width=2.5)

        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    def _build_pie(self, df, x_col, y_col, chart_def):
        title = chart_def.get("title", f"{x_col} Distribution")
        vc = df[x_col].value_counts().head(8)  # limit to 8 slices max

        fig = px.pie(
            values=vc.values,
            names=vc.index,
            title=title,
            color_discrete_sequence=AppConfig.CHART_COLOR_SEQUENCE,
            hole=0.38,
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            insidetextorientation="radial",
            marker_line_width=2,
            marker_line_color="rgba(10,10,15,0.6)",
            textfont_size=11,
        )
        pie_layout = {**LAYOUT_DEFAULTS}
        pie_layout.pop("xaxis", None)
        pie_layout.pop("yaxis", None)
        fig.update_layout(**pie_layout, showlegend=True)
        return fig

    def _build_scatter(self, df, x_col, y_col, chart_def):
        title = chart_def.get("title", f"{x_col} vs {y_col}")
        if not y_col:
            return None

        plot_df = df[[x_col, y_col]].dropna()
        if len(plot_df) > 2000:
            plot_df = plot_df.sample(2000, random_state=42)

        fig = px.scatter(
            plot_df, x=x_col, y=y_col,
            title=title,
            opacity=0.65,
            labels={
                x_col: x_col.replace("_", " ").title(),
                y_col: y_col.replace("_", " ").title()
            },
            color_discrete_sequence=["#6366f1"],
        )

        # Add manual trendline using numpy (no statsmodels needed)
        try:
            import numpy as np
            z = np.polyfit(plot_df[x_col], plot_df[y_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
            fig.add_scatter(
                x=x_line, y=p(x_line),
                mode="lines",
                line=dict(color="#ef4444", width=2, dash="dash"),
                name="Trend",
                showlegend=True,
            )
        except Exception:
            pass

        fig.update_traces(marker_size=6, selector=dict(mode="markers"))
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    def _build_histogram(self, df, x_col, y_col, chart_def):
        title = chart_def.get("title", f"Distribution of {x_col}")
        series = df[x_col].dropna()

        fig = px.histogram(
            series, x=x_col,
            title=title,
            nbins=min(40, max(10, len(series) // 20)),
            color_discrete_sequence=["#6366f1"],
            marginal="box",
            opacity=0.85,
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="rgba(255,255,255,0.08)")
        fig.update_layout(**LAYOUT_DEFAULTS, showlegend=False, bargap=0.05)
        return fig

    def _build_correlation_heatmap(self, df, profile):
        """Build correlation matrix heatmap."""
        num_cols = profile.get("numerical_cols", [])[:10]
        id_cols = set(profile.get("id_cols", []))
        num_cols = [c for c in num_cols if c not in id_cols]

        if len(num_cols) < 2:
            return None

        try:
            corr = df[num_cols].corr()
            labels = [c.replace("_", " ").title() for c in corr.columns]

            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=labels,
                y=labels,
                colorscale=[[0, "#ef4444"], [0.5, "#1a1a2e"], [1, "#6366f1"]],
                zmid=0,
                zmin=-1, zmax=1,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont_size=10,
                hoverongaps=False,
            ))
            heatmap_layout = {k: v for k, v in LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis")}
            fig.update_layout(
                **heatmap_layout,
                title="Correlation Matrix",
                height=CHART_HEIGHT,
                xaxis=dict(tickfont=dict(size=10, color="#8888aa"), side="bottom"),
                yaxis=dict(tickfont=dict(size=10, color="#8888aa"), autorange="reversed"),
            )
            return {
                "title": "Correlation Matrix",
                "description": "Pairwise correlation between numerical columns",
                "figure": fig,
                "type": "heatmap"
            }
        except Exception as e:
            logger.warning(f"Correlation heatmap failed: {e}")
            return None

    # ── Rule Enforcement ───────────────────────────────────────

    def _enforce_chart_rules(self, df, chart_type, x_col, y_col, profile):
        """Override AI chart type using data-type rules."""
        if not x_col or x_col not in df.columns:
            return chart_type

        x_type = self._get_col_type(x_col, profile)
        y_type = self._get_col_type(y_col, profile) if y_col else None

        # Rule: datetime + numerical → line
        if x_type == "datetime" and y_type == "numerical":
            return "line"

        # Rule: numerical only → histogram
        if x_type == "numerical" and y_col is None:
            return "histogram"

        # Rule: categorical + numerical → bar
        if x_type == "categorical" and y_type == "numerical":
            return "bar"

        # Rule: 2 numerical → scatter
        if x_type == "numerical" and y_type == "numerical":
            return "scatter"

        # Rule: categorical only → pie (if few values) or bar
        if x_type == "categorical" and not y_col:
            n_unique = df[x_col].nunique()
            return "pie" if n_unique <= 8 else "bar"

        return chart_type

    def _get_col_type(self, col, profile):
        """Get column type from profile."""
        for col_meta in profile.get("columns", []):
            if col_meta["name"] == col:
                return col_meta["type"]
        return None

    def _rule_based_charts(self, df, profile, existing):
        """Generate additional rule-based charts to meet minimum count."""
        num_cols = [c for c in profile.get("numerical_cols", []) if c not in profile.get("id_cols", [])]
        cat_cols = profile.get("categorical_cols", [])
        dt_cols = profile.get("datetime_cols", [])
        charts = []

        existing_titles = {c["title"] for c in existing}

        def add_chart(chart_type, x_col, y_col, title, desc):
            if title in existing_titles:
                return
            fig = self._build_figure(df, chart_type, x_col, y_col, {"title": title})
            if fig:
                charts.append({"title": title, "description": desc, "figure": fig, "type": chart_type})
                existing_titles.add(title)

        if cat_cols and num_cols:
            add_chart("bar", cat_cols[0], num_cols[0],
                      f"{num_cols[0].replace('_',' ').title()} by {cat_cols[0].replace('_',' ').title()}",
                      "Category breakdown")

        if dt_cols and num_cols:
            add_chart("line", dt_cols[0], num_cols[0],
                      f"{num_cols[0].replace('_',' ').title()} Over Time",
                      "Time series")

        for col in num_cols[:2]:
            add_chart("histogram", col, None,
                      f"Distribution of {col.replace('_',' ').title()}",
                      "Value distribution")

        for col in cat_cols[:2]:
            add_chart("pie", col, None,
                      f"{col.replace('_',' ').title()} Composition",
                      "Proportional breakdown")

        if len(num_cols) >= 2:
            add_chart("scatter", num_cols[0], num_cols[1],
                      f"{num_cols[0].replace('_',' ').title()} vs {num_cols[1].replace('_',' ').title()}",
                      "Correlation")

        if len(cat_cols) >= 2 and num_cols:
            add_chart("bar", cat_cols[1], num_cols[0],
                      f"{num_cols[0].replace('_',' ').title()} by {cat_cols[1].replace('_',' ').title()}",
                      "Secondary breakdown")

        return charts