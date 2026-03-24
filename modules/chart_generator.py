"""
Chart Generator Module
Generates Plotly charts based on data profile and AI/verifier suggestions.

Caching is handled by the caller (stored in session state).
The rigid _enforce_chart_rules() override has been removed. The AnalysisVerifier
already corrects chart types before this module runs, so we trust the input.
Only minimal safety checks remain (column existence, None y-axis).
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.chart_generator")

CHART_HEIGHT = 420
CHART_HEIGHT_WIDE = 460

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
    """
    Generates Plotly chart figures from verified AI suggestions.
    No @st.cache_data; caller stores result in session state.
    """

    def generate(self, df: pd.DataFrame, profile: dict, ai_analysis: dict) -> list:
        """
        Build Plotly figures for each chart suggestion in ai_analysis.
        Now supports the 'color' field for multi-entity line/bar charts
        (e.g. stock prices by ticker on a single chart).
        """
        ai_charts = ai_analysis.get("charts", [])
        valid_cols = set(df.columns)
        generated = []

        for chart_def in ai_charts:
            if len(generated) >= AppConfig.MAX_CHARTS:
                break

            x_col  = chart_def.get("x")
            y_col  = chart_def.get("y")
            color_col = chart_def.get("color")
            chart_type = chart_def.get("type", "bar")

            if x_col and x_col not in valid_cols:
                continue
            if y_col and y_col not in valid_cols:
                continue
            if color_col and color_col not in valid_cols:
                color_col = None  # gracefully ignore bad color col

            fig = self._build_figure(df, chart_type, x_col, y_col, chart_def, color_col=color_col)
            if fig is not None:
                generated.append({
                    "title": chart_def.get("title", f"{x_col} chart"),
                    "description": chart_def.get("description", ""),
                    "figure": fig,
                    "type": chart_type,
                })

        if len(generated) < AppConfig.MIN_CHARTS:
            rule_charts = self._rule_based_charts(df, profile, existing=generated)
            generated.extend(rule_charts[:AppConfig.MAX_CHARTS - len(generated)])

        if len(generated) < AppConfig.MAX_CHARTS and len(profile.get("numerical_cols", [])) >= 3:
            heatmap = self._build_correlation_heatmap(df, profile)
            if heatmap:
                generated.append(heatmap)

        if len(generated) < AppConfig.MAX_CHARTS:
            semantic = self._suggest_semantic_charts(df, profile, existing=generated)
            if semantic:
                remaining = AppConfig.MAX_CHARTS - len(generated)
                generated.extend(semantic[:remaining])

        logger.info(f"Generated {len(generated)} charts")
        return generated[:AppConfig.MAX_CHARTS]

    # ── Figure Builders ────────────────────────────────────────

    def _build_figure(self, df, chart_type, x_col, y_col, chart_def, color_col=None):
        """Dispatch to appropriate chart builder."""
        try:
            builders = {
                "bar":       self._build_bar,
                "line":      self._build_line,
                "pie":       self._build_pie,
                "scatter":   self._build_scatter,
                "histogram": self._build_histogram,
            }
            builder = builders.get(chart_type, self._build_bar)
            return builder(df, x_col, y_col, chart_def, color_col=color_col)
        except Exception as e:
            logger.warning(f"Failed {chart_type} for {x_col}/{y_col}: {e}")
            return None

    def _build_bar(self, df, x_col, y_col, chart_def, color_col=None):
        title = chart_def.get("title", f"{y_col} by {x_col}")
        agg_hint = chart_def.get("aggregation", None)

        if y_col and y_col in df.columns:
            y_series = df[y_col].dropna()
            y_min, y_max = y_series.min(), y_series.max()
            is_binary_y = set(y_series.unique()).issubset({0, 1, 0.0, 1.0})

            if is_binary_y:
                agg_df = df.groupby(x_col)[y_col].agg(
                    count="sum", rate=lambda s: round(s.mean() * 100, 1)
                ).reset_index().sort_values("count", ascending=False).head(20)
                fig = px.bar(
                    agg_df, x=x_col, y="count",
                    title=f"{y_col.replace('_',' ').title()} Count by {x_col.replace('_',' ').title()}",
                    text="rate",
                    color_discrete_sequence=AppConfig.CHART_COLOR_SEQUENCE,
                    labels={x_col: x_col.replace("_"," ").title(), "count": f"Count of {y_col}"}
                )
                fig.update_traces(texttemplate="%{text}%", textposition="outside")
            elif agg_hint == "sum" or (y_max - y_min > 1000 and agg_hint != "mean"):
                agg_df = (df.groupby(x_col)[y_col].sum().reset_index()
                         .sort_values(y_col, ascending=False).head(20))
                fig = px.bar(agg_df, x=x_col, y=y_col, title=title,
                            color_discrete_sequence=AppConfig.CHART_COLOR_SEQUENCE,
                            labels={x_col: x_col.replace("_"," ").title(), y_col: y_col.replace("_"," ").title()})
            else:
                agg_df = (df.groupby(x_col)[y_col].mean().round(2).reset_index()
                         .sort_values(y_col, ascending=False).head(20))
                fig = px.bar(agg_df, x=x_col, y=y_col,
                            title=f"Avg {y_col.replace('_',' ').title()} by {x_col.replace('_',' ').title()}",
                            color_discrete_sequence=AppConfig.CHART_COLOR_SEQUENCE,
                            labels={x_col: x_col.replace("_"," ").title(), y_col: f"Avg {y_col.replace('_',' ').title()}"})
        else:
            vc = df[x_col].value_counts().head(15).reset_index()
            vc.columns = [x_col, "count"]
            fig = px.bar(vc, x=x_col, y="count", title=title,
                        color_discrete_sequence=AppConfig.CHART_COLOR_SEQUENCE)

        fig.update_layout(**LAYOUT_DEFAULTS)
        fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
        return fig

    def _build_line(self, df, x_col, y_col, chart_def, color_col=None):
        title = chart_def.get("title", f"{y_col} over {x_col}")

        plot_df = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna().copy() if y_col else df[[x_col]].dropna().copy()

        try:
            plot_df[x_col] = pd.to_datetime(plot_df[x_col])
            plot_df = plot_df.sort_values(x_col)
            # Only aggregate if no color grouping (single series)
            if y_col and not color_col:
                plot_df = plot_df.groupby(x_col)[y_col].mean().reset_index()
        except Exception:
            plot_df = plot_df.sort_values(x_col)

        if y_col:
            if color_col:
                # Multi-entity line chart (e.g. all stock tickers on one chart)
                fig = px.line(
                    plot_df, x=x_col, y=y_col, color=color_col,
                    title=title,
                    color_discrete_sequence=AppConfig.CHART_COLOR_SEQUENCE,
                    labels={
                        x_col: x_col.replace("_"," ").title(),
                        y_col: y_col.replace("_"," ").title(),
                        color_col: color_col.replace("_"," ").title(),
                    }
                )
                fig.update_traces(line_width=1.8)
            else:
                fig = px.line(
                    plot_df, x=x_col, y=y_col, title=title,
                    labels={x_col: x_col.replace("_"," ").title(), y_col: y_col.replace("_"," ").title()}
                )
                fig.update_traces(line_color="#6366f1", line_width=2.5,
                                  fill="tozeroy", fillcolor="rgba(99,102,241,0.08)")
        else:
            vc = plot_df[x_col].value_counts().sort_index().reset_index()
            vc.columns = [x_col, "count"]
            fig = px.line(vc, x=x_col, y="count", title=title)
            fig.update_traces(line_color="#6366f1", line_width=2.5)

        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    def _build_pie(self, df, x_col, y_col, chart_def, color_col=None):
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

    def _build_scatter(self, df, x_col, y_col, chart_def, color_col=None):
        title = chart_def.get("title", f"{x_col} vs {y_col}")
        if not y_col:
            return None

        cols_needed = [x_col, y_col] + ([color_col] if color_col else [])
        plot_df = df[cols_needed].dropna()
        if len(plot_df) > 3000:
            plot_df = plot_df.sample(3000, random_state=42)

        labels = {
            x_col: x_col.replace("_"," ").title(),
            y_col: y_col.replace("_"," ").title(),
        }
        if color_col:
            labels[color_col] = color_col.replace("_"," ").title()

        fig = px.scatter(
            plot_df, x=x_col, y=y_col,
            color=color_col,
            title=title,
            opacity=0.65,
            labels=labels,
            color_discrete_sequence=AppConfig.CHART_COLOR_SEQUENCE,
        )

        # Add trendline only for single-series (no color grouping)
        if not color_col:
            try:
                z = np.polyfit(plot_df[x_col], plot_df[y_col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
                fig.add_scatter(
                    x=x_line, y=p(x_line), mode="lines",
                    line=dict(color="#ef4444", width=2, dash="dash"),
                    name="Trend", showlegend=True,
                )
            except Exception:
                pass
        else:
            fig.update_traces(marker_size=5)

        fig.update_traces(marker_size=6, selector=dict(mode="markers"))
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    def _build_histogram(self, df, x_col, y_col, chart_def, color_col=None):
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

    def _suggest_semantic_charts(self, df, profile, existing):
        """Suggest extra charts from correlations and seasonality."""
        charts = []
        existing_keys = {
            (c.get("type"), c.get("x"), c.get("y"), c.get("color"))
            for c in existing if isinstance(c, dict)
        }

        num_cols = profile.get("numerical_cols", [])
        dt_cols = profile.get("datetime_cols", [])

        # Correlation-based scatter charts
        corr = profile.get("correlation_matrix") or {}
        pairs = []
        for c1 in num_cols[:10]:
            for c2 in num_cols[:10]:
                if c1 >= c2:
                    continue
                try:
                    r = (corr.get(c1) or {}).get(c2, 0) or 0
                    pairs.append((abs(r), r, c1, c2))
                except Exception:
                    continue
        pairs.sort(reverse=True, key=lambda x: x[0])
        for _, r, c1, c2 in pairs[:3]:
            key = ("scatter", c1, c2, None)
            if key in existing_keys:
                continue
            chart_def = {
                "type": "scatter",
                "x": c1,
                "y": c2,
                "title": f"Correlation: {c1.replace('_',' ').title()} vs {c2.replace('_',' ').title()}",
                "description": f"Correlation coefficient r={r:.2f}",
            }
            fig = self._build_scatter(df, c1, c2, chart_def, color_col=None)
            if fig:
                charts.append({
                    "title": chart_def["title"],
                    "description": chart_def["description"],
                    "figure": fig,
                    "type": "scatter",
                })
                existing_keys.add(key)

        # Seasonality: month trend for first datetime + first numeric
        if dt_cols and num_cols:
            x_col = dt_cols[0]
            y_col = num_cols[0]
            key = ("line", "month", y_col, None)
            if key not in existing_keys:
                try:
                    plot_df = df[[x_col, y_col]].dropna().copy()
                    plot_df[x_col] = pd.to_datetime(plot_df[x_col])
                    plot_df["month"] = plot_df[x_col].dt.to_period("M").astype(str)
                    agg_df = plot_df.groupby("month")[y_col].mean().reset_index()
                    fig = px.line(
                        agg_df, x="month", y=y_col,
                        title=f"Seasonality: Avg {y_col.replace('_',' ').title()} by Month",
                        labels={"month": "Month", y_col: f"Avg {y_col.replace('_',' ').title()}"},
                        color_discrete_sequence=AppConfig.CHART_COLOR_SEQUENCE,
                    )
                    fig.update_layout(**LAYOUT_DEFAULTS)
                    charts.append({
                        "title": f"Seasonality: Avg {y_col.replace('_',' ').title()} by Month",
                        "description": "Monthly seasonality trend",
                        "figure": fig,
                        "type": "line",
                    })
                except Exception:
                    pass

        return charts

    def _rule_based_charts(self, df, profile, existing):
        """Generate rule-based charts to fill gaps. Uses color for multi-entity series."""
        num_cols = [c for c in profile.get("numerical_cols", []) if c not in profile.get("id_cols", [])]
        cat_cols = profile.get("categorical_cols", [])
        dt_cols  = profile.get("datetime_cols", [])
        charts   = []
        existing_titles = {c["title"] for c in existing}

        def add(ctype, x, y, title, desc, color=None):
            if title in existing_titles:
                return
            fig = self._build_figure(df, ctype, x, y, {"title": title, "aggregation": None}, color_col=color)
            if fig:
                charts.append({"title": title, "description": desc, "figure": fig, "type": ctype})
                existing_titles.add(title)

        # Time series — use color grouping if a sensible categorical column exists
        if dt_cols and num_cols:
            color_col = None
            for c in cat_cols:
                n_unique = df[c].nunique() if c in df.columns else 999
                if 2 <= n_unique <= 15:
                    color_col = c
                    break
            add("line", dt_cols[0], num_cols[0],
                f"{num_cols[0].replace('_',' ').title()} Over Time",
                "Time series trend", color=color_col)

        if cat_cols and num_cols:
            add("bar", cat_cols[0], num_cols[0],
                f"{num_cols[0].replace('_',' ').title()} by {cat_cols[0].replace('_',' ').title()}",
                "Category breakdown")

        for col in num_cols[:2]:
            add("histogram", col, None, f"Distribution of {col.replace('_',' ').title()}", "Distribution")

        for col in cat_cols[:2]:
            add("pie", col, None, f"{col.replace('_',' ').title()} Composition", "Proportional breakdown")

        if len(num_cols) >= 2:
            add("scatter", num_cols[0], num_cols[1],
                f"{num_cols[0].replace('_',' ').title()} vs {num_cols[1].replace('_',' ').title()}",
                "Correlation")

        if len(cat_cols) >= 2 and num_cols:
            add("bar", cat_cols[1], num_cols[0],
                f"{num_cols[0].replace('_',' ').title()} by {cat_cols[1].replace('_',' ').title()}",
                "Secondary breakdown")

        return charts

