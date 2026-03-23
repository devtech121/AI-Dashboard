"""
Dashboard Component
Renders insights, anomalies, and data explorer sections
"""

import logging
import pandas as pd
import streamlit as st

logger = logging.getLogger("ai_dashboard.dashboard")


class DashboardRenderer:
    """Renders the main dashboard sections."""

    @staticmethod
    def render_insights(insights: list, anomalies: dict):
        """Render AI insights and anomaly alerts."""
        if not insights and not anomalies:
            st.info("No insights generated yet.")
            return

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 💡 Key Insights")
            if insights:
                for insight in insights[:8]:
                    st.markdown(f"""
<div style="
    background:#16161f;
    border:1px solid #2a2a3a;
    border-left:3px solid #6366f1;
    border-radius:10px;
    padding:0.9rem 1.1rem;
    margin:0.4rem 0;
    font-size:0.88rem;
    color:#8888aa;
    line-height:1.6;
">📌 {insight}</div>
""", unsafe_allow_html=True)
            else:
                st.info("No AI insights available. Check your Groq API key in the sidebar.")

        with col2:
            st.markdown("#### 🚨 Anomalies Detected")
            if anomalies:
                for col_name, info in list(anomalies.items())[:5]:
                    st.markdown(f"""
<div style="
    background:rgba(239,68,68,0.06);
    border:1px solid rgba(239,68,68,0.22);
    border-radius:10px;
    padding:0.65rem 0.9rem;
    margin:0.25rem 0;
    font-size:0.84rem;
    color:#fca5a5;
"><strong>{col_name}</strong><br>{info['n_anomalies']:,} outliers ({info['pct']:.1f}%)</div>
""", unsafe_allow_html=True)
            else:
                st.success("✅ No anomalies detected")

    @staticmethod
    def render_data_explorer(df: pd.DataFrame, profile: dict):
        """Render interactive data exploration section."""
        tabs = st.tabs(["📊 Data Table", "📈 Summary Stats", "🔍 Column Details", "🔗 Correlations"])

        with tabs[0]:
            # Data table with search
            search = st.text_input("🔎 Search in data", placeholder="Type to filter rows...", key="data_search")
            display_df = df
            if search:
                mask = df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
                display_df = df[mask]
                st.caption(f"Found {len(display_df):,} matching rows")

            st.dataframe(
                display_df.head(1000),
                use_container_width=True,
                height=400
            )
            if len(df) > 1000:
                st.caption(f"Showing first 1,000 of {len(df):,} rows")

        with tabs[1]:
            num_cols = profile.get("numerical_cols", [])
            if num_cols:
                stats = df[num_cols].describe().round(4)
                st.dataframe(stats, use_container_width=True, height=350)
            else:
                st.info("No numerical columns for summary statistics.")

        with tabs[2]:
            col_names = [c["name"] for c in profile.get("columns", [])]
            selected_col = st.selectbox("Select column", col_names, key="col_detail_select")

            if selected_col:
                col_meta = next(
                    (c for c in profile.get("columns", []) if c["name"] == selected_col),
                    None
                )
                if col_meta:
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.metric("Type", col_meta["type"])
                        st.metric("Unique Values", f"{col_meta['n_unique']:,}")
                        st.metric("Missing", f"{col_meta['n_null']:,} ({col_meta['null_pct']}%)")
                    with detail_col2:
                        if col_meta["type"] == "numerical":
                            st.metric("Mean", f"{col_meta.get('mean', 0):,.2f}")
                            st.metric("Min", f"{col_meta.get('min', 0):,.2f}")
                            st.metric("Max", f"{col_meta.get('max', 0):,.2f}")
                        elif col_meta["type"] == "categorical":
                            st.metric("Mode", str(col_meta.get("mode", "N/A")))
                            top_vals = col_meta.get("top_values", {})
                            if top_vals:
                                top_df = pd.DataFrame(
                                    list(top_vals.items()),
                                    columns=["Value", "Count"]
                                ).head(10)
                                st.dataframe(top_df, use_container_width=True, hide_index=True)

        with tabs[3]:
            num_cols = [
                c for c in profile.get("numerical_cols", [])
                if c not in profile.get("id_cols", [])
            ]
            if len(num_cols) >= 2:
                corr = df[num_cols].corr().round(3)
                st.dataframe(
                    corr.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                    use_container_width=True,
                    height=350
                )
            else:
                st.info("At least 2 numerical columns required for correlation analysis.")