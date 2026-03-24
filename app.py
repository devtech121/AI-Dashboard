"""
AI Data Analyst Dashboard - Main Application Entry Point
Streamlit-based analytics dashboard with AI-powered insights
"""

import streamlit as st
import pandas as pd
import logging
import traceback
from pathlib import Path

# Page configuration - MUST be first Streamlit call
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": "https://github.com",
        "About": "AI-powered analytics dashboard system"
    }
)

# Internal imports after page config
from utils.config import AppConfig
from utils.helpers import setup_logging, apply_custom_css
from modules.data_loader import DataLoader
from modules.profiler import DataProfiler
from modules.ai_engine import AIEngine
from modules.analysis_verifier import AnalysisVerifier
from modules.kpi_generator import KPIGenerator
from modules.chart_generator import ChartGenerator
from modules.insights_generator import InsightsGenerator
from modules.anomaly_detector import AnomalyDetector
from modules.chat_engine import ChatEngine
from components.dashboard import DashboardRenderer
from components.kpis import KPIRenderer
from components.charts import ChartRenderer

# Initialize logging
logger = setup_logging()


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        "df": None,
        "profile": None,
        "ai_analysis": None,
        "kpis": None,
        "charts": None,
        "insights": None,
        "anomalies": None,
        "chat_history": [],
        "file_processed": False,
        "processing_error": None,
        "selected_model": AppConfig.DEFAULT_MODEL,
        "sample_data_consent": False,
        "ai_analysis_cache": {},
        "custom_charts": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header():
    """Render the application header."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#16161f 0%,#1a1a27 100%);
        border: 1px solid #2a2a3a;
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    ">
        <div style="position:relative;z-index:1;">
            <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.4rem;">
                <span style="font-size:2rem;">🧠</span>
                <h1 style="
                    font-size:2rem;font-weight:700;margin:0;padding:0;
                    background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 50%,#06b6d4 100%);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;
                ">AI Data Analyst</h1>
            </div>
            <p style="color:#8888aa;font-size:1rem;margin:0;font-weight:400;">
                Upload any CSV &bull; Get instant AI-powered insights &bull; Chat with your data
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(df: pd.DataFrame = None):
    """Render sidebar with settings and filters."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # Model selection
        model_options = list(AppConfig.MODELS.keys())
        selected = st.selectbox(
            "AI Model",
            model_options,
            index=model_options.index(st.session_state.selected_model)
            if st.session_state.selected_model in model_options else 0,
            format_func=lambda m: AppConfig.MODELS.get(m, {}).get("name", m),
            help="Select AI model for analysis. Falls back automatically if unavailable."
        )
        st.session_state.selected_model = selected

        # Groq API Key input
        # Never pre-fill the value — keeps the key hidden even from the eye icon
        groq_key = st.text_input(
            "Groq API Key (optional)",
            value="",
            type="password",
            placeholder="Already set via Secrets" if AppConfig.get_groq_key() else "gsk_...",
            help="Only needed if not set in Streamlit Secrets. Get key at console.groq.com"
        )
        if groq_key:
            AppConfig.set_groq_key(groq_key)

        # Show status indicator instead of the key itself
        if AppConfig.get_groq_key():
            st.success("✅ API key active", icon="🔑")
        else:
            st.warning("⚠️ No API key set", icon="🔑")

        # Claude API Key input (Anthropic)
        claude_key = st.text_input(
            "Claude API Key (optional)",
            value="",
            type="password",
            placeholder="Already set via Secrets" if AppConfig.get_claude_key() else "sk-ant-...",
            help="Only needed if not set in Streamlit Secrets. Get key at console.anthropic.com"
        )
        if claude_key:
            AppConfig.set_claude_key(claude_key)

        if AppConfig.get_claude_key():
            st.success("Claude key active", icon="🔑")
        else:
            st.warning("No Claude key set", icon="🔑")

        # User consent gate for sending sample rows to the LLM
        st.markdown("---")
        st.markdown("### 🔒 Data Privacy")
        consent = st.checkbox(
            "Send sample rows to AI for richer insights",
            value=st.session_state.get("sample_data_consent", False),
            help=(
                "When enabled, up to 2 sample rows from your dataset are included "
                "in the prompt sent to Groq. Disabled by default to protect your data."
            ),
            key="sample_data_consent_widget"
        )
        st.session_state.sample_data_consent = consent
        if consent:
            st.caption("⚠️ Sample rows will be sent to Groq's API.")

        st.markdown("---")
        st.markdown("### 📁 Sample Datasets")

        sample_datasets = {
            "🛒 E-commerce Sales": "sample_data/ecommerce.csv",
            "👥 HR Analytics": "sample_data/hr_data.csv",
            "📈 Stock Prices": "sample_data/stocks.csv",
        }

        for name, path in sample_datasets.items():
            if st.button(name, key=f"sample_{name}", use_container_width=True):
                full_path = Path(__file__).parent / path
                if full_path.exists():
                    st.session_state["load_sample"] = str(full_path)
                    st.rerun()

        st.markdown("---")

        # Active filters section (shown when data is loaded)
        if df is not None:
            st.markdown("### 🔍 Data Filters")
            render_filters(df)

        st.markdown("---")
        st.markdown("### 📊 Data Info")
        if df is not None:
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", len(df.columns))
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        else:
            st.info("Upload a CSV to see data info")


def render_filters(df: pd.DataFrame):
    """Render dynamic filters based on data columns."""
    if "active_filters" not in st.session_state:
        st.session_state.active_filters = {}

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    filters_applied = {}

    # Categorical filters (first 3)
    for col in categorical_cols[:3]:
        unique_vals = df[col].dropna().unique().tolist()
        if 2 <= len(unique_vals) <= 50:
            selected_vals = st.multiselect(
                f"{col}",
                options=unique_vals,
                default=[],
                key=f"filter_{col}"
            )
            if selected_vals:
                filters_applied[col] = selected_vals

    # Numeric range filters (first 2) with guardrails for empty/NaN columns
    for col in numeric_cols[:2]:
        series = df[col].dropna()
        if len(series) == 0:
            continue  # skip fully-NaN columns — st.slider would crash
        col_min = float(series.min())
        col_max = float(series.max())
        if col_min == col_max or not (col_min == col_min) or not (col_max == col_max):
            continue  # skip if range is zero or values are still NaN after dropna
        try:
            range_val = st.slider(
                f"{col} range",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
                key=f"range_{col}"
            )
            if range_val != (col_min, col_max):
                filters_applied[col] = range_val
        except Exception as e:
            logger.warning(f"Slider failed for column '{col}': {e}")

    st.session_state.active_filters = filters_applied


def compute_df_hash(df: pd.DataFrame) -> str:
    """Compute a stable hash for a dataframe (values + columns)."""
    import hashlib
    import pandas as pd
    h = hashlib.sha256()
    try:
        values_hash = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        h.update(values_hash)
    except Exception:
        h.update(df.to_csv(index=True).encode("utf-8"))
    h.update("|".join([str(c) for c in df.columns]).encode("utf-8"))
    return h.hexdigest()


def extract_columns_from_text(text: str, columns: list) -> list:
    if not text:
        return []
    import re
    found = []
    lower = text.lower()
    for col in columns:
        c = str(col)
        # Word-boundary match first, fallback to substring
        pattern = r"\b" + re.escape(c.lower()) + r"\b"
        if re.search(pattern, lower) or c.lower() in lower:
            found.append(c)
    return found


def extract_json_block(text: str) -> dict | None:
    if not text:
        return None
    import json, re
    try:
        return json.loads(text)
    except Exception:
        pass
    for pat in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return None
    return None


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply active filters to the dataframe."""
    filtered_df = df.copy()
    filters = st.session_state.get("active_filters", {})

    for col, filter_val in filters.items():
        if col not in filtered_df.columns:
            continue
        if isinstance(filter_val, list):
            filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
        elif isinstance(filter_val, tuple) and len(filter_val) == 2:
            filtered_df = filtered_df[
                (filtered_df[col] >= filter_val[0]) &
                (filtered_df[col] <= filter_val[1])
            ]

    return filtered_df


def process_uploaded_file(uploaded_file):
    """
    Process uploaded CSV file and run full AI pipeline.

    Results are stored in st.session_state (per-session scope) to avoid
    cross-user data leakage in shared deployments.
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Load & preprocess
        status_text.text("📂 Loading and validating data...")
        progress_bar.progress(10)
        loader = DataLoader()
        df = loader.load(uploaded_file)
        st.session_state.df = df

        # Step 2: Profile (no cache — stored in session_state)
        status_text.text("🔬 Profiling dataset...")
        progress_bar.progress(22)
        profiler = DataProfiler()
        profile = profiler.profile(df)
        st.session_state.profile = profile

        # Step 3: Anomaly detection (no cache — stored in session_state)
        status_text.text("🚨 Detecting anomalies...")
        progress_bar.progress(38)
        detector = AnomalyDetector()
        anomalies = detector.detect(df, profile)
        st.session_state.anomalies = anomalies

        # Step 4: AI analysis (session-scoped cache)
        # Only include sample rows if user explicitly opts in
        status_text.text("🤖 Running AI analysis (this may take a moment)...")
        progress_bar.progress(52)
        include_samples = st.session_state.get("sample_data_consent", False)
        df_hash = compute_df_hash(df)
        cache_key = f"{df_hash}:{st.session_state.selected_model}:{include_samples}"
        cache = st.session_state.get("ai_analysis_cache", {})
        if cache_key in cache:
            ai_analysis = cache[cache_key]
            logger.info("AI analysis cache hit")
        else:
            ai_engine = AIEngine(model=st.session_state.selected_model)
            ai_analysis = ai_engine.analyze(profile, include_sample_data=include_samples)
            cache[cache_key] = ai_analysis
            st.session_state.ai_analysis_cache = cache
        st.session_state.ai_analysis = ai_analysis

        # Step 4b: Verify & repair AI output (pure Python, instant)
        status_text.text("🔍 Verifying AI suggestions...")
        progress_bar.progress(62)
        verifier = AnalysisVerifier()
        ai_analysis = verifier.verify(ai_analysis, profile)
        st.session_state.ai_analysis = ai_analysis

        # Step 5: Generate KPIs
        status_text.text("📊 Generating KPIs...")
        progress_bar.progress(72)
        kpi_gen = KPIGenerator()
        kpis = kpi_gen.generate(df, profile, ai_analysis)
        st.session_state.kpis = kpis

        # Step 6: Generate charts
        status_text.text("📈 Building charts...")
        progress_bar.progress(84)
        chart_gen = ChartGenerator()
        charts = chart_gen.generate(df, profile, ai_analysis)
        st.session_state.charts = charts

        # Step 7: Generate insights
        status_text.text("💡 Extracting insights...")
        progress_bar.progress(94)
        insights_gen = InsightsGenerator()
        insights = insights_gen.generate(df, profile, ai_analysis, anomalies)
        st.session_state.insights = insights

        # Done
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        st.session_state.file_processed = True
        st.session_state.processing_error = None

        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        logger.info(f"Successfully processed file: {uploaded_file.name}, shape={df.shape}")

    except Exception as e:
        logger.error(f"Error processing file: {traceback.format_exc()}")
        st.session_state.processing_error = str(e)
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error processing file: {str(e)}")
        return False

    return True


def render_upload_section():
    """Render the file upload section."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📤 Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Drop your CSV file here or click to browse",
            type=["csv"],
            help=f"Max size: {AppConfig.MAX_FILE_SIZE_MB}MB | Max rows: {AppConfig.MAX_ROWS:,}",
            label_visibility="collapsed"
        )

        st.markdown("""
        <div style="display:flex;gap:1rem;justify-content:center;color:#555570;font-size:0.8rem;margin-top:0.5rem;">
            <span>📋 CSV files only</span>
            <span>•</span>
            <span>Max 20MB</span>
            <span>•</span>
            <span>Up to 100K rows</span>
        </div>
        """, unsafe_allow_html=True)

    return uploaded_file


def generate_custom_chart(question: str, df: pd.DataFrame, profile: dict):
    # Use the selected LLM to propose a single chart and build it.
    ai = AIEngine(model=st.session_state.get("selected_model", AppConfig.DEFAULT_MODEL))
    cols = [c["name"] for c in profile.get("columns", [])]
    prompt = f"""Return ONLY valid JSON for ONE chart.
No markdown. No explanation.

Allowed columns: {cols}

Schema:
{{
  "type": "bar|line|pie|scatter|histogram",
  "x": "exact_col_name",
  "y": "exact_col_name_or_null",
  "color": "exact_col_name_or_null",
  "title": "clear descriptive title",
  "description": "what the chart reveals",
  "aggregation": "sum|mean|count|null"
}}

Rules:
1. Avoid ID-like columns entirely.
2. For bar charts with numerical y, set aggregation explicitly.
3. Prefer color grouping for categorical segmentation if it improves readability.

User request: {question}
"""
    raw = ai.call(prompt=prompt, max_tokens=350, temperature=0.1)
    chart_def = extract_json_block(raw)
    if not isinstance(chart_def, dict):
        return None
    verifier = AnalysisVerifier()
    chart_def = verifier.validate_single_chart(chart_def, profile)
    if not chart_def:
        return None
    x = chart_def.get("x")
    y = chart_def.get("y")
    color = chart_def.get("color")
    chart_type = chart_def.get("type", "bar")
    valid_cols = set(df.columns)
    if x not in valid_cols:
        return None
    if y and y not in valid_cols:
        return None
    if color and color not in valid_cols:
        color = None
    gen = ChartGenerator()
    fig = gen._build_figure(df, chart_type, x, y, chart_def, color_col=color)
    if fig is None:
        return None
    return {
        "title": chart_def.get("title", f"{x} chart"),
        "description": chart_def.get("description", ""),
        "figure": fig,
        "type": chart_type,
        "x": x,
        "y": y,
        "color": color,
        "source": "custom",
    }


def render_audit_panel(profile: dict, ai_analysis: dict, df: pd.DataFrame):
    # Explainability panel: columns used and active filters.
    with st.expander("Audit / Explainability", expanded=False):
        cols = [c["name"] for c in profile.get("columns", [])]
        st.markdown("**Model**: " + st.session_state.get("selected_model", AppConfig.DEFAULT_MODEL))
        st.markdown("**Sample Rows Sent**: " + ("Yes" if st.session_state.get("sample_data_consent") else "No"))
        filters = st.session_state.get("active_filters", {})
        if filters:
            st.markdown("**Active Filters**")
            for k, v in filters.items():
                st.markdown(f"- {k}: {v}")
        else:
            st.markdown("**Active Filters**: None")

        st.markdown("**KPI Columns**")
        for kpi in ai_analysis.get("kpis", []):
            st.markdown(f"- {kpi.get('label','KPI')}: {kpi.get('column')} ({kpi.get('aggregation')})")

        st.markdown("**Chart Columns**")
        for ch in ai_analysis.get("charts", []):
            st.markdown(f"- {ch.get('title','Chart')}: x={ch.get('x')} y={ch.get('y')} color={ch.get('color')}")

        custom = st.session_state.get("custom_charts", [])
        if custom:
            st.markdown("**Custom Chart Columns**")
            for ch in custom:
                st.markdown(f"- {ch.get('title','Custom')}: x={ch.get('x')} y={ch.get('y')} color={ch.get('color')}")

        st.markdown("**Insights: Columns Mentioned**")
        for ins in ai_analysis.get("insights", []):
            used = extract_columns_from_text(ins, cols)
            st.markdown(f"- {ins}")
            st.caption("Columns: " + (", ".join(used) if used else "None detected"))


def render_dashboard():
    """Render the full dashboard with all components."""
    df = st.session_state.df
    filtered_df = apply_filters(df)

    # Show filter status
    if st.session_state.get("active_filters"):
        st.info(f"🔍 Filters active — showing {len(filtered_df):,} of {len(df):,} rows")

    # KPI Section
    st.markdown("---")
    st.markdown("### 📊 Key Performance Indicators")
    kpi_renderer = KPIRenderer()
    kpi_renderer.render(st.session_state.kpis, filtered_df)

    # Charts Section
    st.markdown("---")
    st.markdown("### Visual Analytics")

    # Custom chart request
    st.markdown("**Custom Chart Request**")
    custom_prompt = st.text_input(
        "Describe a chart you want to see",
        value="",
        key="custom_chart_prompt",
        placeholder="e.g., Show average sales by region as a bar chart"
    )
    if st.button("Generate Chart", key="custom_chart_btn"):
        if custom_prompt.strip():
            with st.spinner("Building your chart..."):
                chart = generate_custom_chart(custom_prompt, filtered_df, st.session_state.profile)
            if chart:
                st.session_state.custom_charts.append(chart)
                st.success("Custom chart added.")
                st.rerun()
            else:
                st.warning("Could not build that chart. Try a simpler request with clear column names.")
        else:
            st.warning("Please describe the chart you want.")

    charts = list(st.session_state.charts or [])
    charts.extend(st.session_state.get("custom_charts", []))
    chart_renderer = ChartRenderer()
    chart_renderer.render(charts, filtered_df, st.session_state.profile)

    # Insights Section
    st.markdown("---")
    st.markdown("### 💡 AI-Generated Insights")
    DashboardRenderer.render_insights(st.session_state.insights, st.session_state.anomalies)

    # Explainability
    render_audit_panel(st.session_state.profile, {
        "kpis": st.session_state.kpis or [],
        "charts": st.session_state.ai_analysis.get("charts", []) if st.session_state.ai_analysis else [],
        "insights": st.session_state.insights or [],
    }, filtered_df)

    # Data Explorer
    st.markdown("---")
    st.markdown("### 🔍 Data Explorer")
    DashboardRenderer.render_data_explorer(filtered_df, st.session_state.profile)

    # Chat Section
    st.markdown("---")
    st.markdown("### 💬 Chat with Your Data")
    render_chat_section(filtered_df)

    # Export Section
    st.markdown("---")
    st.markdown("### 📥 Export")
    render_export_section(filtered_df)


def render_chat_section(df: pd.DataFrame):
    """Render the chat interface for data Q&A."""
    # Use the user's selected model for chat
    chat_engine = ChatEngine(model=st.session_state.get("selected_model", AppConfig.DEFAULT_MODEL))

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "code" in msg:
                with st.expander("🔍 Generated Code"):
                    st.code(msg["code"], language="python")
            if "result" in msg:
                if isinstance(msg["result"], pd.DataFrame):
                    st.dataframe(msg["result"], use_container_width=True)

    # Chat input
    if prompt := st.chat_input("Ask anything about your data... (e.g., 'What is the average sales by region?')"):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing your question..."):
                response = chat_engine.ask(
                    question=prompt,
                    df=df,
                    profile=st.session_state.profile
                )

            # Stream the answer for a faster, more interactive feel
            try:
                import time
                def _chunk_text(s, size=80):
                    for i in range(0, len(s), size):
                        yield s[i:i+size]
                        time.sleep(0.01)
                st.write_stream(_chunk_text(response["answer"]))
            except Exception:
                st.markdown(response["answer"])

            if response.get("code"):
                with st.expander("🔍 Generated Code"):
                    st.code(response["code"], language="python")

            if response.get("result") is not None:
                result = response["result"]
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, use_container_width=True)
                elif isinstance(result, (int, float)):
                    st.metric("Result", f"{result:,.4f}" if isinstance(result, float) else f"{result:,}")

        # Save to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "code": response.get("code", ""),
            "result": response.get("result")
        })

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


def render_export_section(df: pd.DataFrame):
    """Render export options."""
    col1, col2, col3 = st.columns(3)

    with col1:
        # Export insights as text
        insights_text = "\n\n".join([
            f"• {insight}" for insight in (st.session_state.insights or [])
        ])
        profile = st.session_state.profile or {}
        export_text = f"""AI DATA ANALYST REPORT
{'='*50}
Dataset: {profile.get('row_count', 0):,} rows × {profile.get('column_count', 0)} columns
Generated by: AI Data Analyst Dashboard

INSIGHTS:
{insights_text}

ANOMALIES DETECTED:
{len(st.session_state.anomalies or {})} columns with anomalies

KPIs:
{chr(10).join([f"• {k['label']}: {k['value']}" for k in (st.session_state.kpis or [])])}
"""
        st.download_button(
            "📄 Download Insights Report",
            data=export_text,
            file_name="ai_insights_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col2:
        # Export filtered data as CSV
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📊 Download Filtered Data (CSV)",
            data=csv_data,
            file_name="filtered_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col3:
        # Export summary stats
        summary = df.describe(include="all").to_csv()
        st.download_button(
            "📈 Download Summary Statistics",
            data=summary,
            file_name="summary_statistics.csv",
            mime="text/csv",
            use_container_width=True
        )


def main():
    """Main application entry point."""
    initialize_session_state()
    apply_custom_css()
    render_header()

    # Handle sample dataset loading
    if "load_sample" in st.session_state and st.session_state.load_sample:
        sample_path = st.session_state.load_sample
        st.session_state.load_sample = None

        with open(sample_path, "rb") as f:
            import io
            content = f.read()
            filename = Path(sample_path).name

            class FakeUploadedFile:
                def __init__(self, content, name):
                    self.name = name
                    self._content = content
                    self.size = len(content)

                def read(self):
                    return self._content

                def getvalue(self):
                    return self._content

            fake_file = FakeUploadedFile(content, filename)

        with st.spinner(f"Loading sample dataset: {filename}"):
            if process_uploaded_file(fake_file):
                st.success(f"✅ Sample dataset loaded: {filename}")
                st.rerun()

    # Sidebar
    render_sidebar(st.session_state.df)

    # Main content
    if not st.session_state.file_processed:
        uploaded_file = render_upload_section()

        if uploaded_file is not None:
            with st.spinner("Processing your dataset..."):
                if process_uploaded_file(uploaded_file):
                    st.rerun()

        # Show welcome info
        if not st.session_state.file_processed:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            FEATURE_CARD = """
            <div style="
                background:linear-gradient(135deg,#16161f 0%,#1a1a27 100%);
                border:1px solid #2a2a3a;border-radius:12px;
                padding:1.5rem;text-align:center;height:100%;
            ">
                <div style="font-size:2rem;margin-bottom:0.75rem;">{icon}</div>
                <div style="font-size:1rem;font-weight:600;color:#f0f0ff;margin:0.5rem 0;">{title}</div>
                <div style="font-size:0.85rem;color:#8888aa;line-height:1.5;">{body}</div>
            </div>
            """

            with col1:
                st.markdown(FEATURE_CARD.format(
                    icon="🤖", title="AI-Powered Analysis",
                    body="Automatically detects patterns, generates insights, and suggests the best visualizations for your data."
                ), unsafe_allow_html=True)
            with col2:
                st.markdown(FEATURE_CARD.format(
                    icon="💬", title="Chat with Data",
                    body="Ask questions in plain English. Our AI generates and executes pandas code to answer your questions instantly."
                ), unsafe_allow_html=True)
            with col3:
                st.markdown(FEATURE_CARD.format(
                    icon="📊", title="Smart Dashboards",
                    body="Auto-generated KPI cards, charts, and interactive filters tailored to your specific dataset."
                ), unsafe_allow_html=True)

    else:
        # Reset button
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🔄 New Analysis", type="secondary"):
                for key in list(st.session_state.keys()):
                    if key != "selected_model":
                        del st.session_state[key]
                st.rerun()

        # Render the full dashboard
        render_dashboard()


if __name__ == "__main__":
    main()

