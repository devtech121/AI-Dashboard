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
            help="Select AI model for analysis. Falls back automatically if unavailable."
        )
        st.session_state.selected_model = selected

        # Groq API Key input
        groq_key = st.text_input(
            "Groq API Key",
            value="Login Required" or "",
            type="password",
            help="Get your free key at console.groq.com"
        )
        if groq_key:
            AppConfig.GROQ_API_KEY = groq_key

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

    # Numeric range filters (first 2)
    for col in numeric_cols[:2]:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        if col_min != col_max:
            range_val = st.slider(
                f"{col} range",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
                key=f"range_{col}"
            )
            if range_val != (col_min, col_max):
                filters_applied[col] = range_val

    st.session_state.active_filters = filters_applied


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
    """Process uploaded CSV file and run full AI pipeline."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Load data
        status_text.text("📂 Loading and validating data...")
        progress_bar.progress(10)
        loader = DataLoader()
        df = loader.load(uploaded_file)
        st.session_state.df = df

        # Step 2: Profile data
        status_text.text("🔬 Profiling dataset...")
        progress_bar.progress(25)
        profiler = DataProfiler()
        profile = profiler.profile(df)
        st.session_state.profile = profile

        # Step 3: Detect anomalies
        status_text.text("🚨 Detecting anomalies...")
        progress_bar.progress(40)
        detector = AnomalyDetector()
        anomalies = detector.detect(df, profile)
        st.session_state.anomalies = anomalies

        # Step 4: AI analysis
        status_text.text("🤖 Running AI analysis (this may take a moment)...")
        progress_bar.progress(55)
        ai_engine = AIEngine(model=st.session_state.selected_model)
        ai_analysis = ai_engine.analyze(profile)
        st.session_state.ai_analysis = ai_analysis

        # Step 5: Generate KPIs
        status_text.text("📊 Generating KPIs...")
        progress_bar.progress(70)
        kpi_gen = KPIGenerator()
        kpis = kpi_gen.generate(df, profile, ai_analysis)
        st.session_state.kpis = kpis

        # Step 6: Generate charts
        status_text.text("📈 Building charts...")
        progress_bar.progress(82)
        chart_gen = ChartGenerator()
        charts = chart_gen.generate(df, profile, ai_analysis)
        st.session_state.charts = charts

        # Step 7: Generate insights
        status_text.text("💡 Extracting insights...")
        progress_bar.progress(93)
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
    st.markdown("### 📈 Visual Analytics")
    chart_renderer = ChartRenderer()
    chart_renderer.render(st.session_state.charts, filtered_df, st.session_state.profile)

    # Insights Section
    st.markdown("---")
    st.markdown("### 💡 AI-Generated Insights")
    DashboardRenderer.render_insights(st.session_state.insights, st.session_state.anomalies)

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
    chat_engine = ChatEngine()

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
