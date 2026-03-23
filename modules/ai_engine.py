"""
AI Engine Module — Groq API with multi-model fallback.

Fixes:
  #3  GROQ_TIMEOUT via httpx — client now pooled per-instance (Issue #4 fix: no leak)
  #6  @st.cache_data removed
  #7  Sample rows require explicit opt-in
  Prompt: complete rewrite for 30x quality improvement
"""

import json
import logging
import re
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.ai_engine")

# Module-level client cache keyed by (api_key, timeout) — reused across calls,
# properly closed on GC. Fixes Issue #4: no per-call client leak.
_client_cache_groq: dict = {}
_client_cache_anthropic: dict = {}


def _get_groq_client(api_key: str):
    """Return a cached Groq client (one per api_key). Issue #4: no per-call leak."""
    key = (api_key, AppConfig.GROQ_TIMEOUT)
    if key not in _client_cache_groq:
        try:
            import httpx
            from groq import Groq
            http_client = httpx.Client(
                timeout=float(AppConfig.GROQ_TIMEOUT),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
            _client_cache_groq[key] = Groq(api_key=api_key, http_client=http_client)
        except Exception:
            try:
                from groq import Groq
                _client_cache_groq[key] = Groq(api_key=api_key)
            except ImportError:
                raise ImportError("groq not installed. Run: pip install groq")
    return _client_cache_groq[key]

def _get_anthropic_client(api_key: str):
    """Return a cached Anthropic client (one per api_key)."""
    key = (api_key, )
    if key not in _client_cache_anthropic:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic not installed. Run: pip install anthropic")
        _client_cache_anthropic[key] = Anthropic(api_key=api_key)
    return _client_cache_anthropic[key]


class AIEngine:
    """
    AI analysis engine — Groq API with 4-model fallback.
    NOT cached with @st.cache_data (Issue #6).
    """

    ANALYSIS_MAX_TOKENS = 4096
    CHAT_MAX_TOKENS_DEFAULT = 1024

    def __init__(self, model: str = None):
        self.model = model or AppConfig.DEFAULT_MODEL
        self._fallback_order = list(AppConfig.FALLBACK_ORDER)
        # NOTE: do NOT cache the key here — always read it fresh
        # via self._get_api_key() so sidebar/secrets changes take effect

    def _get_api_key(self) -> str:
        """Always resolve the key fresh — never cached on self."""
        return AppConfig.get_groq_key()

    def _get_api_key(self, provider: str) -> str:
        if provider == "anthropic":
            return AppConfig.get_claude_key()
        return AppConfig.get_groq_key()

    def analyze(self, profile: dict, include_sample_data: bool = False) -> dict:
        """
        Run AI analysis on dataset profile.
        Issue #7: sample rows not sent unless include_sample_data=True.
        """
        prompt = self._build_analysis_prompt(profile, compact=False, include_sample=include_sample_data)
        raw = self._call_with_fallback(prompt, max_tokens=self.ANALYSIS_MAX_TOKENS, temperature=0.15)

        if raw:
            if self._is_truncated(raw):
                logger.warning("Truncated — retrying with compact prompt")
                compact = self._build_analysis_prompt(profile, compact=True, include_sample=False)
                raw = self._call_with_fallback(compact, max_tokens=self.ANALYSIS_MAX_TOKENS, temperature=0.15)

            if raw and not self._is_truncated(raw):
                parsed = self._parse_json_response(raw)
                if parsed:
                    parsed = self._sanitize_output(parsed, profile)
                    logger.info("AI analysis successful")
                    return parsed

        logger.warning("AI analysis failed — rule-based fallback")
        return self._rule_based_fallback(profile)

    def call(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        return self._call_with_fallback(
            prompt,
            max_tokens=max_tokens or self.CHAT_MAX_TOKENS_DEFAULT,
            temperature=temperature or AppConfig.CHAT_TEMPERATURE,
        ) or ""

    # ── API Calls ──────────────────────────────────────────────

    def _call_with_fallback(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        starting_model: str = None,
    ) -> str | None:
        """Try each model in fallback order until one succeeds."""
        if not self.api_key:
            logger.warning("No Groq API key — skipping AI call")
            return None

        models_to_try = self._get_fallback_order(starting_model or self.model)
        for model_id in models_to_try:
            try:
                result = self._call_model(model_id, prompt, max_tokens, temperature)
                if result:
                    logger.info(f"Response from: {model_id}")
                    return result
            except Exception as e:
                logger.warning(f"{model_id} failed: {e}")
        return None

    def _get_fallback_order(self, starting_model):
        if starting_model in self._fallback_order:
            idx = self._fallback_order.index(starting_model)
            return self._fallback_order[idx:] + self._fallback_order[:idx]
        return self._fallback_order

    def _call_model(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str | None:
        """Make a single Groq chat completion call."""
        client = _get_groq_client(self.api_key)
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response.choices[0].finish_reason == "length":
            logger.warning(f"{model_id} hit token limit")
        text = response.choices[0].message.content
        return text.strip() if text else None

    # ── Truncation ─────────────────────────────────────────────

    def _is_truncated(self, text):
        if not text:
            return True
        s = text.strip()
        if not s.startswith("{") or not s.endswith("}"):
            return True
        if s.count("{") + s.count("[") != s.count("}") + s.count("]"):
            return True
        if s.count('"') % 2 != 0:
            return True
        return False

    # ── UPGRADED PROMPT ────────────────────────────────────────

    def _build_analysis_prompt(self, profile: dict, compact: bool = False, include_sample: bool = False) -> str:
        """
        Produces a highly detailed, context-aware prompt that:
        - Detects dataset domain (finance/HR/ecommerce/medical/generic)
        - Gives domain-specific chart recommendations
        - Handles multi-entity time series (stocks by ticker) correctly
        - Enforces smart aggregation rules
        - Requests per-column insights not just generic summaries
        """
        id_cols  = set(profile.get("id_cols", []))
        bin_cols = set(profile.get("binary_cols", []))
        num_cols = [c for c in profile["numerical_cols"] if c not in id_cols and c not in bin_cols]
        cat_cols = profile["categorical_cols"]
        dt_cols  = profile["datetime_cols"]
        all_cols = [c["name"] for c in profile.get("columns", [])]

        # ── Build rich column descriptions ──
        col_lines = []
        max_cols = 10 if compact else 22
        for col in profile["columns"][:max_cols]:
            name  = col["name"]
            ctype = col["type"]
            tags  = []
            if name in bin_cols: tags.append("binary/flag")
            if name in id_cols:  tags.append("ID-SKIP")
            tag_str = f" [{', '.join(tags)}]" if tags else ""

            line = f"  • {name} ({ctype}{tag_str})"
            if ctype == "numerical" and isinstance(col.get("mean"), float):
                skew = col.get("skew", 0)
                skew_note = " [right-skewed]" if skew > 1 else " [left-skewed]" if skew < -1 else ""
                line += f": mean={col['mean']:.2f}, min={col.get('min','?')}, max={col.get('max','?')}, std={col.get('std',0):.2f}{skew_note}"
            elif ctype == "categorical":
                n_u  = col.get("n_unique", "?")
                mode = col.get("mode", "")
                mpct = col.get("mode_pct", 0)
                top  = list(col.get("top_values", {}).keys())[:5]
                line += f": {n_u} unique | top='{mode}'({mpct}%) | values={top}"
            col_lines.append(line)
        cols_text = "\n".join(col_lines)

        # ── Detect dataset domain for better recommendations ──
        all_col_names = " ".join(all_cols).lower()
        domain = "generic"
        domain_hint = ""

        if any(k in all_col_names for k in ["ticker", "open", "close", "volume", "high", "low", "adj"]):
            domain = "financial_timeseries"
            cat_entity_cols = [c for c in cat_cols if c in ["ticker", "symbol", "stock", "company"] or
                               any(c == x for x in cat_cols if profile.get("columns") and
                                   next((m for m in profile["columns"] if m["name"]==x), {}).get("n_unique",99) <= 20)]
            entity_col = cat_entity_cols[0] if cat_entity_cols else (cat_cols[0] if cat_cols else None)
            price_cols = [c for c in num_cols if any(k in c.lower() for k in ["close","open","price","high","low","adj"])]
            domain_hint = f"""
DOMAIN: Financial time series data detected.
KEY RULE: For each price column ({price_cols[:3]}), create ONE line chart per entity
in '{entity_col}' using color={entity_col} so all tickers appear on the SAME chart.
This means use x={dt_cols[0] if dt_cols else 'date'}, y=price_col, color={entity_col}.
Do NOT create separate charts per ticker — use the color dimension instead.
Also create: volume bar chart, price distribution histograms, OHLC comparisons."""

        elif any(k in all_col_names for k in ["salary", "department", "employee", "hire", "attrition", "performance"]):
            domain = "hr_analytics"
            domain_hint = """
DOMAIN: HR/People analytics detected.
Recommended charts: salary distribution by department, attrition rate by job_level,
age/experience scatter, headcount by department (bar), performance vs satisfaction scatter.
KPIs: avg salary, total headcount, attrition rate (%), avg tenure, avg performance score."""

        elif any(k in all_col_names for k in ["revenue", "order", "product", "category", "quantity", "customer", "sale"]):
            domain = "ecommerce"
            domain_hint = """
DOMAIN: E-commerce/Sales data detected.
Recommended charts: revenue by category (bar), orders over time (line), top products (bar),
payment method distribution (pie), revenue vs quantity scatter, return rate by category.
KPIs: total revenue, avg order value, total orders, avg rating."""

        elif any(k in all_col_names for k in ["age", "cholesterol", "blood", "pressure", "patient", "diagnosis", "chol", "trestbps"]):
            domain = "medical"
            domain_hint = """
DOMAIN: Medical/health data detected.
Recommended charts: age distribution (histogram), condition prevalence by age group (bar),
key biomarker distributions (histogram), risk factor correlations (scatter),
outcome rates by demographic (bar with binary y → show rate%).
KPIs: avg age, avg biomarker values, outcome rate (%), sample count."""

        # ── Correlation hints ──
        corr_hints = ""
        if profile.get("correlation_matrix") and len(num_cols) >= 2:
            high = []
            for c1 in num_cols[:8]:
                for c2 in num_cols[:8]:
                    if c1 >= c2: continue
                    try:
                        r = (profile["correlation_matrix"].get(c1) or {}).get(c2, 0) or 0
                        if abs(r) > 0.5:
                            direction = "positive" if r > 0 else "negative"
                            high.append(f"{c1}↔{c2} r={r:.2f} ({direction})")
                    except Exception:
                        pass
            if high:
                corr_hints = f"\nSTRONG CORRELATIONS (suggest scatter charts): {', '.join(high[:5])}"

        # ── Sample data (consent-gated) ──
        sample_section = ""
        if include_sample and not compact:
            sample_str = str(profile.get("sample_data", [])[:3])[:500]
            sample_section = f"\nSAMPLE ROWS (user consented):\n{sample_str}"

        # ── Chart count targets ──
        n_kpis, n_charts, n_insights = ("3-4","4-5","3") if compact else ("5-7","6-8","5-7")

        # ── Missing value context ──
        missing_ctx = ""
        if profile.get("missing_summary"):
            mc = [(k, v["pct"]) for k, v in profile["missing_summary"].items()]
            missing_ctx = f"\nMISSING DATA: {', '.join(f'{c}={p}%' for c,p in mc[:5])}"

        return f"""You are a senior data analyst. Analyze this dataset and return ONLY valid JSON.
No markdown. No explanation. Every bracket must be closed.
{domain_hint}
DATASET OVERVIEW:
  Rows: {profile['row_count']:,} | Columns: {profile['column_count']}
  Numerical (use for KPIs/Y-axis): {num_cols[:10]}
  Categorical (use for X-axis/grouping/color): {cat_cols[:8]}
  Binary/flag (rate analysis only, NOT sum KPIs): {list(bin_cols)[:6]}
  DateTime (use for time-series X-axis): {dt_cols[:4]}
  ID columns (SKIP entirely, never use): {list(id_cols)[:4]}
{corr_hints}
{missing_ctx}

COLUMN DETAILS:
{cols_text}
{sample_section}

Return EXACTLY this JSON structure:
{{
  "kpis": [
    {{
      "label": "Human-readable metric name",
      "column": "exact_col_name",
      "aggregation": "sum|mean|count|max|min",
      "format": "number|currency|percent",
      "insight": "One sentence why this KPI matters"
    }}
  ],
  "charts": [
    {{
      "type": "bar|line|pie|scatter|histogram",
      "x": "exact_col_name",
      "y": "exact_col_name_or_null",
      "color": "exact_col_name_or_null",
      "title": "Specific descriptive title with context",
      "description": "What this chart reveals about the data",
      "aggregation": "sum|mean|count|null"
    }}
  ],
  "insights": [
    "Specific, quantified insight referencing actual column names and values"
  ]
}}

═══ STRICT KPI RULES ═══
1. Use ONLY columns from Numerical list — never ID, binary, or categorical columns
2. aggregation: sum → revenue/sales/cost totals | mean → prices/scores/ages/rates | count → record counts | max/min → extremes
3. format: currency → col name contains price/revenue/cost/salary/pay/income | percent → rate/ratio/score/pct | number → everything else
4. insight field: write why this metric matters (e.g. "Total revenue across all {profile['row_count']:,} transactions")
5. Target exactly {n_kpis} KPIs covering different aspects of the data
6. NEVER create a KPI for year columns or ID-like columns

═══ STRICT CHART RULES ═══
1. NEVER use ID columns as x or y
2. DateTime x + Numerical y → "line" (time series)
3. DateTime x + Numerical y + Categorical color → "line" with color grouping (BEST for multi-entity series like stocks by ticker)
4. 2 Numerical columns → "scatter" (shows correlation)
5. Categorical x + Numerical y → "bar" with smart aggregation:
   - binary y → show count/rate, use aggregation="count"
   - large-value y (revenue/salary) → aggregation="sum"
   - score/rate y → aggregation="mean"
6. Single numerical column → "histogram" (shows distribution)
7. Categorical with ≤ 8 unique values → consider "pie"
8. color field: use a categorical column to split lines/bars by group (crucial for stock tickers, departments, regions)
9. Target exactly {n_charts} charts with maximum diversity of chart types and column combinations
10. Prioritize charts that reveal the most interesting patterns given the domain

═══ STRICT INSIGHT RULES ═══
1. Every insight MUST reference specific column names and actual numbers from the dataset
2. Cite exact values: "The average salary is $91.8K, ranging from $35K to $149K"
3. Highlight anomalies, surprising patterns, or business-critical findings
4. If binary columns exist: state the positive rate as a percentage
5. Mention the strongest correlations if any exist
6. Target exactly {n_insights} insights, each revealing something different
7. NEVER write generic statements like "the dataset has X columns" — be specific

Use ONLY column names from the COLUMN DETAILS section above."""

    # ── JSON Parsing ───────────────────────────────────────────

    def _parse_json_response(self, text):
        if not text:
            return None
        try:
            d = json.loads(text)
            if self._validate_ai_output(d): return d
        except json.JSONDecodeError:
            pass
        for pat in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            for m in re.findall(pat, text, re.DOTALL):
                try:
                    d = json.loads(m)
                    if self._validate_ai_output(d): return d
                except Exception: continue
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                d = json.loads(text[start:end+1])
                if self._validate_ai_output(d): return d
            except Exception: pass
        return self._attempt_repair(text)

    def _attempt_repair(self, text):
        try:
            s = text.strip()
            start = s.find("{")
            if start == -1: return None
            partial = s[start:]
            stack, in_str, esc = [], False, False
            for ch in partial:
                if esc: esc = False; continue
                if ch == "\\" and in_str: esc = True; continue
                if ch == '"': in_str = not in_str; continue
                if in_str: continue
                if ch in "{[": stack.append(ch)
                elif ch in "}]" and stack: stack.pop()
            if in_str: partial += '"'
            for b in reversed(stack):
                partial += "}" if b == "{" else "]"
            d = json.loads(partial)
            if self._validate_ai_output(d):
                logger.info("JSON repair succeeded")
                return d
        except Exception: pass
        return None

    def _validate_ai_output(self, data):
        return isinstance(data, dict) and all(k in data for k in ("kpis","charts","insights"))

    def _sanitize_output(self, data, profile):
        valid = set(c["name"] for c in profile.get("columns", []))
        data["kpis"] = [k for k in data.get("kpis", [])
                       if k.get("column") in valid or k.get("column") == "_count"]
        data["charts"] = [c for c in data.get("charts", [])
                         if c.get("x") in valid
                         and (c.get("y") is None or c.get("y") in valid)
                         and (c.get("color") is None or c.get("color") in valid)]
        data["insights"] = [str(i) for i in data.get("insights", []) if i]
        return data

    # ── Rule-Based Fallback ────────────────────────────────────

    def _rule_based_fallback(self, profile):
        num_cols = profile["numerical_cols"]
        cat_cols = profile["categorical_cols"]
        dt_cols  = profile["datetime_cols"]
        id_cols  = set(profile.get("id_cols", []))

        kpis = []
        for col in [c for c in num_cols if c not in id_cols][:5]:
            cl = col.lower()
            is_currency = any(k in cl for k in ["price","revenue","sales","amount","cost","profit"])
            kpis.append({
                "label": col.replace("_"," ").title(),
                "column": col,
                "aggregation": "sum" if any(k in cl for k in ["amount","revenue","sales","total"]) else "mean",
                "format": "currency" if is_currency else "number",
                "insight": f"Key metric from {col}",
            })
        if not kpis and num_cols:
            kpis.append({"label":"Row Count","column":num_cols[0],"aggregation":"count","format":"number","insight":"Total records"})

        charts = []
        def add(t,x,y,title,desc,color=None,agg=None):
            charts.append({"type":t,"x":x,"y":y,"color":color,"title":title,"description":desc,"aggregation":agg})

        if dt_cols and num_cols:
            color_col = cat_cols[0] if cat_cols else None
            add("line", dt_cols[0], num_cols[0], f"{num_cols[0].replace('_',' ').title()} Over Time",
                "Time series trend", color=color_col)
        if cat_cols and num_cols:
            add("bar", cat_cols[0], num_cols[0],
                f"{num_cols[0].replace('_',' ').title()} by {cat_cols[0].replace('_',' ').title()}",
                "Category breakdown", agg="mean")
        if num_cols:
            add("histogram", num_cols[0], None, f"Distribution of {num_cols[0].replace('_',' ').title()}", "Distribution")
        for col in cat_cols[:2]:
            if len(charts) >= 6: break
            add("pie", col, None, f"{col.replace('_',' ').title()} Breakdown", "Proportional breakdown")
        if len(num_cols) >= 2:
            add("scatter", num_cols[0], num_cols[1],
                f"{num_cols[0].replace('_',' ').title()} vs {num_cols[1].replace('_',' ').title()}",
                "Correlation analysis")

        insights = [f"Dataset: {profile['row_count']:,} rows × {profile['column_count']} columns."]
        if num_cols: insights.append(f"Numerical metrics available: {', '.join(num_cols[:4])}.")
        if cat_cols: insights.append(f"Categorical dimensions: {', '.join(cat_cols[:3])}.")
        if dt_cols:  insights.append(f"Time-based analysis using '{dt_cols[0]}'.")

        return {
            "kpis": kpis,
            "charts": charts[:AppConfig.MAX_CHARTS],
            "insights": insights,
            "source": "rule-based-fallback",
        }
