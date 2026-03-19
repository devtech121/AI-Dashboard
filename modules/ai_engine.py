"""
AI Engine Module
Groq API integration with multi-model fallback system.
Uses the official groq Python SDK with OpenAI-compatible chat completions.
"""

import json
import logging
import re
import streamlit as st
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.ai_engine")


def _get_groq_client(api_key: str):
    """Initialise and return a Groq client."""
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        raise ImportError("groq package not installed. Run: pip install groq")


class AIEngine:
    """
    AI analysis engine powered by Groq API.
    Automatically falls back through the model chain on failure.
    """

    # Groq rate limits vary — use 4096 for analysis, 1024 for chat
    ANALYSIS_MAX_TOKENS = 4096
    CHAT_MAX_TOKENS_DEFAULT = 1024

    def __init__(self, model: str = None):
        self.model = model or AppConfig.DEFAULT_MODEL
        self.api_key = AppConfig.GROQ_API_KEY
        self._fallback_order = list(AppConfig.FALLBACK_ORDER)

    @st.cache_data(show_spinner=False)
    def analyze(_self, profile: dict) -> dict:
        """
        Run AI analysis on the dataset profile.
        Returns structured dict with: kpis, charts, insights.

        Strategy:
          1. Try full prompt with high token budget
          2. If response is truncated, retry with compact prompt
          3. If parse still fails, use rule-based fallback
        """
        # ── Attempt 1: full prompt ──
        prompt = _self._build_analysis_prompt(profile, compact=False)
        raw = _self._call_with_fallback(
            prompt,
            max_tokens=_self.ANALYSIS_MAX_TOKENS,
            temperature=0.2,
        )

        if raw:
            if _self._is_truncated(raw):
                logger.warning("Response truncated — retrying with compact prompt")
                # ── Attempt 2: compact prompt ──
                compact_prompt = _self._build_analysis_prompt(profile, compact=True)
                raw = _self._call_with_fallback(
                    compact_prompt,
                    max_tokens=_self.ANALYSIS_MAX_TOKENS,
                    temperature=0.2,
                )

            if raw and not _self._is_truncated(raw):
                parsed = _self._parse_json_response(raw)
                if parsed:
                    # Validate column names against profile
                    parsed = _self._sanitize_output(parsed, profile)
                    logger.info("AI analysis successful via Groq")
                    return parsed

        logger.warning("AI analysis failed — using rule-based fallback")
        return _self._rule_based_fallback(profile)

    def call(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Public single-call method used by ChatEngine."""
        return self._call_with_fallback(
            prompt,
            max_tokens=max_tokens or self.CHAT_MAX_TOKENS_DEFAULT,
            temperature=temperature or AppConfig.CHAT_TEMPERATURE,
        ) or ""

    # ── Private: API Calls ─────────────────────────────────────

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
                    logger.info(f"Groq response from: {model_id}")
                    return result
            except Exception as e:
                logger.warning(f"Model {model_id} failed: {e}")
                continue
        return None

    def _get_fallback_order(self, starting_model: str) -> list:
        """Return models starting from preferred, then fall back."""
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
                {
                    "role": "system",
                    "content": (
                        "You are an expert data analyst. "
                        "You MUST return ONLY valid, complete JSON — "
                        "no explanation, no markdown fences, no truncation. "
                        "Every opening bracket must have a closing bracket."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Log finish reason for debugging truncation
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            logger.warning(
                f"Model {model_id} hit token limit (finish_reason=length). "
                f"Consider using compact prompt."
            )

        text = response.choices[0].message.content
        return text.strip() if text else None

    # ── Private: Truncation Detection ─────────────────────────

    def _is_truncated(self, text: str) -> bool:
        """
        Detect if a JSON response was cut off mid-stream.
        Checks bracket balance and common truncation signatures.
        """
        if not text:
            return True

        stripped = text.strip()

        # Must start with { for our responses
        if not stripped.startswith("{"):
            return True

        # Count unmatched brackets — truncated JSON will be unbalanced
        opens  = stripped.count("{") + stripped.count("[")
        closes = stripped.count("}") + stripped.count("]")
        if opens != closes:
            logger.debug(f"Bracket mismatch: {opens} open vs {closes} close")
            return True

        # Must end with closing brace
        if not stripped.endswith("}"):
            return True

        # Check for obviously incomplete string (ends inside a quoted value)
        if stripped.count('"') % 2 != 0:
            return True

        return False

    # ── Private: Prompt Builders ───────────────────────────────

    def _build_analysis_prompt(self, profile: dict, compact: bool = False) -> str:
        """
        Build the structured analysis prompt.
        compact=True generates a shorter prompt requesting fewer items,
        reducing the output token count needed.
        """
        col_summary = []
        max_cols = 10 if compact else 20
        for col in profile["columns"][:max_cols]:
            summary = f"- {col['name']} ({col['type']})"
            if col["type"] == "numerical" and isinstance(col.get("mean"), float):
                summary += f": mean={col['mean']:.1f}, max={col.get('max','?')}"
            elif col["type"] == "categorical":
                top = list(col.get("top_values", {}).keys())[:2]
                summary += f": e.g. {top}"
            col_summary.append(summary)

        cols_text = "\n".join(col_summary)

        # Compact mode: skip sample data, request fewer items
        if compact:
            n_kpis   = "3-4"
            n_charts = "4-5"
            n_insights = "3"
            sample_section = ""
        else:
            n_kpis   = "4-5"
            n_charts = "5-6"
            n_insights = "4-5"
            sample_str = json.dumps(
                profile.get("sample_data", [])[:2], default=str
            )[:400]
            sample_section = f"\nSAMPLE DATA:\n{sample_str}\n"

        num_cols = [c for c in profile["numerical_cols"] if c not in profile.get("id_cols", [])]
        cat_cols = profile["categorical_cols"]
        dt_cols  = profile["datetime_cols"]

        return f"""Return ONLY valid complete JSON. No markdown. No explanation. No truncation.

DATASET: {profile['row_count']:,} rows, {profile['column_count']} columns
Numerical: {num_cols[:8]}
Categorical: {cat_cols[:6]}
DateTime: {dt_cols[:3]}

COLUMNS:
{cols_text}
{sample_section}
Return this exact JSON (all brackets must be closed):
{{
  "kpis": [
    {{"label": "Name", "column": "col", "aggregation": "sum|mean|count|max|min", "format": "number|currency|percent"}}
  ],
  "charts": [
    {{"type": "bar|line|pie|scatter|histogram", "x": "col", "y": "col_or_null", "title": "Title", "description": "reason"}}
  ],
  "insights": ["insight 1", "insight 2", "insight 3"]
}}

STRICT RULES:
- kpis: exactly {n_kpis} items, use only numerical columns, skip ID-like columns
- charts: exactly {n_charts} items, use ONLY columns listed above
- insights: exactly {n_insights} short sentences with specific numbers from this dataset
- y can be null for pie/histogram
- Every string must be properly closed with quotes
- The entire response must be one valid JSON object"""

    # ── Private: JSON Parsing ──────────────────────────────────

    def _parse_json_response(self, text: str) -> dict | None:
        """Extract and validate JSON from the model response."""
        if not text:
            return None

        # Direct parse
        try:
            data = json.loads(text)
            if self._validate_ai_output(data):
                return data
        except json.JSONDecodeError:
            pass

        # Strip markdown fences
        for pattern in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            for match in re.findall(pattern, text, re.DOTALL):
                try:
                    data = json.loads(match)
                    if self._validate_ai_output(data):
                        return data
                except Exception:
                    continue

        # Extract outermost { }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
                if self._validate_ai_output(data):
                    return data
            except Exception:
                pass

        # Last resort: try to repair truncated JSON by auto-closing
        repaired = self._attempt_json_repair(text)
        if repaired:
            return repaired

        logger.warning(f"Could not parse AI JSON: {text[:200]}")
        return None

    def _attempt_json_repair(self, text: str) -> dict | None:
        """
        Attempt to close a truncated JSON string by appending
        the right number of closing brackets.
        """
        try:
            stripped = text.strip()
            start = stripped.find("{")
            if start == -1:
                return None
            partial = stripped[start:]

            # Count unclosed brackets
            stack = []
            in_string = False
            escape_next = False
            for char in partial:
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\" and in_string:
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char in "{[":
                    stack.append(char)
                elif char in "}]":
                    if stack:
                        stack.pop()

            # Close any open string
            if in_string:
                partial += '"'

            # Close remaining open brackets
            for bracket in reversed(stack):
                partial += "}" if bracket == "{" else "]"

            data = json.loads(partial)
            if self._validate_ai_output(data):
                logger.info("JSON repair succeeded")
                return data
        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")
        return None

    def _validate_ai_output(self, data: dict) -> bool:
        """Check the AI response has the required top-level keys."""
        return isinstance(data, dict) and all(k in data for k in ("kpis", "charts", "insights"))

    def _sanitize_output(self, data: dict, profile: dict) -> dict:
        """
        Remove any chart/KPI entries that reference columns
        not present in the actual dataset.
        """
        valid_cols = set(col["name"] for col in profile.get("columns", []))

        # Filter KPIs
        data["kpis"] = [
            k for k in data.get("kpis", [])
            if k.get("column") in valid_cols or k.get("column") == "_count"
        ]

        # Filter charts — x must be valid; y can be null
        data["charts"] = [
            c for c in data.get("charts", [])
            if c.get("x") in valid_cols
            and (c.get("y") is None or c.get("y") in valid_cols)
        ]

        # Ensure insights is a flat list of strings
        data["insights"] = [
            str(i) for i in data.get("insights", []) if i
        ]

        return data

    # ── Rule-Based Fallback ────────────────────────────────────

    def _rule_based_fallback(self, profile: dict) -> dict:
        """
        Pure rule-based dashboard config.
        Used when Groq is unavailable or the API key is not set.
        """
        num_cols = profile["numerical_cols"]
        cat_cols = profile["categorical_cols"]
        dt_cols  = profile["datetime_cols"]
        id_cols  = set(profile.get("id_cols", []))

        # ── KPIs ──
        kpis = []
        for col in [c for c in num_cols if c not in id_cols][:5]:
            col_l = col.lower()
            is_currency = any(k in col_l for k in ["price", "revenue", "sales", "amount", "cost", "profit"])
            kpis.append({
                "label": col.replace("_", " ").title(),
                "column": col,
                "aggregation": "sum" if any(k in col_l for k in ["amount", "revenue", "sales", "total"]) else "mean",
                "format": "currency" if is_currency else "number",
            })
        if not kpis and num_cols:
            kpis.append({"label": "Row Count", "column": num_cols[0], "aggregation": "count", "format": "number"})

        # ── Charts ──
        charts = []

        def add(ctype, x, y, title, desc):
            charts.append({"type": ctype, "x": x, "y": y, "title": title, "description": desc})

        if cat_cols and num_cols:
            add("bar", cat_cols[0], num_cols[0],
                f"{num_cols[0].replace('_',' ').title()} by {cat_cols[0].replace('_',' ').title()}",
                "Category breakdown")
        if dt_cols and num_cols:
            add("line", dt_cols[0], num_cols[0],
                f"{num_cols[0].replace('_',' ').title()} Over Time", "Time series trend")
        if num_cols:
            add("histogram", num_cols[0], None,
                f"Distribution of {num_cols[0].replace('_',' ').title()}", "Value distribution")
        for col in cat_cols[:3]:
            if len(charts) >= 6:
                break
            add("pie", col, None, f"{col.replace('_',' ').title()} Breakdown", "Proportional breakdown")
        if len(num_cols) >= 2:
            add("scatter", num_cols[0], num_cols[1],
                f"{num_cols[0].replace('_',' ').title()} vs {num_cols[1].replace('_',' ').title()}",
                "Correlation analysis")
        if len(cat_cols) > 1 and len(num_cols) > 1:
            add("bar", cat_cols[1], num_cols[1],
                f"{num_cols[1].replace('_',' ').title()} by {cat_cols[1].replace('_',' ').title()}",
                "Secondary breakdown")

        # ── Insights ──
        insights = [f"Dataset contains {profile['row_count']:,} records across {profile['column_count']} dimensions."]
        if profile.get("missing_summary"):
            mc = list(profile["missing_summary"].keys())
            insights.append(f"Data quality: {len(mc)} column(s) have missing values — {', '.join(mc[:3])}.")
        if num_cols:
            insights.append(f"{len(num_cols)} numerical metric(s) available: {', '.join(num_cols[:3])}.")
        if cat_cols:
            insights.append(f"Categorical dimensions for segmentation: {', '.join(cat_cols[:3])}.")
        if dt_cols:
            insights.append(f"Time-based analysis available using '{dt_cols[0]}' column.")

        return {
            "kpis": kpis,
            "charts": charts[:AppConfig.MAX_CHARTS],
            "insights": insights,
            "source": "rule-based-fallback",
        }