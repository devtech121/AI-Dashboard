"""
Analysis Verifier Module
Pure-Python post-processing layer that sits between AIEngine output
and the KPI/Chart generators. Fixes deterministic bugs that AI
consistently makes — ID column misuse, wrong aggregations, duplicate
charts, poor chart diversity. Zero API calls, instant execution.

Pipeline position:
  AI analysis → AnalysisVerifier.verify() → KPIGenerator / ChartGenerator
"""

import logging
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.analysis_verifier")

# Aggregations that make no sense on ID-like or high-cardinality columns
BAD_ID_AGGREGATIONS = {"sum", "mean", "min", "max", "median"}

# Max counts per chart type to enforce diversity
MAX_BAR   = 3
MAX_PIE   = 2
MAX_HIST  = 2
MAX_SCATTER = 2

# Threshold: if unique_ratio exceeds this, treat column as ID even if not named so
HIGH_CARD_ID_RATIO = 0.95


class AnalysisVerifier:
    """
    Verifies and repairs AI-generated analysis output.
    Never calls an external API — all rules are deterministic.
    """

    def verify(self, ai_analysis: dict, profile: dict) -> dict:
        """
        Run all verification passes on ai_analysis.

        Returns a cleaned copy of ai_analysis with:
          - Bad KPIs removed / aggregations corrected
          - Bad charts removed / duplicates dropped
          - Chart diversity enforced
          - Count-per-category chart injected if missing
          - Insight quality filtered
        """
        if not isinstance(ai_analysis, dict):
            return ai_analysis

        result = {
            "kpis":     list(ai_analysis.get("kpis", [])),
            "charts":   list(ai_analysis.get("charts", [])),
            "insights": list(ai_analysis.get("insights", [])),
            "source":   ai_analysis.get("source", "ai"),
        }

        id_cols      = set(profile.get("id_cols", []))
        binary_cols  = set(profile.get("binary_cols", []))
        num_cols     = set(profile.get("numerical_cols", []))
        cat_cols     = set(profile.get("categorical_cols", []))
        all_cols     = set(c["name"] for c in profile.get("columns", []))

        # Extended ID detection: columns not explicitly flagged but with very high cardinality
        for col_meta in profile.get("columns", []):
            if col_meta.get("unique_ratio", 0) >= HIGH_CARD_ID_RATIO and col_meta["type"] == "numerical":
                id_cols.add(col_meta["name"])

        result["kpis"]     = self._verify_kpis(result["kpis"], id_cols, binary_cols, num_cols, all_cols)
        result["charts"]   = self._verify_charts(result["charts"], id_cols, binary_cols, cat_cols, num_cols, all_cols, profile)
        result["charts"]   = self._inject_missing_charts(result["charts"], cat_cols, num_cols, id_cols, binary_cols, profile)
        result["charts"]   = self._enforce_diversity(result["charts"])
        result["insights"] = self._verify_insights(result["insights"], id_cols)

        n_kpi_removed    = len(ai_analysis.get("kpis", [])) - len(result["kpis"])
        n_chart_removed  = len(ai_analysis.get("charts", [])) - len(result["charts"])
        logger.info(
            f"Verifier: removed {n_kpi_removed} KPIs, "
            f"{n_chart_removed} chart issues fixed, "
            f"{len(result['charts'])} charts final"
        )
        return result

    def validate_single_chart(self, chart: dict, profile: dict) -> dict | None:
        """
        Validate and repair a single chart definition.
        Returns a cleaned chart dict or None if invalid.
        """
        if not isinstance(chart, dict):
            return None

        id_cols      = set(profile.get("id_cols", []))
        binary_cols  = set(profile.get("binary_cols", []))
        num_cols     = set(profile.get("numerical_cols", []))
        cat_cols     = set(profile.get("categorical_cols", []))
        all_cols     = set(c["name"] for c in profile.get("columns", []))

        cleaned = self._verify_charts([chart], id_cols, binary_cols, cat_cols, num_cols, all_cols, profile)
        if not cleaned:
            return None
        return cleaned[0]

    # ── KPI Verification ───────────────────────────────────────

    def _verify_kpis(self, kpis, id_cols, binary_cols, num_cols, all_cols):
        cleaned = []
        seen_cols = set()

        for kpi in kpis:
            col = kpi.get("column", "")
            agg = kpi.get("aggregation", "sum")

            # Skip if column doesn't exist
            if col and col not in all_cols and col != "_count":
                logger.debug(f"KPI skip: column '{col}' not in dataset")
                continue

            # Skip ID columns entirely
            if col in id_cols:
                logger.debug(f"KPI skip: '{col}' is ID column")
                continue

            # Skip non-numerical columns — KPIs require aggregatable data
            if col not in num_cols and col != "_count":
                logger.debug(f"KPI skip: '{col}' is not numerical (type check)")
                continue

            # Fix binary columns — sum(target) → count, mean(target) → count
            if col in binary_cols:
                kpi = dict(kpi)
                if agg in ("sum", "mean"):
                    kpi["aggregation"] = "count"
                    kpi["label"] = f"Total {kpi.get('label', col).replace('Average ', '').replace('Sum of ', '')}"
                    logger.debug(f"KPI fix: binary col '{col}' agg→count")

            # Fix: sum on a non-total numerical col (e.g. year, score, age) → mean
            if col in num_cols and agg == "sum":
                col_lower = col.lower()
                is_total_col = any(k in col_lower for k in ["revenue", "sales", "amount", "cost", "profit", "total", "income", "spend", "price"])
                if not is_total_col:
                    kpi = dict(kpi)
                    kpi["aggregation"] = "mean"
                    logger.debug(f"KPI fix: '{col}' sum→mean (not a total col)")

            # Deduplicate by column
            if col in seen_cols:
                logger.debug(f"KPI skip: duplicate column '{col}'")
                continue
            seen_cols.add(col)

            cleaned.append(kpi)

        return cleaned

    # ── Chart Verification ─────────────────────────────────────

    def _verify_charts(self, charts, id_cols, binary_cols, cat_cols, num_cols, all_cols, profile):
        cleaned = []
        seen_combos = set()

        for chart in charts:
            x = chart.get("x", "")
            y = chart.get("y")
            ctype = chart.get("type", "bar")

            # Skip charts with missing/nonexistent columns
            if x and x not in all_cols:
                logger.debug(f"Chart skip: x='{x}' not in dataset")
                continue
            if y and y not in all_cols:
                logger.debug(f"Chart skip: y='{y}' not in dataset")
                continue

            # Skip charts where x IS an ID column — never useful
            if x in id_cols:
                logger.debug(f"Chart skip: x='{x}' is ID column")
                continue

            # Skip charts where y IS an ID column
            if y and y in id_cols:
                logger.debug(f"Chart skip: y='{y}' is ID column")
                continue

            chart = dict(chart)

            # Fix: high-cardinality categorical x → should be histogram not bar
            x_meta = next((c for c in profile.get("columns", []) if c["name"] == x), None)
            if x_meta and x_meta.get("n_unique", 0) > 50 and ctype == "bar" and not y:
                chart["type"] = "histogram"
                logger.debug(f"Chart fix: high-card bar→histogram for '{x}'")

            # Fix: binary y with scatter → switch to bar
            if y and y in binary_cols and ctype == "scatter":
                chart["type"] = "bar"
                logger.debug(f"Chart fix: binary y scatter→bar for '{x}'vs'{y}'")

            # Fix: same column for x and y
            if x and y and x == y:
                chart["y"] = None
                chart["type"] = "histogram"
                logger.debug(f"Chart fix: x==y '{x}', switched to histogram")

            # Deduplicate — use the MUTATED chart values, not the original x/y variables
            dedup_key = (chart["type"], chart.get("x"), chart.get("y"), chart.get("color"))
            if dedup_key in seen_combos:
                logger.debug(f"Chart skip: duplicate combo {dedup_key}")
                continue
            seen_combos.add(dedup_key)

            cleaned.append(chart)

        return cleaned

    # ── Chart Injection ────────────────────────────────────────

    def _inject_missing_charts(self, charts, cat_cols, num_cols, id_cols, binary_cols, profile):
        """
        If no 'count per category' chart exists and categorical columns are present,
        inject one. This covers the common case where AI sums a numeric instead of
        counting records per category.
        """
        existing_x_cols = {c.get("x") for c in charts}

        # Find a categorical column not already used as x
        usable_cat = [c for c in cat_cols if c not in id_cols and c not in existing_x_cols]

        if usable_cat and len(charts) < AppConfig.MAX_CHARTS:
            col = usable_cat[0]
            charts.append({
                "type": "bar",
                "x": col,
                "y": None,
                "title": f"Records by {col.replace('_', ' ').title()}",
                "description": f"Count of records in each {col} category",
                "_injected": True,
            })
            logger.debug(f"Injected count-per-category chart for '{col}'")

        return charts

    # ── Diversity Enforcement ──────────────────────────────────

    def _enforce_diversity(self, charts):
        """
        Enforce per-type limits to prevent all charts being the same type.
        If over the limit, less informative duplicates are dropped.
        """
        type_counts = {"bar": 0, "pie": 0, "histogram": 0, "scatter": 0, "line": 0}
        limits = {"bar": MAX_BAR, "pie": MAX_PIE, "histogram": MAX_HIST, "scatter": MAX_SCATTER}
        result = []

        for chart in charts:
            ctype = chart.get("type", "bar")
            limit = limits.get(ctype, 99)
            if type_counts.get(ctype, 0) < limit:
                result.append(chart)
                type_counts[ctype] = type_counts.get(ctype, 0) + 1
            else:
                logger.debug(f"Chart diversity: dropping extra {ctype} chart")

        # If fewer than MIN_CHARTS remain, the gap will be filled by ChartGenerator's rule-based fallback
        return result

    # ── Insight Verification 

    def _verify_insights(self, insights, id_cols):
        """
        Remove low-quality insights:
          - Generic ones with no numbers
          - Ones that reference ID columns
          - Near-duplicates (first 60 chars overlap)
        """
        cleaned = []
        seen_prefixes = set()

        for ins in insights:
            if not isinstance(ins, str) or len(ins.strip()) < 15:
                continue

            # Drop insights about ID columns
            skip = False
            for id_col in id_cols:
                if id_col.lower() in ins.lower():
                    logger.debug(f"Insight skip: mentions ID col '{id_col}'")
                    skip = True
                    break
            if skip:
                continue

            # Drop near-duplicates
            prefix = ins.strip()[:60].lower()
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)

            cleaned.append(ins.strip())

        return cleaned
