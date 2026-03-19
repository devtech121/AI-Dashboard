"""
Insights Generator Module
Combines AI insights with statistical observations for comprehensive analysis
"""

import logging
import numpy as np
import pandas as pd
from utils.helpers import format_number

logger = logging.getLogger("ai_dashboard.insights_generator")


class InsightsGenerator:
    """Generates data-driven insights combining AI and statistical analysis."""

    def generate(self, df: pd.DataFrame, profile: dict, ai_analysis: dict, anomalies: dict) -> list:
        """
        Generate a final list of insights.
        
        Returns list of insight strings.
        """
        insights = []

        # 1. AI insights
        ai_insights = ai_analysis.get("insights", [])
        for insight in ai_insights:
            if isinstance(insight, str) and len(insight.strip()) > 10:
                insights.append(insight.strip())

        # 2. Statistical insights
        stat_insights = self._generate_statistical_insights(df, profile)
        insights.extend(stat_insights)

        # 3. Anomaly insights
        if anomalies:
            anom_insights = self._generate_anomaly_insights(anomalies)
            insights.extend(anom_insights)

        # 4. Data quality insights
        quality_insights = self._generate_quality_insights(profile)
        insights.extend(quality_insights)

        # Deduplicate and limit
        seen = set()
        unique_insights = []
        for ins in insights:
            key = ins[:50].lower()
            if key not in seen:
                seen.add(key)
                unique_insights.append(ins)

        logger.info(f"Generated {len(unique_insights)} insights")
        return unique_insights[:12]

    # ── Private Methods ────────────────────────────────────────

    def _generate_statistical_insights(self, df: pd.DataFrame, profile: dict) -> list:
        """Generate insights from statistical analysis."""
        insights = []
        num_cols = [c for c in profile.get("numerical_cols", []) if c not in profile.get("id_cols", [])]
        cat_cols = profile.get("categorical_cols", [])

        # Dataset overview
        insights.append(
            f"Dataset contains {profile['row_count']:,} records with "
            f"{profile['column_count']} attributes ({len(num_cols)} numerical, "
            f"{len(cat_cols)} categorical)."
        )

        # Numerical insights
        for col in num_cols[:3]:
            try:
                series = df[col].dropna()
                if len(series) == 0:
                    continue

                mean_val = series.mean()
                std_val = series.std()
                cv = (std_val / mean_val * 100) if mean_val != 0 else 0

                if cv > 50:
                    insights.append(
                        f"'{col}' shows high variability (CV={cv:.0f}%), suggesting "
                        f"significant spread from {format_number(series.min())} to {format_number(series.max())}."
                    )
                else:
                    insights.append(
                        f"'{col}' averages {format_number(mean_val)} with values ranging "
                        f"from {format_number(series.min())} to {format_number(series.max())}."
                    )
            except Exception:
                continue

        # Categorical insights
        for col in cat_cols[:2]:
            try:
                vc = df[col].value_counts()
                if len(vc) == 0:
                    continue
                top_val = vc.index[0]
                top_pct = vc.iloc[0] / len(df) * 100
                insights.append(
                    f"'{col}' has {len(vc)} unique values; "
                    f"'{top_val}' is the most frequent ({top_pct:.1f}% of records)."
                )
            except Exception:
                continue

        # Correlation insights
        if len(num_cols) >= 2:
            try:
                corr = df[num_cols].corr()
                high_corr = []
                cols = list(corr.columns)
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        r = corr.iloc[i, j]
                        if abs(r) > 0.7:
                            direction = "positively" if r > 0 else "negatively"
                            high_corr.append(f"'{cols[i]}' and '{cols[j]}' ({direction} correlated, r={r:.2f})")

                if high_corr:
                    insights.append(f"Strong correlations detected: {'; '.join(high_corr[:2])}.")
            except Exception:
                pass

        return insights

    def _generate_anomaly_insights(self, anomalies: dict) -> list:
        """Generate insights from anomaly detection results."""
        insights = []
        for col, info in list(anomalies.items())[:3]:
            n = info.get("n_anomalies", 0)
            pct = info.get("pct", 0)
            if n > 0:
                insights.append(
                    f"⚠️ Anomaly Alert: '{col}' contains {n:,} outlier values "
                    f"({pct:.1f}% of data) — these may represent data quality issues or rare events."
                )
        return insights

    def _generate_quality_insights(self, profile: dict) -> list:
        """Generate data quality observations."""
        insights = []
        missing = profile.get("missing_summary", {})

        if missing:
            high_missing = {k: v for k, v in missing.items() if v["pct"] > 20}
            if high_missing:
                cols_list = ", ".join([f"'{c}' ({v['pct']:.0f}%)" for c, v in list(high_missing.items())[:3]])
                insights.append(
                    f"Data completeness concern: {len(high_missing)} column(s) have >20% missing values — {cols_list}."
                )
            else:
                low_missing = list(missing.keys())
                insights.append(
                    f"Minor missing data in {len(low_missing)} column(s): {', '.join(low_missing[:3])}. "
                    f"Overall data quality appears good."
                )
        else:
            insights.append("✅ Excellent data completeness — no missing values detected across all columns.")

        if profile.get("duplicate_count", 0) > 0:
            insights.append(
                f"Note: {profile['duplicate_count']:,} duplicate rows were detected and removed during preprocessing."
            )

        return insights
