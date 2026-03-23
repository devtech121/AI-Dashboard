"""
Anomaly Detector Module
Statistical anomaly detection using Z-score and IQR methods
"""

import logging
import numpy as np
import pandas as pd
from utils.config import AppConfig

logger = logging.getLogger("ai_dashboard.anomaly_detector")


class AnomalyDetector:
    """Detects statistical anomalies in numerical columns.

    Issue #6: NOT decorated with @st.cache_data. app.py caches in st.session_state.
    """

    def detect(self, df: pd.DataFrame, profile: dict) -> dict:
        """
        Detect anomalies using Z-score and IQR methods.
        Returns dict keyed by column name with anomaly info.
        """
        num_cols = [
            c for c in profile.get("numerical_cols", [])
            if c not in profile.get("id_cols", [])
        ]

        results = {}
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            try:
                anomaly_info = self._detect_column(series, col)
                if anomaly_info["n_anomalies"] > 0:
                    results[col] = anomaly_info
            except Exception as e:
                logger.warning(f"Anomaly detection failed for {col}: {e}")

        logger.info(f"Anomaly detection: {len(results)} columns with anomalies")
        return results

    def get_anomaly_mask(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Return boolean mask for anomalous rows in a column."""
        series = df[col].dropna()
        if len(series) < 10:
            return pd.Series(False, index=df.index)

        z_mask = self._zscore_mask(series)
        iqr_mask = self._iqr_mask(series)
        combined = z_mask | iqr_mask

        # Reindex to full df
        full_mask = pd.Series(False, index=df.index)
        full_mask[combined.index[combined]] = True
        return full_mask

    # ── Private Methods ────────────────────────────────────────

    def _detect_column(self, series: pd.Series, col_name: str) -> dict:
        """Detect anomalies in a single series."""
        z_mask = self._zscore_mask(series)
        iqr_mask = self._iqr_mask(series)
        combined_mask = z_mask | iqr_mask

        n_anomalies = int(combined_mask.sum())
        pct = n_anomalies / len(series) * 100

        anomaly_values = series[combined_mask]

        return {
            "col": col_name,
            "n_anomalies": n_anomalies,
            "pct": round(pct, 2),
            "min_anomaly": float(anomaly_values.min()) if len(anomaly_values) else None,
            "max_anomaly": float(anomaly_values.max()) if len(anomaly_values) else None,
            "method": "z-score + IQR",
            "z_count": int(z_mask.sum()),
            "iqr_count": int(iqr_mask.sum()),
        }

    def _zscore_mask(self, series: pd.Series) -> pd.Series:
        """Identify outliers using Z-score method."""
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(False, index=series.index)
        z_scores = np.abs((series - mean) / std)
        return z_scores > AppConfig.Z_SCORE_THRESHOLD

    def _iqr_mask(self, series: pd.Series) -> pd.Series:
        """Identify outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(False, index=series.index)
        lower = q1 - AppConfig.IQR_MULTIPLIER * iqr
        upper = q3 + AppConfig.IQR_MULTIPLIER * iqr
        return (series < lower) | (series > upper)