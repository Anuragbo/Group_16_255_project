"""Dashboard component registry."""

from . import anomaly_detection, customer_segments, eda_olap, predictive_modeling

__all__ = [
    "anomaly_detection",
    "customer_segments",
    "eda_olap",
    "predictive_modeling",
]
