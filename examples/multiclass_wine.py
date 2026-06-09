"""Multiclass example."""

from sklearn.datasets import load_wine

from separatix import diagnose

data = load_wine()
report = diagnose(data.data, data.target, return_report=True, random_state=0)
print(report.class_summary)
print(report.recommendation_text)
