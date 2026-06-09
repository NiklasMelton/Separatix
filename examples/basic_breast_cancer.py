"""Basic binary classification example."""

from sklearn.datasets import load_breast_cancer

from separatix import diagnose

data = load_breast_cancer()
report = diagnose(data.data, data.target, return_report=True, random_state=0)
print(report.recommendation_text)
print(report.scores)
