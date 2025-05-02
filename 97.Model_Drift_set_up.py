import numpy as np

def detect_drift(current_mean,baseline_mean, threshold=0.1):
    drift=abs(current_mean-baseline_mean)
    return drift > threshold

# Example:
baseline_mean=0.5
current_mean=0.7
if detect_drift(current_mean, baseline_mean):
    print("Data drift detected! Retraining needed.")
    