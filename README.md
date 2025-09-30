# 🏦 Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

A production-ready machine learning system that predicts bank customer churn with 86%+ accuracy.

## 🎯 Quick Demo
```python
from src.prediction import predictor

result = predictor.predict({
    'credit_score': 650,
    'age': 42,
    'balance': 125000,
    # ... other features
})
print(f"Churn probability: {result['churn_probability']:.2%}")