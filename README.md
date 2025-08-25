# Shrinkage Regression in Python

This repository contains a **Python implementation of shrinkage regression estimators**, translated from the original **R code and methodology** provided by Asimit, V., Cidota, MA., Chen, Z., and Asimit, J. in [\[savvySh\]](https://github.com/Ziwei-ChenChen/savvySh/tree/main).  

The goal of this project is to make these methods more accessible in Python, and to apply them in a **quantitative finance context**, specifically for **commodity futures prediction and portfolio construction**.  

---

## 🔹 Background

Shrinkage regression is a class of estimators designed to improve prediction accuracy by trading off bias and variance.  
This project implements:  

- **Stein Estimator (St)**  
- **Diagonal Shrinkage (DSh)**  
- **Linear Shrinkage (LSh)**  
- **Slab Regression (SR, GSR)**  
- **Shrinkage Ridge Regression (SRR)**  

---

## 🔹 My Contribution

- Translated the original **R implementations** into **Python**.  
- Verified correctness against the R version on toy datasets.  
- Applied these methods to a **dataset of 35 commodity futures**, predicting future returns.  
- Constructed **long-short portfolios** based on predicted returns.  
- Compared shrinkage estimators against **OLS** in terms of **out-of-sample RMSE** and **Sharpe ratio**.  

---

## 🔹 Results (Example)

Using a rolling-window backtest (3 years training → 1 months prediction):  

- **OLS Sharpe Ratio**: 1.228  
- **Diagonal Shrinkage Sharpe Ratio**: **1.346 (+0.118 improvement)**  
- Stein estimator exhibited higher variance but provided robustness in certain regimes.  

---

## 🔹 Usage

```python
from model.savvy_sh import SavvySh

# Fit a multiplicative shrinkage model
model = SavvySh(model_class="Multiplicative", include_Sh=True)
result = model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
```

---

## 🔹 References

- Original R code: [\[saavySh\]](https://github.com/Ziwei-ChenChen/savvySh/tree/main)  

---

## 🔹 Notes

⚠️ This repository is **not an original research contribution**.  
It is a **Python translation and application** of existing R methods.  
All credit for the methodology belongs to Asimit, V., Cidota, MA., Chen, Z., and Asimit, J..  
My contribution is in **implementation, validation, and empirical application** to financial data.  
