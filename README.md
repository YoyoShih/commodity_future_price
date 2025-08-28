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
- Compared shrinkage estimators against **OLS** in terms of **out-of-sample MSE**.  
- Constructed **long-short portfolios** based on predicted returns.

Currently doing:
 

---

## 🔹 Results (Example)

### Mean Square Error of Prediction

Using a rolling-window backtest (3 years training → 1 months prediction) and applying Stein estimators and Diagonal Shrinkage regression.  
Note that for simplicity, here we only consider front-month contract (cc1).

MSE of Prediction (over each commodity)
- OLS: mean 0.0261, std 0.0519 (over each year)  
- **Stein**: **mean 0.0183, std 0.0348 (over each year)**
- **Diagonal Shrinkage**: **mean 0.0186, std 0.0366 (over each year)** 

Both shrinkage methods lead to lower mean and std of MSE, while Diagonal Shrinkage is with slightly higher value on both measure.

### Long-short Portfolio

To backtest a long-short portfolio based on the prediction of return, we now introduced second-nearby contract data to bring us more close to reality.

Annualized Return & Sharpe Ratio
- OLS: 32.23% / 2.8523
- Stein: 30.85% / 2.7349
- **Diagonal Shrinkage**: **31.67% / 3.0752**

We could see that although Stein is with lower mean and std than OLS, the annualized return and sharpe ratio is lower than OLS.  
On the other hand, DSh leads to a higher sharpe ratio despite of slightly lower return. Surely it is because of DSh is more stable as a regression method than OLS is.  
Stein, also a shrinkage method though, views all covariates equally reliable, which is too simple in reality, especially when dimension is not very low.  
DSh, comparing to Stein, consider the volatility of each covariates and with more power of shrinkage when it higher volatile and vice versa.  It therefore performs better than Stein.

---

## 🔹 Future Improvement

While there has been some initial results achieved, obviously there are still many things to do, including but not limit to:
- Data issue: Adding more commodity futures data; Data correctness issue solving; Try other index or other target's futures
- Model: Try GLM instead of MLS
- Strategy: Considering more complex trading strategies

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
