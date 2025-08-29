# Shrinkage Regression in Python

This repository contains a **Python implementation of shrinkage regression estimators**, translated from the original **R code and methodology** provided by Asimit, V., Cidota, MA., Chen, Z., and Asimit, J. in [\[savvySh\]](https://github.com/Ziwei-ChenChen/savvySh/tree/main).  

The goal of this project is to make these methods more accessible in Python, and to apply them in a **quantitative finance context**, specifically for **commodity futures prediction and portfolio construction**.  

---

## 🔹 Background

Shrinkage regression is a class of estimators designed to improve prediction accuracy by trading off bias and variance.  
This project implements:  

- **Stein Estimator (St)**  
- **Diagonal Shrinkage (DSh)**  
- **Shrinkage Estimator**
- **Linear Shrinkage (LSh)**  
- **Slab Regression (SR, GSR)**  
- **Shrinkage Ridge Regression (SRR)**  

Note that although all of the methods described are shrinkage estimators, "Shrinkage Estimator" (capital S) is a specific type of shrinkage method mentioned in Theorem 1. (iii), denoted as *Sh*, in the reference paper.

---

## 🔹 My Contribution

- Translated the original **R implementations** into **Python**.  
- Verified correctness against the R version on toy datasets.  
- Applied these methods to a **dataset of 35 commodity futures**, predicting future returns.   
- Compared shrinkage estimators against **OLS** in terms of **out-of-sample MSE**.  
- Constructed **long-short portfolios** based on predicted returns.

Currently doing:
- Considering liquidity constraint since some commodity contracts could not be shorted easily
- Applying shrinkage to GLM

---

## 🔹 Results (Example)

### Mean Square Error of Prediction

Using a rolling-window backtest (3 years training → 3 months prediction) and applying Stein estimators and Diagonal Shrinkage regression.  
Note that for simplicity, here we only consider front-month contract (cc1).

MSE of Prediction (over each commodity)
- OLS: mean 0.0374, std 0.0493 (over each year)  
- **Stein**: **mean 0.0240, std 0.0283 (over each year)**
- **Diagonal Shrinkage**: **mean 0.0233, std 0.0288 (over each year)** 
- **Shrinkage Estimator**: **mean 0.0111, std 0.0095(over each year)**
- **Linear Shrinkage**: **mean 0.0279, std 0.0314(over each year)**
- **Slab**: **mean 0.0325, std 0.0357(over each year)**
- Generalized Slab: mean 0.3327, std 1.2484(over each year)
- **Shrinkage Ridge**: **mean 0.0269, std 0.0327(over each year)**

We could see that all shrinkage method except GSR perform better than OLS, while Shrinkage estimator is the best, then Stein and Diagonal Shrinkage, then Linear Shrinkage, then Slab.  

### Long-short Portfolio

To backtest a long-short portfolio based on the prediction of return, we now introduced second-nearby contract data to bring us more close to reality.

Annualized Return & Sharpe Ratio
- OLS: 3.73% / 0.5227
- Stein: 4.45% / 0.5506
- **Diagonal Shrinkage**: **5.88% / 0.7886**
- Shrinkage Estimator: 4.02% / 0.3806
- Slab Shrinkage: 3.87% / 0.5318
- Generalized Slab: 4.19% / 0.5863
- Linear Shrinkage: 4.67% / 0.6096
- **Shrinkage Ridge**: **5.43% / 0.7081**

We could see that all shrinkage methods are with higher return than OLS, where Diagonal Shrinkage and Shrinkage Ridge Regression are the top-2 models.  
Notice that Shrinkage estimator's sharpe ratio is notably lower than OLS in spite of higher annualized return, which implies a far higher standard deviation of returns. 

---

## 🔹 Future Improvement

While there has been some initial results achieved, obviously there are still many things to do, including but not limit to:
- Data issue: Adding more commodity futures data; Try other index or other target's futures
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
