# Linear Assumption Tester

**Object-oriented Python toolkit for validating the Gauss-Markov assumptions of Ordinary Least Squares (OLS) regression.**

This repository provides a clean, reusable class (`LinearAssumptionTester`) that automates both formal statistical tests and visual diagnostics for the four key assumptions of linear regression:

- **Linearity**
- **Independence** (no autocorrelation)
- **Normality** of residuals
- **Homoscedasticity** (constant variance)

A fully documented Jupyter notebook demonstrates the tool on a realistic retail analytics use case (predicting weekly sales from advertising spend).

---

## Features

- Clean OOP design — encapsulate model fitting, testing, and plotting in a single class
- Formal statistical tests:
  - Durbin-Watson (Independence)
  - Shapiro-Wilk (Normality)
  - Breusch-Pagan (Homoscedasticity)
  - Residual mean check (Linearity)
- Publication-ready 2×2 diagnostic plot grid
- Ready-to-run Jupyter notebook with synthetic retail data
- Minimal, well-documented dependencies

---

## Installation

```bash
git clone https://github.com/ewangila/Linear_Assumption.git
cd Linear_Assumption
pip install -r requirements.txt
```
**Requirements:**

- Python ≥ 3.9
- numpy, pandas, statsmodels, scipy, matplotlib, seaborn

---

## Quick Start

```python

import numpy as np
import pandas as pd
from linear_assumption_tester import LinearAssumptionTester

# Generate synthetic data
np.random.seed(42)
ad_spend = pd.Series(np.random.uniform(10, 150, 200), name="Ad_Spend")
sales = pd.Series(50 + 3.2 * ad_spend + np.random.normal(0, 25, 200), name="Weekly_Sales")

# Run diagnostics
analyzer = LinearAssumptionTester(x=ad_spend, y=sales)
model = analyzer.fit()

# Statistical tests
print(model.summary().tables[1])
for test, results in analyzer.run_statistical_tests().items():
    print(f"{test:<35} | {results}")

# Visual diagnostics
analyzer.plot_diagnostics()

```

---

## Notebook

Open `Linear_Assumption.ipynb` for a complete walkthrough:
1. Class definition and documentation  
2. Synthetic data generation  
3. Model fitting + statistical validation  
4. Visual residual diagnostics  

---

## Project Structure

```
Linear_Assumption/
├── Linear_Assumption.ipynb          # Full demonstration notebook
├── linear_assumption_tester.py      # Core class
├── requirements.txt
├── LICENSE                          # MIT
└── README.md

```

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
