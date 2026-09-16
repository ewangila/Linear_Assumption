import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

class LinearAssumptionTester:
    """Class to test OLS Gauss-Markov assumptions."""
    
    def __init__(self, x: pd.Series, y: pd.Series):
        self.x = sm.add_constant(x)
        self.y = y
        self.model = None
        self.predictions = None
        self.residuals = None
        sns.set_theme(style="whitegrid")

    def fit(self):
        """Fit OLS model and extract predictions/residuals."""
        self.model = sm.OLS(self.y, self.x).fit()
        self.predictions = self.model.predict(self.x)
        self.residuals = self.model.resid
        return self.model

    def run_statistical_tests(self) -> Dict[str, Any]:
        """Run formal statistical tests for the assumptions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        results = {}

        # 1. Independence (Durbin-Watson)
        results['Independence (Durbin-Watson)'] = {'Statistic': round(durbin_watson(self.residuals), 3)}

        # 2. Normality (Shapiro-Wilk)
        shapiro_stat, shapiro_p = stats.shapiro(self.residuals)
        results['Normality (Shapiro-Wilk)'] = {'Statistic': round(shapiro_stat, 3), 'p-value': round(shapiro_p, 4)}

        # 3. Homoscedasticity (Breusch-Pagan)
        bp_stat, bp_p, _, _ = sms.het_breuschpagan(self.residuals, self.model.model.exog)
        results['Homoscedasticity (Breusch-Pagan)'] = {'Statistic': round(bp_stat, 3), 'p-value': round(bp_p, 4)}

        # 4. Linearity (Residual Mean Check)
        results['Linearity (Residual Mean)'] = {'Mean': round(np.mean(self.residuals), 5)}

        return results

    def plot_diagnostics(self):
        """Generate 2x2 grid of diagnostic plots."""
        if self.model is None:
            raise ValueError("Model not fitted.")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Linear Regression Assumptions Diagnostics', fontsize=16, weight='bold')

        # Linearity: Residuals vs Fitted
        sns.residplot(x=self.predictions, y=self.residuals, lowess=True, 
                      scatter_kws={'alpha': 0.6, 'edgecolor': 'k'}, 
                      line_kws={'color': 'red', 'lw': 2}, ax=axes[0, 0])
        axes[0, 0].set_title('Linearity: Residuals vs Fitted', fontsize=12)

        # Normality: Q-Q Plot
        sm.qqplot(self.residuals, line='45', fit=True, ax=axes[0, 1], alpha=0.6)
        axes[0, 1].set_title('Normality: Q-Q Plot', fontsize=12)

        # Homoscedasticity: Scale-Location Plot
        standardized_res = self.model.get_influence().resid_studentized_internal
        sns.regplot(x=self.predictions, y=np.sqrt(np.abs(standardized_res)), scatter=True, lowess=True, 
                    scatter_kws={'alpha': 0.6, 'edgecolor': 'k'}, line_kws={'color': 'red', 'lw': 2}, ax=axes[1, 0])
        axes[1, 0].set_title('Homoscedasticity: Scale-Location Plot', fontsize=12)

        # Independence: Residuals vs Order
        axes[1, 1].plot(self.residuals.index, self.residuals, color='teal', marker='o', linestyle='-', alpha=0.6)
        axes[1, 1].axhline(0, color='red', linestyle='--')
        axes[1, 1].set_title('Independence: Residuals vs Order', fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()


if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    ad_spend = pd.Series(np.random.uniform(10, 150, 200), name="Ad_Spend")
    sales = pd.Series(50 + 3.2 * ad_spend + np.random.normal(0, 25, 200), name="Weekly_Sales")

    # Run diagnostics
    analyzer = LinearAssumptionTester(x=ad_spend, y=sales)
    model_results = analyzer.fit()
    
    print(model_results.summary().tables[1])
    
    for test_name, metrics in analyzer.run_statistical_tests().items():
        print(f"{test_name.ljust(35)} | {metrics}")
        
    analyzer.plot_diagnostics()