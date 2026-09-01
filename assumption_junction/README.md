# Name: Sung-Tse Wu (Jay)

## Dataset

This project uses the Telco Customer Churn dataset from Kaggle: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>.

Each row represents one telecom customer, with account information, subscribed services, contract type, payment method, tenure, monthly charges, total charges, and whether the customer churned.

The task is to build interpretable models that predict customer churn and explain which factors are associated with higher or lower churn risk.

Notebook: `telco_churn_assumption_junction.ipynb`

## Assumption Checks

| Model | Key Assumptions Checked | Evidence | Concern |
|---|---|---|---|
| Linear regression | Linearity, residual patterns, multicollinearity, constant variance, normal residuals | LOWESS plots, residuals vs fitted values, Breusch-Pagan test, QQ plot, VIF table | Churn is binary, so residual normality and constant variance are not realistic. Predicted values can also fall outside 0 to 1. |
| Logistic regression | Binary target, linearity in the logit, multicollinearity, probability calibration | Churn encoded as 0/1, binned-logit plots, VIF table, calibration table, Brier score | Some numeric relationships, especially tenure, appear non-linear. Logistic regression may miss curved effects unless features are transformed. |
| GAM | Binary target, additive structure, smooth numeric effects, calibration | Binomial GAM, spline terms for tenure, MonthlyCharges, and TotalCharges, calibration table, residual plots, partial dependence plots | GAM is more flexible but still additive. It does not automatically capture interactions such as tenure combined with contract type. |

## Model Comparison

| Model | Performance Evidence | Interpretability Strength | Interpretability Weakness |
|---|---|---|---|
| Linear regression | ROC AUC 0.832, Brier 0.142, accuracy 0.800, recall 0.537, F1 0.588 | Coefficients are simple additive changes in predicted churn probability | Poor fit for a binary target and can produce invalid probabilities |
| Logistic regression | ROC AUC 0.840, Brier 0.171, accuracy 0.733, recall 0.797, F1 0.613 | Odds ratios are clear and easy to explain to business stakeholders | Assumes numeric features are linear on the log-odds scale |
| GAM | ROC AUC 0.842, Brier 0.137, accuracy 0.797, recall 0.531, F1 0.581 | Partial dependence curves show non-linear churn patterns | More complex to explain and monitor than logistic regression |

## Recommendation

Recommended model: Logistic regression as the first business-facing churn model, with GAM used as a supporting model to study non-linear effects.

Why this model: This recommendation is based on the executed notebook results. Logistic regression had the best recall at 0.797 and the best F1 score at 0.613, which is useful for retention outreach because missing a likely churner may be costly. GAM had the best ROC AUC at 0.842 and the best Brier score at 0.137, so it is a strong supporting model when probability ranking or calibration is the priority.

What the company can responsibly conclude: The models identify customer characteristics associated with churn risk, such as contract type, tenure, payment method, internet service, and support/security services. These signals can help prioritize retention outreach.

What the company should not conclude yet: The analysis does not prove causality. It should not be used to claim that changing one feature will automatically prevent churn, and it does not yet test fairness, pricing history, competitor activity, or time-based churn dynamics.

One next analysis we would run: Tune the classification threshold using the retention team's outreach capacity and the cost of false positives versus false negatives. If customer signup and churn dates are available, also run time-aware validation or survival analysis.
