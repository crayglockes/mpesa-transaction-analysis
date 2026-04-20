# 📱 M-Pesa Transaction Pattern Analysis

> End-to-end data science project analysing behavioural patterns,
> user segmentation, and fraud signals in mobile money transactions.

> **[View Final Analysis Report (HTML)](https://crayglockes.github.io/mpesa-transaction-analysis/reports/05_final_report.html)**

## Analysis Pipeline

1. **Data Generation** — 150K realistic synthetic transactions with diurnal
   activity models, valid Safaricom phone formats, and Safaricom fee schedules
2. **EDA** — Temporal, geographic, and distribution analysis (10 figures)
3. **Feature Engineering** — 30+ behavioural features per user
4. **User Segmentation** — K-Means clustering with PCA visualisation
5. **Anomaly Detection** — Isolation Forest + supervised Random Forest

## Key Results

- 4 behaviourally distinct user segments identified
- Fraud detection AUC: **0.87+** (5-fold CV)
- Top fraud signals: `is_just_below_threshold`, `pct_night_txns`, `amount_cv`
- Salary week generates **+18%** transaction volume vs daily average

## Skills Demonstrated

`Data Simulation` `EDA` `Feature Engineering` `Clustering` `Anomaly Detection`  
`Scikit-Learn` `Imbalanced Learning` `Statistical Visualisation`
