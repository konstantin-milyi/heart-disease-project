# Heart Disease Prediction & Interpretability Analysis

This project focuses on data analysis and the development of ML models to predict the risk of cardiovascular disease. The main focus is not only on performance metrics but also on the medical interpretability of the results using SHAP, Partial Dependence, and deep Exploratory Data Analysis (EDA).

In medical diagnostics, the cost of an error is asymmetrical: missing a sick patient (False Negative) is much more dangerous than sending a healthy one for further examination (False Positive). Therefore, Recall (Sensitivity) was chosen as the key metric for model optimization.

![hrp](png\hrp.png)

## About the Data

The project uses the classic Cleveland Heart Disease dataset (from the UCI repository).

- **Size:** $303$ patients, $14$ features.
- **Class Balance:** Classes are perfectly balanced (about 46% positive class after cleaning).
- **Target Variable:** Binarized ($0$ = Healthy, $1$ = Diseased, combining $4$ stages of the disease).

## Tech Stack

- **Data Manipulation:** `pandas`, `numpy`, `scipy`
- **Visualization:** `matplotlib`, `seaborn`
- **Feature Engineering:** `feature-engine`, `category_encoders`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Interpretability (XAI):** `shap`, `PartialDependenceDisplay`, `permutation_importance`

## Project Structure

### 1. Exploratory Data Analysis (EDA)

- **Preprocessing:** Handling missing values and applying the Yeo-Johnson transformation to normalize skewed distributions (e.g., blood pressure and cholesterol levels).
- **Multidimensional Analysis:** Using Pairplots and 3D Scatter plots to identify risk clusters (e.g., `Exercise Angina` × `Diseased Vessels` ×` ST Depression`).

![3D_plot](png\3D_plot.png)

- **Baseline Model:** Training an out-of-the-box CatBoost strictly to extract feature importance and plot 2D Partial Dependence Plots (identifying synergistic risk factors).

### 2. Pipelines, Tuning, and Error Analysis

- **Pipeline Experiments:** Testing combinations of `Yeo-Johnson` + `MinMaxScaler` / `StandardScaler` / `RobustScaler`.
- **GridSearchCV:** Tuning and comparing linear models, trees, and gradient boosting algorithms.
- **Model Selection:** `LogisticRegression (C=0.1, solver='liblinear')` showed the most stable results. The model achieved an **F1-score of 0.842 and a Recall of 0.800** on the test set with no signs of overfitting.

![final_model](png\final_model.png)

- **Interpretability:** Comparing native Logistic Regression coefficients and SHAP values to explain the global logic of the model.
- **Detailed Breakdown of False Negative Errors:** Using `shap.plots.waterfall` for individual analysis of each missed patient.

## Key Insights

- **Top Predictors:** Thallium test results, chest pain type, number of diseased vessels, and ST-segment depression on the ECG carry the highest predictive power.


![importance](png\importance.png)

- **Age and Cholesterol:** By themselves, they are weak linear predictors for this dataset, but they act as strong risk amplifiers (triggers) in conjunction with other pathologies (e.g., age-related wear + poor thallium test).
- **Simpler is Better:** On a dataset of 300 rows, XGBoost and Random Forest achieved an Accuracy of $1.000$ on the training set but dropped significantly on the test set. A heavily regularized Logistic Regression showed excellent generalization ability.
- **Why the Model Fails:** SHAP analysis revealed that false-negative errors occurred in atypical cases. The missed patients had clear vessels, a normal thallium test, and no exercise-induced angina. The algorithm did not make a mathematical error, as it logically relied on classic markers. However, due to the small data volume, the algorithm simply lacks the statistical base to learn the patterns of atypical disease progression, forcing the model to rely only on the most obvious symptoms.

![waterfalls](png\waterfalls.png)
