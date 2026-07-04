import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import to_hex
from unittest.mock import patch


def plot_features_vs_target(df, feature_cols, target_col='Chest Pain', n_cols=3):
    """
    Plots a grid of bar charts to analyze relationships between categorical variables.

    Args:
        df (pandas.DataFrame): Source dataframe containing the data.
        feature_cols (list[str]): List of categorical feature names to visualize.
        target_col (str, optional): Name of the target categorical variable for the X-axis. 
                                    Defaults to 'Chest Pain'.
        n_cols (int, optional): Number of columns in the subplot grid. Defaults to 3.
    """
    
    n_features = len(feature_cols)
    n_rows = math.ceil(n_features / n_cols)

    # Setup 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    # Plotting
    for ax, var in zip(axes, feature_cols):
        sns.countplot(data=df, x=target_col, hue=var, ax=ax, palette="deep", edgecolor='black')
        
        # Title and tick formatting
        ax.set_title(f"{var.upper()} × {target_col}", fontweight='bold', fontsize=13)
        ax.tick_params(axis='x', labelsize=15, labelrotation=10)
        
        # Grid layout and legend styling
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title=var, fontsize=14, title_fontsize=8, edgecolor='black')

    # Cleanup unused subplots
    for ax in axes[n_features:]:
        ax.remove()

    plt.show()


def transform_and_plot_yj(df, cols):
    """
    Applies Yeo-Johnson transformation to specified columns and plots Before/After.
    """
    df_out = df.copy()
    
    for col in cols:
        orig = df[col]
            
        # Transform
        trans, lmbda = stats.yeojohnson(orig)
        trans_series = pd.Series(trans, index=orig.index)
        df_out.loc[orig.index, col] = trans_series
        
        # Plot setup
        fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        fig.suptitle(f"{col} | $\lambda$ = {lmbda:.4f}", fontweight='bold', fontsize=14)
        
        # Plotting helper function
        def _plot_row(data, row_idx, title, color):
            sns.histplot(data, kde=True, ax=axs[row_idx, 0], color=color)
            axs[row_idx, 0].set_title(f"{title} Hist (Skew: {data.skew():.3f})")
            
            stats.probplot(data, dist="norm", plot=axs[row_idx, 1])
            axs[row_idx, 1].set_title(f"{title} Q-Q")
            
            sns.boxplot(y=data, ax=axs[row_idx, 2], color=color, width=0.4)
            axs[row_idx, 2].set_title(f"{title} Boxplot")
            
            for ax in axs[row_idx]: ax.grid(True, linestyle='--', alpha=0.5)

        # Render rows
        _plot_row(orig, 0, "[ORIGINAL]", '#4C72B0')
        _plot_row(trans_series, 1, "[YEO-JOHNSON]", '#DD8452')
        
        plt.show()
        
    return df_out


def analyze_categorical_features(df, cols, target='target', n_cols=2):
    """
    Plots stacked bar charts for categorical features relative to a target variable.

    Creates a grid of stacked bar charts. For each category, it displays its 
    overall sample share (bar height), absolute sample count, and the internal 
    percentage distribution of the target variable (segments inside the bar).

    Args:
        df (pandas.DataFrame): Source dataframe containing the data.
        cols (list[str]): List of categorical feature names to analyze.
        target (str, optional): Name of the target variable. Defaults to 'target'.
        n_cols (int, optional): Number of charts per row in the grid. Defaults to 2.
    """
    
    # Grid setup
    n_rows = math.ceil(len(cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 5), 
                             squeeze=False, constrained_layout=True)
    
    # Plotting
    for ax, var in zip(axes.flat, cols):
        # Compute sorted statistics
        ct = pd.crosstab(df[var], df[target])
        ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
        
        overall_pct = ct / ct.values.sum() * 100
        within_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        
        overall_pct.plot.bar(stacked=True, ax=ax, rot=0, edgecolor='black', 
                             color=sns.color_palette("deep").as_hex(), linewidth=1.2)
        
        ax.set(xlabel="", ylabel="Percent of total (%)")
        ax.set_title(var.upper(), fontsize=14, fontweight='bold')
        
        ax.tick_params(axis='x', labelsize=13) 
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)  # Padding for top labels
        
        # Internal labels (% within category)
        for j, cont in enumerate(ax.containers):
            labels = [f'{v:.0f}%' if v >= 1 else '' for v in within_pct.iloc[:, j]]
            ax.bar_label(cont, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=12)

        # Top labels (Overall % and absolute counts)
        top_y = overall_pct.sum(axis=1)
        counts = ct.sum(axis=1)
        for k in range(len(ct)):
            ax.text(k, top_y.iloc[k] + 1, f'{top_y.iloc[k]:.1f}%\n({counts.iloc[k]})',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
            
        # Legend management
        if ax != axes.flat[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title=target, edgecolor='black')

    # Cleanup unused subplots
    for ax in axes.flat[len(cols):]:
        ax.remove()

    plt.show()


def feature_importance_dashboard(model, X_train, shap_values, shap_interactions, perm_result):
    """
    Visualizes a comprehensive dashboard of model feature importance and contributions.
    
    Displays a 2x2 grid containing:
    1. Built-in Feature Importance from the model.
    2. Permutation Importance (mean drop in score with standard deviation).
    3. SHAP Summary Plot (mean absolute SHAP values).
    4. Top 14 Feature Interactions based on SHAP interaction values.

    Args:
        model: Trained model object (e.g., CatBoostClassifier) supporting .feature_importances_.
        X_train (pd.DataFrame): Training feature matrix.
        shap_values: Calculated SHAP values object.
        shap_interactions (np.ndarray): Calculated SHAP interaction values matrix.
        perm_result: Object returned by sklearn.inspection.permutation_importance.
    """
    # Calculate parameters for Permutation Importance
    perm_mean = perm_result.importances_mean
    perm_std = perm_result.importances_std
    perm_indices = perm_mean.argsort()

    # Style configuration
    text_color = '#333333'
    title_kws = {'fontweight': 'bold', 'fontsize': 14}
    label_kws = {'fontsize': 9, 'color': text_color}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. CatBoost Feature Importances
    df_imp = pd.DataFrame({"Feature": X_train.columns, "Importance": model.feature_importances_}).sort_values("Importance")
    axes[0, 0].barh(df_imp["Feature"], df_imp["Importance"], color='#4C72B0', edgecolor='black')
    axes[0, 0].set_xlabel("Importance", **label_kws)
    axes[0, 0].set_title("CatBoost Feature Importances", **title_kws)

    # 2. Permutation Importances
    axes[0, 1].barh(range(len(perm_indices)), perm_mean[perm_indices], xerr=perm_std[perm_indices], color='#55A868', edgecolor='black')
    axes[0, 1].set_yticks(range(len(perm_indices)))
    axes[0, 1].set_yticklabels(X_train.columns[perm_indices])
    axes[0, 1].set_xlabel("Permutation Importance (mean drop in score)", **label_kws)
    axes[0, 1].set_title("Permutation Importances", **title_kws)

    # 3. SHAP Summary Plot
    plt.sca(axes[1, 0])
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, plot_size=None, color='#DD8452')
    for patch in axes[1, 0].patches:
        patch.set_edgecolor('black')
    axes[1, 0].set_xlabel("mean(|SHAP value|) (average impact on model output)", **label_kws)
    axes[1, 0].set_title("SHAP Feature Importances", **title_kws)

    # 4. Top Feature Interactions
    S = np.abs(shap_interactions).mean(0)
    iu, ju = np.triu_indices(S.shape[0], 1)
    v = S[iu, ju]
    idx = np.argsort(-v)[:14]

    feature_pairs = [f"{X_train.columns[i]} & {X_train.columns[j]}" for i, j in zip(iu[idx], ju[idx])]
    axes[1, 1].barh(feature_pairs[::-1], v[idx][::-1], color='#C44E52', edgecolor='black')
    axes[1, 1].set_xlabel("Mean(|SHAP interaction value|)", **label_kws)
    axes[1, 1].set_title("Top Feature Interactions", **title_kws)

    # Axes and grid styling
    for ax in axes.flatten():
        ax.set_axisbelow(True) 
        ax.grid(True, axis='x', linestyle='--', alpha=0.6) 
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0, labelsize=12, labelcolor=text_color)
        ax.tick_params(axis='x', labelsize=11, labelcolor=text_color)

    plt.tight_layout()
    plt.show()


def display_model_dashboard(model_name, model, X_train, y_train, X_test, y_test, threshold=0.5):
    """
    Evaluates a classification model and visualizes its performance metrics.

    Calculates key metrics (F1 Macro, PR AUC for class 0, and Balanced Accuracy) 
    for both training and testing datasets. Displays a 1x3 dashboard featuring 
    confusion matrices and a metric comparison bar chart, followed by a
    classification report for the test data.

    Args:
        model_name (str): The display name of the model for the plot title.
        model (estimator): The trained machine learning model or pipeline.
        X_train (array-like): Training features.
        y_train (array-like): Training target labels.
        X_test (array-like): Testing features.
        y_test (array-like): Testing target labels.
        threshold (float, optional): The probability threshold for positive class prediction. Defaults to 0.5.
    """
    # Get probabilities (if the model supports it)
    if hasattr(model, "predict_proba"):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Apply custom threshold to probabilities
        y_train_pred = (y_train_proba >= threshold).astype(int)
        y_test_pred = (y_test_proba >= threshold).astype(int)
        
    else:
        # Warning for models without predict_proba 
        if threshold != 0.5:
            print(f" Warning: model {model_name} does not support predict_proba. Threshold {threshold} ignored.")
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        if hasattr(model, "decision_function"):
            y_train_proba = model.decision_function(X_train) 
            y_test_proba = model.decision_function(X_test)
        else:
            y_train_proba = y_train_pred
            y_test_proba = y_test_pred

    # Calculate metrics
    metrics = {
        'F1 score (train)': f1_score(y_train, y_train_pred),
        'F1 score (test)': f1_score(y_test, y_test_pred),
        'ROC AUC (train)': roc_auc_score(y_train, y_train_proba),
        'ROC AUC (test)': roc_auc_score(y_test, y_test_proba),
        'Accuracy (train)': accuracy_score(y_train, y_train_pred),
        'Accuracy (test)': accuracy_score(y_test, y_test_pred)
    }

    diffs = {
        'F1': metrics['F1 score (train)'] - metrics['F1 score (test)'],
        'ROC AUC': metrics['ROC AUC (train)'] - metrics['ROC AUC (test)'],
        'Accuracy': metrics['Accuracy (train)'] - metrics['Accuracy (test)']
    }

    # VISUALIZATION (1x3 grid)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Current threshold in the title 
    fig.suptitle(f"{model_name} (Threshold: {threshold})", fontsize=16, fontweight='bold')

    # Confusion Matrix: Train 
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=ax[0], cbar=False)
    ax[0].set_title("Confusion Matrix (Train)", fontsize=12)
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    # Confusion Matrix: Test
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", ax=ax[1], cbar=False)
    ax[1].set_title("Confusion Matrix (Test)", fontsize=12)
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")

    # Metrics Barplot 
    df_metrics = pd.DataFrame({'Metric': list(metrics.keys()), 'Score': list(metrics.values())})
    
    sns.barplot(
        data=df_metrics, 
        x='Score', 
        y='Metric', 
        hue='Metric',
        palette='Paired', 
        edgecolor='black',
        ax=ax[2]
    )
    
    # Add labels to the bars
    for container in ax[2].containers:
        ax[2].bar_label(container, fmt='%.3f', padding=5, fontsize=11, fontweight='bold')
        
    # Create a title showing the difference (Train - Test) 
    diff_str = " | ".join([f"{k}: {v:.3f}" for k, v in diffs.items()])
    ax[2].set_title(f"Key Metrics\nDifference (Train - Test): {diff_str}", fontsize=12, color='darkred')
    
    ax[2].set_xlim(0, 1.15) # Extra space for labels
    ax[2].set_ylabel("")
    ax[2].set_xlabel("Score")

    plt.tight_layout()
    plt.show()

    # OUTPUT TEST CLASSIFICATION REPORT
    print(f"Classification Report (TEST data, Threshold = {threshold}):")
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).T.round(3)
    display(df_report)


def feature_importance_logit(X_train, shap_values, logreg_model):
    """
    Visualizes a dashboard of model feature importance and contributions.
    
    Displays a 1x2 grid containing:
    SHAP Summary Plot (mean absolute SHAP values).
    Logistic Regression Coefficients (sorted by absolute magnitude).

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        shap_values: Calculated SHAP values object.
        logreg_model: Trained Logistic Regression model.
    """
    # Style configuration
    text_color = '#333333'
    color = sns.color_palette("deep").as_hex()
    title_kws = {'fontweight': 'bold', 'fontsize': 14}
    label_kws = {'fontsize': 10, 'color': text_color}

    # Set up a 1x2 grid
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # SHAP Summary Plot
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, plot_size=None, color=color[0])
    
    # Add borders to SHAP bars
    for patch in axes[0].patches:
        patch.set_edgecolor('black')
        
    axes[0].set_xlabel("mean(|SHAP value|) (average impact on model output)", **label_kws)
    axes[0].set_title("SHAP Feature Importances", **title_kws)

    # Logistic Regression Coefficients
    coefs = logreg_model.coef_[0]
    df_coefs = pd.DataFrame({"Feature": X_train.columns, "Coefficient": coefs})
    
    # Sort by absolute value 
    df_coefs['Abs_Coef'] = df_coefs['Coefficient'].abs()
    df_coefs = df_coefs.sort_values(by='Abs_Coef', ascending=True)

    # Assign colors
    bar_colors = [color[1] if c > 0 else color[0] for c in df_coefs['Coefficient']]
    
    axes[1].barh(df_coefs['Feature'], df_coefs['Coefficient'], color=bar_colors, edgecolor='black')
    axes[1].set_xlabel("Coefficient Value", **label_kws)
    axes[1].set_title("Logistic Regression Coefficients", **title_kws)

    # Axes and grid styling for all subplots
    for ax in axes:
        ax.set_axisbelow(True) 
        ax.grid(True, axis='x', linestyle='--', alpha=0.6) 
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0, labelsize=12, labelcolor=text_color)
        ax.tick_params(axis='x', labelsize=11, labelcolor=text_color)
        
        # Add a vertical line at x=0 for the Logistic Regression plot
        if ax == axes[1]:
            ax.axvline(x=0, color='black', linewidth=1.2)

    plt.tight_layout()
    plt.show()


def shap_waterfalls_dashboard(shap_values, max_display=13):
    """
    Visualizes a grid of SHAP waterfall plots for multiple patients.
    """
    n_rows = math.ceil(len(shap_values) / 2)
    
    # Create a grid of the required size
    fig, axes = plt.subplots(n_rows, 2, figsize=(18, 7 * n_rows))

    axes = axes.flatten()
    color = sns.color_palette("deep").as_hex()
    
    for i in range(len(axes)):
        ax = axes[i]
        
        if i < len(shap_values):
            # Force plotting in this specific axis
            plt.sca(ax)
            
            with patch("matplotlib.figure.Figure.set_size_inches"):
                shap.plots.waterfall(shap_values[i], max_display=max_display, show=False)
            
            # Find and recolor all shapes (bars and arrows)
            for pch in ax.patches:
                if to_hex(pch.get_facecolor()) == to_hex(shap.plots.colors.red_rgb):
                    pch.set_facecolor(color[1])
                    pch.set_edgecolor('black') 
                elif to_hex(pch.get_facecolor()) == to_hex(shap.plots.colors.blue_rgb):
                    pch.set_facecolor(color[0])
                    pch.set_edgecolor('black') 
                    
            # Find and recolor the text itself (numbers inside the plot)
            for txt in ax.texts:
                if to_hex(txt.get_color()) == to_hex(shap.plots.colors.red_rgb):
                    txt.set_color(color[1])
                elif to_hex(txt.get_color()) == to_hex(shap.plots.colors.blue_rgb):
                    txt.set_color(color[0])
                    
            ax.set_title(f"Patient #{i+1} Breakdown", fontweight='bold', fontsize=14, pad=15)
            
        else:
            # Hide empty frames if the number of patients is odd
            ax.set_visible(False)
            
    plt.tight_layout()
    plt.show()


# Mappings 
value_maps = {
    'Sex': {1: 'Male', 0: 'Female'},
    'Chest Pain': {1: 'Typical', 2: 'Atypical', 3: 'Non-Anginal', 4: 'Asymptomatic'},
    'Blood Sugar >120': {1: 'Yes', 0: 'No'},
    'Resting ECG': {0: 'Normal', 1: 'ST-T Abn.', 2: 'LVH'},
    'Exercise Angina': {1: 'Yes', 0: 'No'},
    'ST Slope': {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'},
    'Thallium Test': {3: 'Normal', 6: 'Fixed', 7: 'Reversible'}
}