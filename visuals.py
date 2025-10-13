import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

def plot_capital_features_distribution(data: pd.DataFrame, transformed: bool = False) -> None:
    """
    Plots the distributions for 'capital-gain' and 'capital-loss' features.

    Args:
        data (pd.DataFrame): The dataset containing the features.
        transformed (bool, optional): Flag indicating if the data is 
                                      log-transformed. Defaults to False.
    """
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    features = ['capital-gain', 'capital-loss']
    plot_color = '#00A0A0'

    # Iterate over features and axes together
    for ax, feature in zip(axes, features):
        ax.hist(data[feature], bins=25, color=plot_color)
        ax.set_title(f"'{feature}' Feature Distribution", fontsize=14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        
        # Set custom y-axis to handle high skewness
        ax.set_ylim((0, 2000))
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

    # Set the main title for the figure
    title_prefix = "Log-Transformed" if transformed else "Skewed"
    fig.suptitle(f"{title_prefix} Distributions of Continuous Census Data Features", 
                 fontsize=16, y=1.03)

    fig.tight_layout()
    plt.show()
def plot_evaluation(results, accuracy, f1):
    """
    Plots evaluation metrics for different learners.
    Args:
        results (dict): Results from learners.
        accuracy (float): Naive predictor accuracy.
        f1 (float): Naive predictor F1 score.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    bar_width = 0.3
    colors = ['#A00000', '#00A0A0', '#00A000']
    metrics = ['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']
    for learner_idx, learner in enumerate(results):
        for metric_idx, metric in enumerate(metrics):
            for size_idx in range(3):
                axes[metric_idx // 3, metric_idx % 3].bar(
                    size_idx + learner_idx * bar_width,
                    results[learner][size_idx][metric],
                    width=bar_width,
                    color=colors[learner_idx]
                )
                axes[metric_idx // 3, metric_idx % 3].set_xticks([0.45, 1.45, 2.45])
                axes[metric_idx // 3, metric_idx % 3].set_xticklabels(["1%", "10%", "100%"])
                axes[metric_idx // 3, metric_idx % 3].set_xlabel("Training Set Size")
                axes[metric_idx // 3, metric_idx % 3].set_xlim((-0.1, 3.0))
    ylabels = ["Time (in seconds)", "Accuracy Score", "F-score"]
    for row in range(2):
        for col in range(3):
            axes[row, col].set_ylabel(ylabels[col])
    titles = [
        "Model Training", "Accuracy Score on Training Subset", "F-score on Training Subset",
        "Model Predicting", "Accuracy Score on Testing Set", "F-score on Testing Set"
    ]
    for idx, ax in enumerate(axes.flat):
        ax.set_title(titles[idx])
    # Horizontal lines for naive predictors
    for ax in [axes[0, 1], axes[1, 1]]:
        ax.axhline(y=accuracy, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    for ax in [axes[0, 2], axes[1, 2]]:
        ax.axhline(y=f1, xmin=-0.1, xmax=3.0, linewidth=1, color='k', linestyle='dashed')
    # Set y-limits for score panels
    for ax in [axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]:
        ax.set_ylim((0, 1))
    # Legend
    legend_patches = [mpatches.Patch(color=colors[i], label=learner) for i, learner in enumerate(results)]
    plt.legend(handles=legend_patches, bbox_to_anchor=(-.80, 2.40), loc='upper center',
               borderaxespad=0., ncol=3, fontsize='x-large')
    plt.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize=16, y=0.99)
    plt.show()

def plot_feature_importances(importances, X_train):
    """
    Plots the five most important features and their cumulative weights.
    Args:
        importances (np.ndarray): Feature importances.
        X_train (pd.DataFrame): Training data (for feature names).
    """
    indices = np.argsort(importances)[::-1]
    top_features = X_train.columns.values[indices[:5]]
    top_values = importances[indices][:5]
    plt.figure(figsize=(9, 5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize=16)
    plt.bar(np.arange(5), top_values, width=0.6, align="center", color='#00A000', label="Feature Weight")
    plt.bar(np.arange(5) - 0.3, np.cumsum(top_values), width=0.2, align="center", color='#00A0A0', label="Cumulative Feature Weight")
    plt.xticks(np.arange(5), top_features)
    plt.xlim((-0.5, 4.5))
    plt.ylabel("Weight", fontsize=12)
    plt.xlabel("Feature", fontsize=12)
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.show()
