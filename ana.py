"""
Author: Alex (Tai-Jung) Chen

Offering some plots for analysis purpose.
"""
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


def main():
    # Read the CSV file
    # file_name = "COVID_case1_recall.csv"
    # file_name = "COVID_case2_specificity.csv"
    # file_name = "HMEQ_case1_specificity.csv"
    file_name = "NIJ_case1_npv.csv"
    df = pd.read_csv(file_name)

    # Extract unique metrics (excluding 'Method' and 'Model')
    metrics = [col for col in df.columns if col in ['acc_mean', 'kappa_mean', 'bacc_mean', 'precision_mean',
                                                    'recall_mean', 'specificity_mean', 'f1_mean', 'npv_mean',
                                                    'auc_mean', 'apr_mean']]
    models = df['model_simp'].unique()
    methods = df['tune'].unique()

    # Subplots setup
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 10), sharey=True)
    axes = axes.flatten()

    # Loop through each metric and plot
    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(models))  # X-axis positions
        width = 0.35  # Width of each bar

        # Plot bars for each method
        for j, method in enumerate(methods):
            values = df[df['tune'] == method][metric]
            ax.bar(x + (j - 0.5) * width, values, width, label=method, alpha=0.7)

        # Titles and labels
        ax.set_title(metric, fontsize=14)
        ax.set_xlabel('Model Type', fontsize=12)
        if i == 0:  # Add y-axis label only for the first subplot
            ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)

    # Add legend to the first subplot
    axes[0].legend(title='Method', loc='lower right', fontsize=10)

    # Adjust layout
    plt.tight_layout()
    # plt.show()
    plt.savefig(file_name[:-4] + '.png')


if __name__ == '__main__':
    main()
