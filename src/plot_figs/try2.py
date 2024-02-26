import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


if __name__ == '__main__':

    xlsx_path = '/home/acy/data/lung_2023/data/seg_results.xls'
    save_path = '/home/acy/data/lung_2023/results'  

    results = pd.read_excel(xlsx_path)

    # Define colors and alpha for boxplot and stripplot
    box_colors = ['skyblue', 'lightgreen']
    strip_colors = ['blue', 'green']
    alpha_value = 0.5
    box_width = 0.75

    # Create subplots with different widths
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), dpi=100, gridspec_kw={'width_ratios': [2, 1]})

    # First subplot for Dice and Recall
    sns.boxplot(x='Category', y='Value', hue='Set', data=results[results['Category'] != 'MED'], ax=axes[0], palette=box_colors, width=box_width)
    sns.stripplot(x='Category', y='Value', hue='Set', data=results[results['Category'] != 'MED'], ax=axes[0], palette=strip_colors, dodge=True, jitter=True, alpha=alpha_value, zorder=1)
    axes[0].set_ylabel('Values')
    axes[0].set_xlabel('')
    axes[0].legend([],[], frameon=False) 

    # Second subplot for MED
    sns.boxplot(x='Category', y='Value', hue='Set', data=results[results['Category'] == 'MED'], ax=axes[1], palette=box_colors, width=box_width)
    sns.stripplot(x='Category', y='Value', hue='Set', data=results[results['Category'] == 'MED'], ax=axes[1], palette=strip_colors, dodge=True, jitter=True, alpha=alpha_value, zorder=1)
    axes[1].set_ylabel('')
    axes[1].set_xlabel('')
    axes[1].legend(loc='upper right')
    
    # Calculate the mean for each 'Category' and 'Set'
    means = results.groupby(['Category', 'Set'])['Value'].mean().reset_index()

    # Iterate over each 'Set' and 'Category' to draw a horizontal line at the mean 'Value'
    for index, row in means.iterrows():
        category = row['Category']
        set_name = row['Set']
        mean_value = row['Value']
        # Determine the axis to use (0 for Dice and Recall, 1 for MED)
        ax_index = 0 if category != 'MED' else 1
        ax = axes[ax_index]
        # Determine the x position for the line
        x_position = ['Internal Validation Set', 'External Validation Set'].index(set_name)
        # Adjust x_position for the subplot with two categories
        if ax_index == 0:
            x_position *= 2  # Multiply by 2 for the subplot with two categories
            if category == 'Recall':
                x_position += 1  # Offset by 1 for Recall
        # Draw the line and add text annotation
        ax.axhline(mean_value, xmin=x_position + 0.1, xmax=x_position + 0.9, color='red', linestyle='--', linewidth=2)
        ax.text(x_position + 0.5, mean_value, f'{mean_value:.2f}', horizontalalignment='center', color='red')

    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(save_path, 'box_240110.png'), dpi=300, bbox_inches='tight')