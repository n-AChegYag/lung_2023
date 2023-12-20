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
    axes[0].set_title('Box Plot of Dice and Recall')
    axes[0].set_ylabel('Values')
    axes[0].set_xlabel('')
    axes[0].legend([],[], frameon=False) 

    # Second subplot for MED
    sns.boxplot(x='Category', y='Value', hue='Set', data=results[results['Category'] == 'MED'], ax=axes[1], palette=box_colors, width=box_width)
    sns.stripplot(x='Category', y='Value', hue='Set', data=results[results['Category'] == 'MED'], ax=axes[1], palette=strip_colors, dodge=True, jitter=True, alpha=alpha_value, zorder=1)
    axes[1].set_title('Box Plot of MED')
    axes[1].set_ylabel('')
    axes[1].set_xlabel('')
    axes[1].legend(loc='upper right')

    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(save_path, 'box.png'), dpi=300, bbox_inches='tight')
