import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    
    xlsx_path = '/home/acy/data/lung_2023/data/seg_results.xls'
    save_path = '/home/acy/data/lung_2023/results'  
    
    results = pd.read_excel(xlsx_path)

    # Adjust subplot sizes: make subplot 2 narrower
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100, gridspec_kw={'width_ratios': [3, 1]})
    
    # Define custom colors and transparency for boxplot and scatter
    custom_palette = ['red', 'blue']  # Define your own colors
    box_alpha = 0.7  # Transparency for boxplot
    scatter_alpha = 0.5  # Transparency for scatter

    # Boxplot and scatter for subplot 1
    sns.boxplot(x='Category', y='Value', hue='Set', data=results[results['Category'] != 'MED'], ax=axes[0], palette=custom_palette, width=0.5)
    sns.stripplot(x='Category', y='Value', hue='Set', data=results[results['Category'] != 'MED'], ax=axes[0], palette=custom_palette, dodge=True, jitter=True, alpha=scatter_alpha, zorder=1)
    axes[0].set_title('Box Plot of Dice and Recall')
    axes[0].set_ylabel('Values')
    axes[0].legend()

    # Boxplot and scatter for subplot 2
    sns.boxplot(x='Category', y='Value', hue='Set', data=results[results['Category'] == 'MED'], ax=axes[1], palette=custom_palette, width=0.5)
    sns.stripplot(x='Category', y='Value', hue='Set', data=results[results['Category'] == 'MED'], ax=axes[1], palette=custom_palette, dodge=0.3, jitter=True, alpha=scatter_alpha, zorder=1)
    axes[1].set_title('Box Plot of MED')
    axes[1].legend([],[], frameon=False)  # Hide legend for the second plot

    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(save_path, 'box.png'), dpi=300, bbox_inches='tight')
