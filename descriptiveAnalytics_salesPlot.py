import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data and robustly parse dates
df = pd.read_csv("train_cleansed.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True, errors='coerce')
df['month'] = df['Order Date'].dt.to_period('M')

def plot_trend_group(group, size, color='blue'):
    sales_by_group = df.groupby(group)['Sales'].sum().sort_values(ascending=False).head(size)
    nrows = int(np.ceil(size / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 4 * nrows), sharex=True, sharey=True)
    fig.suptitle(f'Sales trend by {group}', fontsize=18)
    axes = axes.flatten()
    # For each top group, plot its monthly sales trend
    for ax, group_name in zip(axes, sales_by_group.index):
        sales_group = df[df[group] == group_name].groupby('month')['Sales'].sum().sort_index()
        sales_group.index = sales_group.index.to_timestamp()
        ax.plot(sales_group.index, sales_group.values, color=color, marker='o')
        ax.set_title(group_name, fontsize=10)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylabel('Sales')
    # Hide unused axes
    for ax in axes[len(sales_by_group):]:
        ax.axis('off')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Example usages:
plot_trend_group('State', 12)           # Top 12 states
plot_trend_group('Region', 3, 'red')    # By region
plot_trend_group('Segment', 3, 'green') # By segment
plot_trend_group('Category', 3, 'orange') # By category
plot_trend_group('Sub-Category', 12, 'black') # By sub-category
plot_trend_group('Product Name', 12, 'red')   # Top 12 products