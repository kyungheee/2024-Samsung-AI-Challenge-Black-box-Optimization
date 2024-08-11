import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv

def histogram(df, figsize=(20,20), save_path=None):
    fig, axes = plt.subplots(4, 3, figsize=figsize)
    axes = axes.ravel()
    
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)


def boxplots(df, figsize=(20,20), save_path=None):
    fig, axes = plt.subplots(4, 3, figsize=figsize)
    axes = axes.ravel()

    for i, col in enumerate(df.columns):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'Box-plot of {col}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        
def heatmap(df, save_path=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Features Correlation Heatmap')
    if save_path:
        plt.savefig(save_path)
        
def pairplot(df, save_path=None):
    sns.pairplot(df)
    plt.title('Pairplot of Features')
    if save_path:
        plt.savefig(save_path)
        
def sweetviz_report(df, report_name='sweetviz_report.html', save_path=None):
    report = sv.analyze(df)
    if save_path:
        report_name = save_path
    report.show_html(report_name)
