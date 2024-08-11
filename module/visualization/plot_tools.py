import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv

def histogram(df,figsize=(20,20)):
    fig, axes = plt.subplots(4,3,figsize=figsize)
    axes = axes.ravel() # Converts a two-dimensional array to a one-dimensional array
    
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], ax=axes[i], kde=True)
        axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()

def boxplots(df,figsize=(20,20)):
    fig, axes = plt.subplots(4, 3, figsize=figsize)
    axes = axes.ravel()

    for i, col in enumerate(df.columns):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'Box-plot of {col}')

    plt.tight_layout()
    plt.show()

def heatmap(df):
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, cmap=True)
    plt.title('Features Correlation Heatmap')
    plt.show()
    
def pairplot(df):
    sns.pairplot(df)
    plt.title('Pairplot of Features')
    plt.show()
    
def sweetviz_report(df, report_name='sweetviz_report.html'):
    report = sv.analyze(df)
    report.show_html(report_name)