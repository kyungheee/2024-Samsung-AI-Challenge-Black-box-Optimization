import pandas as pd
from pycaret.regression import setup, compare_models, save_model, load_model, predict_model
from data.load_data import load_data
from preprocessing.pipeline import create_pipeline
from auto_ml.model import run_automl, get_top_features, train_with_top
from visualization.plot_tools import histogram,boxplots,heatmap,pairplot,sweetviz_report

def main():
    # 1. load the datasets
    train_df = load_data('../data/train.csv').drop(columns=['ID'])
    test_df = load_data('../data/test.csv').drop(columns=['ID'])

    # 2. create the processing pipeline
    pipeline = create_pipeline(train_df)
    
    # 3. separate X and y
    train_X = train_df.drop(columns=['y'])
    train_y = train_df['y']
    
    new_train_X = pipeline.fit_transform(train_X)
    new_train_X = pd.DataFrame(new_train_X)
    
    new_train_df = pd.concat([new_train_X, train_y.reset_index(drop=True)], axis=1)
    
    # 4. AutoML setup and model training
    top5_models = run_automl(new_train_X, train_y)
    
    for i, model in enumerate(top5_models):
        top5_features = get_top_features(model, new_train_X)
        print(model, top5_features)
        
        # 5. train with top5 features
        final_model = train_with_top(new_train_X)
        save_model(final_model, f'final_model_{i+1} : {model}')
        
        # 6. preprocess the test data and make prediction
        new_test_X = pipeline.transform(test_df)
        new_test_df = pd.DataFrame(new_test_X, columns=test_df.columns)
        preds = predict_model(final_model, data=new_test_df)
        print(preds)

        # 7. prepare the submission file
        submission = pd.read_csv('data/sample_submission.csv')
        submission['y'] = preds.iloc[:, -1]
        submission.to_csv(f'module/results/result_submission_{i+1}.csv', index = False)
    
    # # 7. generate visualization
    # histogram(train_df, save_path='module/results/histogram.png')
    # boxplots(train_df, save_path='module/results/boxplots.png')
    # heatmap(train_df, save_path='module/results/heatmap.png')
    # pairplot(train_df, save_path='module/results/pairplot.png')
    # sweetviz_report(train_df, 'module/results/sweetviz_report.html')

if __name__ == "__main__":
    main()