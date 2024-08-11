import pandas as pd
from pycaret.regression import setup, compare_models, save_model, load_model, predict_model
from data.load_data import load_data
from preprocessing.pipeline import create_pipeline
from visualization.plot_tools import histogram,boxplots,heatmap,pairplot,sweetviz_report

def main():
    # 1. load the datasets
    train_df = load_data('data/train.csv').drop(columns=['ID'])
    test_df = load_data('data/test.csv').drop(columns=['ID'])

    # 2. create the processing pipeline
    pipeline = create_pipeline(train_df)
    
    # 3. separate X and y
    train_X = train_df.drop(columns=['y'])
    train_y = train_df['y']
    
    new_train_X = pipeline.fit_transform(train_X)
    new_train_X = pd.DataFrame(new_train_X, columns=train_X.columns)
    
    new_train_df = pd.concat([new_train_X, train_y.reset_index(drop=True)], axis=1)
    
    # 4. AutoML setup and model training
    regressor = setup(data=new_train_df, target='y', session_id=42)
    best_model = compare_models()
    
    save_model(best_model, 'best_model')

    # 5. preprocess the test data and make prediction
    new_test_X = pipeline.transform(test_df)
    new_test_df = pd.DataFrame(new_test_X, columns=test_df.columns)
    preds = predict_model(best_model, data=new_test_df)
    
    # 6. prepare the submission file
    submission = pd.read_csv('data/sample_submission.csv')
    submission['y'] = preds['Label']
    submission.to_csv('../result_submission.csv', index = False)
    
    # 7. generate visualization
    histogram(train_df)
    boxplots(train_df)
    heatmap(train_df)
    pairplot(train_df)
    sweetviz_report(train_df, 'sweetviz_report.html')

if __name__ == "__main__":
    main()