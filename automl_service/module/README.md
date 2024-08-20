# How to run this module?

### Prerequisites

```
conda create -n ss_env python=3.8
conda activate ss_env
conda install numpy pandas scikit-learn seaborn matplotlib 
pip install --upgrade pip setuptools wheel
pip install pycaret sweetviz joblib==1.2.0

# pip install -r module/requirements.txt
```

### Running the Script

```
python module/main.py
```



# Directory Structure
```
project_root/
│
├── data/
│   ├── train.csv                 # Training dataset
│   ├── test.csv                  # Test dataset
│   └── sample_submission.csv     # Sample submission file
│
├── module/                       # Contains all project-specific Python modules
│   ├── data/
│   │   └── load_data.py          # Script for loading datasets
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── bimodal_transformer.py
│   │   ├── detect_skewness.py
│   │   ├── detect_bimodal.py
│   │   └── pipeline.py           # Contains the preprocessing pipeline
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plot_tools.py         # Script with functions for data visualization
|   ├── optimization/             # **Let's build the optimization algorithms**
│   │   ├── __init__.py
│   │   └── README.md
|   ├── results/
│   │   └── result_submission.csv     
│   ├── README.md                 # Documentation for the module
│   ├── main.py                   # Main script to run the pipeline and model
│   └── requirements.txt          # List of required Python packages
```
