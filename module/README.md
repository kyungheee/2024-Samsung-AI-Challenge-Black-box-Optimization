# How to run this module?

### Prerequisites

```bash
python3.8 -m venv ss_env
source ss_env/bin/activate
pip install -r module/requirements.txt
```

### Running the Script

```bash
python main.py
```

### Directory Structure
project_root/
│
├── data/
│ ├── train.csv # Training dataset
│ ├── test.csv # Test dataset
│ └── sample_submission.csv # Sample submission file
│
├── module/ # Contains all project-specific Python modules
│ ├── data/
│ │ └── load_data.py # Script for loading datasets
│ ├── preprocessing/
│ │ ├── init.py
│ │ ├── bimodal_transformer.py
│ │ ├── detect_skewness.py
│ │ ├── detect_bimodal.py
│ │ └── pipeline.py # Contains the preprocessing pipeline
│ ├── visualization/
│ │ ├── init.py
│ │ └── plot_tools.py # Script with functions for data visualization
│ ├── README.md # Documentation for the module
│ ├── main.py # Main script to run the pipeline and model
│ └── requirements.txt # List of required Python packages