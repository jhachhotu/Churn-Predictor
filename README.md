# Customer Churn Prediction System

## Overview

This repository contains a machine learning application designed to predict customer churn probability using deep learning techniques. The system employs an Artificial Neural Network (ANN) trained on customer behavioral data to provide real-time risk assessments.

## Technical Architecture

### Core Components
- **Frontend**: Streamlit-based web interface for user interaction
- **Backend**: TensorFlow/Keras neural network model
- **Data Processing**: Scikit-learn preprocessing pipeline
- **Model Artifacts**: Serialized model, scaler, and feature configuration

### Model Specifications
- **Algorithm**: Feed-forward Neural Network
- **Input Features**: Customer demographics and financial metrics
- **Preprocessing**:
  - Gender: Binary encoding (Female=0, Male=1)
  - Geography: One-hot encoding
  - Numerical features: Standard scaling
- **Output**: Churn probability (0-1 range)
- **Decision Threshold**: 0.5 (≥50% = High Risk, <50% = Low Risk)

## Repository Contents

```
├── app.py                 # Main Streamlit application
├── model.ipynb           # Model training notebook
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata
├── uv.lock              # Dependency lock file
├── artifacts/           # Model artifacts directory
│   ├── churn_ann_model.keras    # Trained ANN model
│   ├── scaler.pkl              # Feature scaler
│   └── feature_columns.json    # Feature configuration
└── Artificial_Neural_Network_Case_Study_data.csv  # Training dataset
```

## System Requirements

- Python 3.12+
- TensorFlow 2.21+
- Streamlit 1.44+
- Scikit-learn 1.8+
- Pandas 3.0+
- NumPy 2.4+

## Installation Guide

### 1. Clone Repository
```bash
git clone https://github.com/jhachhotu/Churn-Predictor.git
cd Churn_Prediction_Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import streamlit, tensorflow, sklearn, pandas; print('All dependencies installed')"
```

## Usage Instructions

### Running the Application
```bash
streamlit run app.py
```

The application will start on `http://localhost:8501` (or next available port).

### Input Parameters
- **Credit Score**: Integer (300-850)
- **Geography**: Categorical (France, Germany, Spain)
- **Gender**: Binary (Female/Male)
- **Age**: Integer (18-100)
- **Tenure**: Integer (0-10 years)
- **Balance**: Float (0-250,000)
- **Number of Products**: Integer (1-4)
- **Credit Card**: Boolean
- **Active Member**: Boolean
- **Estimated Salary**: Float (0-500,000)

### Output Metrics
- Churn Probability (%)
- Retention Probability (%)
- Risk Classification (High/Low)
- Risk Indicator (Progress bar)

## Model Training (Optional)

If you wish to retrain the model:

1. Open `model.ipynb` in Jupyter
2. Execute all cells sequentially
3. Verify artifacts are generated in `/artifacts/`
4. Restart the Streamlit application

## Deployment Information

### Local Development
- Run `streamlit run app.py`
- Access via browser at localhost:8501

### Cloud Deployment
- Platform: Streamlit Community Cloud
- URL: https://churn-predictor-dui5eddtfmydpmmxdzejmb.streamlit.app/
- Status: Active

## Important Notes

- Ensure all model artifacts exist before running the application
- The model was trained on historical customer data
- Predictions are probabilistic estimates, not definitive outcomes
- For production use, consider model retraining with recent data
- GPU acceleration not required for inference

## Troubleshooting

### Common Issues
- **Missing artifacts**: Run `model.ipynb` to generate required files
- **Import errors**: Verify all dependencies are installed
- **Port conflicts**: Streamlit will automatically use next available port

### Performance Considerations
- Model inference is optimized for CPU
- Typical prediction time: <100ms
- Memory usage: ~500MB during model loading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

