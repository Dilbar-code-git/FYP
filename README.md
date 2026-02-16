# Sales Prediction and Business Insights Tool

A comprehensive web-based intelligent Sales Prediction and Business Insights Tool built with Flask and Machine Learning.

## Project Overview

This tool helps businesses:
- Upload sales datasets (CSV/Excel)
- Analyze sales data automatically
- Train multiple ML models (Linear Regression, Random Forest, ARIMA, XGBoost)
- Compare model performance
- Make future sales predictions
- Generate business insights (growth trends, seasonal patterns, performance comparisons)

## Project Structure

```
cpi/
├── app/                          # Flask application package
│   ├── __init__.py              # Flask app factory
│   ├── models/                  # ML models
│   │   ├── __init__.py
│   │   └── ml_models.py         # ModelTrainer class with all algorithms
│   ├── routes/                  # API endpoints
│   │   ├── __init__.py
│   │   ├── upload.py            # File upload routes
│   │   ├── predict.py           # Model training and prediction routes
│   │   └── insights.py          # Business insights routes
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── data_processor.py    # Data loading, cleaning, preprocessing
├── templates/                   # HTML templates
├── static/                      # CSS, JS, images
├── uploads/                     # User uploaded files
├── config.py                    # Configuration settings
├── run.py                       # Application entry point
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── version.py                   # Version info
```

## Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.py` to customize:
- `UPLOAD_FOLDER`: Path for uploaded files
- `ALLOWED_EXTENSIONS`: Supported file types
- `MAX_CONTENT_LENGTH`: Maximum file size
- `TEST_SIZE`: Train-test split ratio (default 0.2)
- `RANDOM_STATE`: Random seed for reproducibility

## API Endpoints

### File Upload
- `POST /api/upload/file` - Upload CSV or Excel file
- `POST /api/upload/preview` - Preview uploaded data

### Model Training & Prediction
- `POST /api/predict/train` - Train all models on dataset
- `POST /api/predict/evaluate` - Evaluate specific models
- `POST /api/predict/forecast` - Make future predictions
- `POST /api/predict/feature-importance` - Get feature importance

### Business Insights
- `POST /api/insights/summary` - Get sales summary statistics
- `POST /api/insights/growth` - Get growth insights and trends
- `POST /api/insights/seasonal` - Get seasonal patterns
- `POST /api/insights/comparison` - Compare performance across categories

## Usage Examples

### 1. Upload File
```bash
curl -X POST -F "file=@sales_data.csv" http://localhost:5000/api/upload/file
```

### 2. Train Models
```bash
curl -X POST http://localhost:5000/api/predict/train \
  -H "Content-Type: application/json" \
  -d '{
    "filepath": "uploads/sales_data.csv",
    "target_column": "sales",
    "test_size": 0.2
  }'
```

### 3. Make Predictions
```bash
curl -X POST http://localhost:5000/api/predict/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "filepath": "uploads/sales_data.csv",
    "target_column": "sales",
    "model": "XGBoost",
    "periods": 12
  }'
```

### 4. Get Insights
```bash
curl -X POST http://localhost:5000/api/insights/summary \
  -H "Content-Type: application/json" \
  -d '{
    "filepath": "uploads/sales_data.csv",
    "sales_column": "sales"
  }'
```

## Running the Application

```bash
python run.py
```

The application will be available at `http://localhost:5000`

## Available Machine Learning Models

1. **Linear Regression** - Simple linear trend analysis
2. **Random Forest** - Ensemble method for non-linear patterns
3. **XGBoost** - Gradient boosting for complex relationships
4. **ARIMA** - Time series forecasting (for time-based data)

## Data Processing Features

- Automatic missing value handling (mean imputation, forward fill, dropping)
- Duplicate removal
- Data normalization and scaling
- Automatic numeric column detection
- Date/time feature extraction
- Train-test data splitting

## Supported File Formats

- CSV (.csv)
- Excel (.xlsx, .xls)

## Model Evaluation Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## Requirements

- Python 3.8+
- Flask 2.3.2
- pandas 2.0.3
- scikit-learn 1.3.0
- XGBoost 2.0.0
- statsmodels 0.14.0
- numpy 1.24.3

## Future Enhancements

- Frontend dashboard with visualizations
- Advanced time series analysis
- Customer segmentation
- Confidence intervals for predictions
- Model persistence (save/load trained models)
- Automated report generation
- API authentication and rate limiting

## License

MIT License

## Support

For issues or questions, please refer to the project documentation or contact the development team.
