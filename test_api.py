"""
Sales Prediction Tool - API Test Script

This script demonstrates how to use the Sales Prediction Tool API.
Before running, make sure the Flask server is running on localhost:5000
"""

import requests
import json

BASE_URL = "http://localhost:5000/api"

def test_file_upload():
    """Test file upload"""
    print("\n=== Testing File Upload ===")
    
    files = {'file': open('sample_sales_data.csv', 'rb')}
    response = requests.post(f"{BASE_URL}/upload/file", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ File uploaded successfully")
        print(f"  Filename: {data['filename']}")
        print(f"  Filepath: {data['filepath']}")
        filepath = data['filepath']
        return filepath
    else:
        print(f"✗ Upload failed: {response.text}")
        return None

def test_data_preview(filepath):
    """Test data preview"""
    print("\n=== Testing Data Preview ===")
    
    payload = {
        'filepath': filepath,
        'rows': 3
    }
    response = requests.post(f"{BASE_URL}/upload/preview", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Data preview loaded")
        print(f"  Total rows: {data['total_rows']}")
        print(f"  Preview:\n{json.dumps(data['preview'], indent=2)}")
    else:
        print(f"✗ Preview failed: {response.text}")

def test_model_training(filepath):
    """Test model training"""
    print("\n=== Testing Model Training ===")
    
    payload = {
        'filepath': filepath,
        'target_column': 'sales',
        'test_size': 0.2
    }
    response = requests.post(f"{BASE_URL}/predict/train", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Models trained successfully")
        print(f"  Training results: {data['training_results']}")
        print(f"  Best model: {data['best_model']['model_name']}")
        print(f"    RMSE: {data['best_model']['metrics']['rmse']}")
    else:
        print(f"✗ Training failed: {response.text}")

def test_forecasting(filepath):
    """Test sales forecasting"""
    print("\n=== Testing Sales Forecasting ===")
    
    payload = {
        'filepath': filepath,
        'target_column': 'sales',
        'model': 'XGBoost',
        'periods': 12
    }
    response = requests.post(f"{BASE_URL}/predict/forecast", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Forecasting completed")
        print(f"  Model: {data['model']}")
        print(f"  Forecast periods: {data['forecast_periods']}")
        print(f"  Predictions: {data['predictions'][:3]}...")  # Show first 3
    else:
        print(f"✗ Forecasting failed: {response.text}")

def test_insights(filepath):
    """Test business insights"""
    print("\n=== Testing Business Insights ===")
    
    # Summary
    payload = {
        'filepath': filepath,
        'sales_column': 'sales'
    }
    response = requests.post(f"{BASE_URL}/insights/summary", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Sales summary retrieved")
        summary = data['summary']
        print(f"  Total sales: ${summary['total_sales']:.2f}")
        print(f"  Average sales: ${summary['average_sales']:.2f}")
        print(f"  Max sales: ${summary['max_sales']:.2f}")
        print(f"  Min sales: ${summary['min_sales']:.2f}")
    else:
        print(f"✗ Summary failed: {response.text}")

def test_feature_importance(filepath):
    """Test feature importance"""
    print("\n=== Testing Feature Importance ===")
    
    payload = {
        'filepath': filepath,
        'target_column': 'sales',
        'model': 'XGBoost'
    }
    response = requests.post(f"{BASE_URL}/predict/feature-importance", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Feature importance retrieved")
        print(f"  Model: {data['model']}")
        importance = data['feature_importance'][:3]
        for item in importance:
            print(f"    {item['feature']}: {item['importance']:.4f}")
    else:
        print(f"✗ Feature importance failed: {response.text}")

def main():
    """Run all tests"""
    print("=" * 50)
    print("Sales Prediction Tool - API Test")
    print("=" * 50)
    
    # Test file upload
    filepath = test_file_upload()
    if not filepath:
        print("\nFailed to upload file. Stopping tests.")
        return
    
    # Test remaining features
    test_data_preview(filepath)
    test_model_training(filepath)
    test_forecasting(filepath)
    test_insights(filepath)
    test_feature_importance(filepath)
    
    print("\n" + "=" * 50)
    print("Testing completed!")
    print("=" * 50)

if __name__ == '__main__':
    main()
