import os
import sys
from flask import jsonify

def create_sample_data():
    """Create sample CSV data for testing"""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create sample sales data
    dates = [datetime(2022, 1, 1) + timedelta(days=x) for x in range(365)]
    sales = [100 + (x % 30) * 2 + (x // 30) * 5 for x in range(365)]
    quantity = [x % 50 + 10 for x in range(365)]
    region = ['North', 'South', 'East', 'West'] * 91 + ['North'] * 1
    category = ['Electronics', 'Clothing', 'Food', 'Sports'] * 91 + ['Electronics']
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'quantity': quantity,
        'region': region,
        'category': category
    })
    
    return df

def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Sales Prediction Tool is running'
    }), 200

def get_api_status():
    """Get API status"""
    return {
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            'upload': [
                'POST /api/upload/file',
                'POST /api/upload/preview'
            ],
            'predict': [
                'POST /api/predict/train',
                'POST /api/predict/evaluate',
                'POST /api/predict/forecast',
                'POST /api/predict/feature-importance'
            ],
            'insights': [
                'POST /api/insights/summary',
                'POST /api/insights/growth',
                'POST /api/insights/seasonal',
                'POST /api/insights/comparison'
            ]
        }
    }
