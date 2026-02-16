from flask import Blueprint, request, jsonify, current_app
import os
import pandas as pd
import numpy as np
from app.utils.data_processor import DataProcessor

bp = Blueprint('insights', __name__, url_prefix='/api/insights')

@bp.route('/summary', methods=['POST'])
def get_summary():
    """Get sales summary insights"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        sales_column = data.get('sales_column')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        processor = DataProcessor()
        df = processor.load_file(filepath)
        
        if sales_column not in df.columns:
            return jsonify({'error': f'Column "{sales_column}" not found'}), 400
        
        sales_data = pd.to_numeric(df[sales_column], errors='coerce').dropna()
        
        summary = {
            'total_sales': float(sales_data.sum()),
            'average_sales': float(sales_data.mean()),
            'median_sales': float(sales_data.median()),
            'std_dev': float(sales_data.std()),
            'min_sales': float(sales_data.min()),
            'max_sales': float(sales_data.max()),
            'variance': float(sales_data.var())
        }
        
        return jsonify({
            'status': 'success',
            'summary': summary
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/growth', methods=['POST'])
def get_growth_insights():
    """Get sales growth patterns"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        date_column = data.get('date_column')
        sales_column = data.get('sales_column')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        processor = DataProcessor()
        df = processor.load_file(filepath)
        
        # Convert to numeric
        df[sales_column] = pd.to_numeric(df[sales_column], errors='coerce')
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        df = df.dropna(subset=[date_column, sales_column])
        df = df.sort_values(date_column)
        
        # Calculate growth rate
        sales_data = df[sales_column].values
        growth_rate = np.diff(sales_data) / sales_data[:-1] * 100
        
        # Group by month/quarter for trends
        df['year_month'] = df[date_column].dt.to_period('M')
        monthly_sales = df.groupby('year_month')[sales_column].sum()
        
        insights = {
            'growth_rate': float(np.mean(growth_rate)),
            'volatility': float(np.std(growth_rate)),
            'trend': 'Upward' if np.mean(growth_rate) > 0 else 'Downward',
            'monthly_trend': monthly_sales.to_dict(orient='index') if hasattr(monthly_sales, 'to_dict') else {}
        }
        
        return jsonify({
            'status': 'success',
            'growth_insights': insights
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/seasonal', methods=['POST'])
def get_seasonal_patterns():
    """Get seasonal trends"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        date_column = data.get('date_column')
        sales_column = data.get('sales_column')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        processor = DataProcessor()
        df = processor.load_file(filepath)
        
        # Extract time features
        df = processor.extract_time_features(df, date_column)
        df[sales_column] = pd.to_numeric(df[sales_column], errors='coerce')
        
        # Seasonal analysis by month
        monthly_avg = df.groupby('month')[sales_column].mean().to_dict()
        quarterly_avg = df.groupby('quarter')[sales_column].mean().to_dict()
        
        insights = {
            'monthly_averages': monthly_avg,
            'quarterly_averages': quarterly_avg,
            'peak_month': max(monthly_avg.items(), key=lambda x: x[1])[0] if monthly_avg else None,
            'low_month': min(monthly_avg.items(), key=lambda x: x[1])[0] if monthly_avg else None
        }
        
        return jsonify({
            'status': 'success',
            'seasonal_patterns': insights
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/comparison', methods=['POST'])
def compare_performance():
    """Compare performance across categories"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        category_column = data.get('category_column')
        sales_column = data.get('sales_column')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        processor = DataProcessor()
        df = processor.load_file(filepath)
        
        df[sales_column] = pd.to_numeric(df[sales_column], errors='coerce')
        
        # Group by category
        category_stats = df.groupby(category_column)[sales_column].agg(['sum', 'mean', 'count']).to_dict()
        
        insights = {
            'categories': list(df[category_column].unique()),
            'statistics': category_stats
        }
        
        return jsonify({
            'status': 'success',
            'comparison': insights
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
