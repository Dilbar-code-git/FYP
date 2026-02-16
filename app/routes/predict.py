from flask import Blueprint, request, jsonify, current_app
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from app.utils.data_processor import DataProcessor
from app.models.ml_models import ModelTrainer

bp = Blueprint('predict', __name__, url_prefix='/api/predict')

@bp.route('/train', methods=['POST'])
def train_models():
    """Train ML models on uploaded dataset"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        target_column = data.get('target_column')
        test_size = data.get('test_size', 0.2)
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        if not target_column:
            return jsonify({'error': 'Target column not specified'}), 400
        
        # Load data
        processor = DataProcessor()
        df = processor.load_file(filepath)
        
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found'}), 400
        
        # Ensure target is numeric
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        df = df.dropna(subset=[target_column])
        
        # Select only numeric features (exclude strings/dates)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        trainer = ModelTrainer()
        training_results = trainer.train_all_models(X_train_scaled, y_train)
        evaluation_results = trainer.evaluate_all_models(X_test_scaled, y_test)
        best_model = trainer.get_best_model()
        
        # Format results for display
        formatted_results = {}
        for model_name, metrics in evaluation_results.items():
            formatted_results[model_name] = {
                'performance': metrics.get('performance', {}),
                'errors': metrics.get('errors', {}),
                'quality_score': calculate_quality_score(metrics)
            }
        
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'training_summary': training_results,
            'model_performance': formatted_results,
            'best_model': {
                'model_name': best_model['model_name'],
                'metrics': {
                    'performance': best_model['metrics'].get('performance', {}),
                    'errors': best_model['metrics'].get('errors', {}),
                    'quality_score': calculate_quality_score(best_model['metrics'])
                }
            },
            'data_summary': {
                'features_used': numeric_cols,
                'feature_count': len(numeric_cols),
                'total_samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'target_column': target_column
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def calculate_quality_score(metrics):
    """Calculate overall quality score (0-100)"""
    try:
        performance = metrics.get('performance', {})
        accuracy = performance.get('accuracy', 0)
        r2 = performance.get('r2_score', 0)
        
        # Weight: 50% accuracy, 50% R2 score
        quality_score = (accuracy * 0.5) + (min(r2, 1.0) * 100 * 0.5)
        return round(quality_score, 2)
    except:
        return 0.0


@bp.route('/evaluate', methods=['POST'])
def evaluate_models():
    """Evaluate specific trained models"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        target_column = data.get('target_column')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load data
        df = pd.read_csv(filepath)
        df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
        df = df.dropna(subset=[target_column])
        
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        y = df[target_column]
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate
        trainer = ModelTrainer()
        trainer.train_all_models(X_train_scaled, y_train)
        evaluation_results = trainer.evaluate_all_models(X_test_scaled, y_test)
        best = trainer.get_best_model()
        
        # Format results
        formatted_results = {}
        for model_name, metrics in evaluation_results.items():
            formatted_results[model_name] = {
                'performance': metrics.get('performance', {}),
                'errors': metrics.get('errors', {}),
                'quality_score': calculate_quality_score(metrics)
            }
        
        return jsonify({
            'status': 'success',
            'model_performance': formatted_results,
            'best_model': {
                'model_name': best['model_name'],
                'metrics': {
                    'performance': best['metrics'].get('performance', {}),
                    'errors': best['metrics'].get('errors', {}),
                    'quality_score': calculate_quality_score(best['metrics'])
                }
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
