import os
import traceback
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np

from ml_models import SalesPredictionEngine
from data_processor import DataProcessor

app = Flask(__name__)
app.secret_key = 'salesiq_secret_2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024   # 50 MB

ALLOWED = {'csv', 'xlsx', 'xls'}

def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload CSV or Excel (.xlsx/.xls)'}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Sanitize filename
    safe_name = os.path.basename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    file.save(filepath)

    try:
        result = DataProcessor().load_and_analyze(filepath)
        session['filepath'] = filepath
        session['columns']  = result['columns']
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    data     = request.get_json(force=True)
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'No uploaded file found. Please upload your data first.'}), 400

    date_col  = data.get('date_column')
    sales_col = data.get('sales_column')
    if not date_col or not sales_col:
        return jsonify({'error': 'Please provide both date_column and sales_column.'}), 400

    try:
        proc = DataProcessor()
        df   = proc.prepare_data(filepath, date_col, sales_col)
        result = proc.get_analysis(df, sales_col)
        session['date_col']  = date_col
        session['sales_col'] = sales_col
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/train', methods=['POST'])
def train():
    data     = request.get_json(force=True)
    filepath = session.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'No uploaded file found. Please upload your data first.'}), 400

    date_col  = session.get('date_col')  or data.get('date_column')
    sales_col = session.get('sales_col') or data.get('sales_column')
    if not date_col or not sales_col:
        return jsonify({'error': 'Column configuration missing. Please go back to Upload and re-analyze.'}), 400

    forecast_periods = int(data.get('forecast_periods', 12))
    forecast_periods = max(3, min(forecast_periods, 36))

    try:
        proc = DataProcessor()
        df   = proc.prepare_data(filepath, date_col, sales_col)
        results = SalesPredictionEngine().train_all_models(df, sales_col, forecast_periods)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/sample-data')
def sample_data():
    try:
        proc     = DataProcessor()
        filepath = proc.generate_sample_data()
        result   = proc.load_and_analyze(filepath)
        session['filepath'] = filepath
        session['columns']  = result['columns']
        return jsonify({**result, 'message': 'Sample dataset loaded successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True, port=5000)
