from flask import Blueprint, request, jsonify, current_app
import os
from werkzeug.utils import secure_filename
from app.utils.data_processor import DataProcessor

bp = Blueprint('upload', __name__, url_prefix='/api/upload')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/file', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format. Use CSV or Excel.'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        # Load and analyze data
        processor = DataProcessor()
        df = processor.load_file(filepath)
        data_info = processor.get_data_info(df)
        numeric_cols = processor.identify_numeric_columns(df)
        datetime_cols = processor.identify_datetime_columns(df)
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'filename': filename,
            'filepath': filepath,
            'data_info': data_info,
            'numeric_columns': numeric_cols,
            'datetime_columns': datetime_cols
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/preview', methods=['POST'])
def preview_data():
    """Preview uploaded file data"""
    try:
        data = request.get_json()
        filepath = data.get('filepath')
        rows = data.get('rows', 5)
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        processor = DataProcessor()
        df = processor.load_file(filepath)
        preview = df.head(rows).to_dict('records')
        
        return jsonify({
            'status': 'success',
            'preview': preview,
            'total_rows': len(df)
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
