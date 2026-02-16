import os
from flask import Flask
from config import config

def create_app(config_name='development'):
    """Application factory function"""
    # Get root directory (parent of app package)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_folder = os.path.join(root_dir, 'templates')
    static_folder = os.path.join(root_dir, 'static')
    
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
    
    # Load configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Register blueprints
    from app.routes import upload, predict, insights, ui
    app.register_blueprint(ui.bp)
    app.register_blueprint(upload.bp)
    app.register_blueprint(predict.bp)
    app.register_blueprint(insights.bp)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app

def register_error_handlers(app):
    """Register error handlers"""
    @app.errorhandler(400)
    def bad_request(error):
        return {'error': 'Bad request'}, 400
    
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Resource not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
