import os
from app import create_app
from config import config

if __name__ == '__main__':
    # Get configuration from environment or default to development
    env = os.getenv('FLASK_ENV', 'development')
    
    # Create Flask app
    app = create_app(env)
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )
