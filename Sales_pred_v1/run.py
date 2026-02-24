"""
SalesIQ - Run this file to start the application
Usage: python run.py
Then open: http://localhost:5000
"""
import os
import sys
import subprocess

def check_dependencies():
    required = ['flask', 'pandas', 'numpy', 'sklearn', 'openpyxl']
    missing = []
    for pkg in required:
        try:
            __import__(pkg if pkg != 'sklearn' else 'sklearn')
        except ImportError:
            missing.append(pkg if pkg != 'sklearn' else 'scikit-learn')
    return missing

def main():
    print("=" * 50)
    print("   SalesIQ - Sales Prediction Tool")
    print("=" * 50)

    # Check deps
    missing = check_dependencies()
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    print("\n✅ All dependencies found")

    # Optional packages
    try:
        import statsmodels
        print("✅ ARIMA (statsmodels) available")
    except ImportError:
        print("⚠️  ARIMA skipped (statsmodels not installed)")

    try:
        import xgboost
        print("✅ XGBoost available")
    except ImportError:
        print("⚠️  XGBoost using GradientBoosting fallback")

    # Create upload folder
    os.makedirs('static/uploads', exist_ok=True)

    print("\n🚀 Starting server...")
    print("📌 Open your browser at: http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    print("=" * 50)

    from app import app
    app.run(debug=False, port=5000, host='0.0.0.0')

if __name__ == '__main__':
    main()
