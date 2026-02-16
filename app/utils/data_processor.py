import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

class DataProcessor:
    """Class to handle data loading, cleaning, and preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
    
    @staticmethod
    def load_file(filepath):
        """Load CSV or Excel file"""
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            return df
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    @staticmethod
    def get_data_info(df):
        """Get basic information about the dataset"""
        return {
            'rows': len(df),
            'columns': list(df.columns),
            'column_count': len(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': len(df[df.duplicated()]),
            'statistics': df.describe().to_dict('index')
        }
    
    @staticmethod
    def identify_numeric_columns(df):
        """Identify numeric columns"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    @staticmethod
    def identify_datetime_columns(df):
        """Identify date/time columns"""
        datetime_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col], errors='coerce')
                if df[col].dtype == 'object':
                    datetime_cols.append(col)
            except:
                pass
        return datetime_cols
    
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values"""
        numeric_cols = self.identify_numeric_columns(df)
        
        if strategy == 'mean':
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        elif strategy == 'drop':
            df = df.dropna()
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        return df.drop_duplicates()
    
    def normalize_data(self, df, numeric_cols=None):
        """Normalize numeric columns"""
        if numeric_cols is None:
            numeric_cols = self.identify_numeric_columns(df)
        
        df_copy = df.copy()
        df_copy[numeric_cols] = self.scaler.fit_transform(df_copy[numeric_cols])
        return df_copy
    
    def prepare_for_modeling(self, df, target_column=None, test_size=0.2):
        """Prepare data for modeling"""
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df)
        
        # Get numeric columns
        numeric_cols = self.identify_numeric_columns(df)
        
        # Normalize data
        df = self.normalize_data(df, numeric_cols)
        
        # Split data if target is specified
        if target_column and target_column in df.columns:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
        
        return {'data': df, 'numeric_cols': numeric_cols}

    @staticmethod
    def extract_time_features(df, date_column):
        """Extract time-based features from datetime column"""
        df[date_column] = pd.to_datetime(df[date_column])
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['quarter'] = df[date_column].dt.quarter
        df['day_of_week'] = df[date_column].dt.dayofweek
        df['week_of_year'] = df[date_column].dt.isocalendar().week
        return df
