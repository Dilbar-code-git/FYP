import pandas as pd
import numpy as np
import os


class DataProcessor:

    def _read_file(self, filepath):
        ext = filepath.rsplit('.', 1)[-1].lower()
        if ext == 'csv':
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return pd.read_csv(filepath, encoding=enc)
                except UnicodeDecodeError:
                    continue
            return pd.read_csv(filepath, encoding='utf-8', errors='replace')
        else:
            return pd.read_excel(filepath)

    def load_and_analyze(self, filepath):
        df = self._read_file(filepath)
        columns = list(df.columns)
        dtypes = {col: str(df[col].dtype) for col in columns}
        preview = df.head(5).fillna('').astype(str).to_dict(orient='records')
        shape = df.shape

        date_candidates = []
        sales_candidates = []
        for col in columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ['date', 'time', 'month', 'year', 'period', 'week']):
                date_candidates.append(col)
            if any(k in col_lower for k in ['sale', 'revenue', 'amount', 'price', 'total',
                                              'quantity', 'qty', 'value', 'profit', 'income']):
                sales_candidates.append(col)

        if not sales_candidates:
            sales_candidates = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]

        if not date_candidates:
            for col in columns:
                try:
                    sample = df[col].dropna().head(10).astype(str)
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() >= 5:
                        date_candidates.append(col)
                except Exception:
                    pass

        missing = {col: int(df[col].isnull().sum()) for col in columns}

        return {
            'columns': columns,
            'dtypes': dtypes,
            'preview': preview,
            'shape': {'rows': int(shape[0]), 'cols': int(shape[1])},
            'date_candidates': date_candidates,
            'sales_candidates': sales_candidates,
            'missing_values': missing
        }

    def prepare_data(self, filepath, date_col, sales_col):
        df = self._read_file(filepath)

        # pandas 3.x compatible date parsing (removed infer_datetime_format)
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=False)
        df = df.dropna(subset=[date_col])

        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
        df = df.dropna(subset=[sales_col])
        df = df[df[sales_col] >= 0]
        df = df.sort_values(date_col).reset_index(drop=True)

        # Aggregate to monthly periods
        df['_period'] = df[date_col].dt.to_period('M')
        agg = df.groupby('_period')[sales_col].sum().reset_index()
        agg['Date'] = agg['_period'].dt.to_timestamp()
        agg = agg[['Date', sales_col]].sort_values('Date').reset_index(drop=True)

        if len(agg) < 6:
            raise ValueError(
                f"Only {len(agg)} monthly periods found after aggregation. "
                "Need at least 6. Check that the date column has valid dates."
            )
        return agg

    def get_analysis(self, df, sales_col):
        date_col = df.columns[0]
        sales = df[sales_col]
        dates = df[date_col]

        stats = {
            'total_sales': float(sales.sum()),
            'avg_sales': float(sales.mean()),
            'max_sales': float(sales.max()),
            'min_sales': float(sales.min()),
            'std_sales': float(sales.std()),
            'count': int(len(sales))
        }

        if len(sales) > 1:
            growth = (sales.iloc[-1] - sales.iloc[0]) / (abs(sales.iloc[0]) + 1e-10) * 100
            stats['overall_growth'] = float(growth)
            stats['avg_monthly_growth'] = float(sales.pct_change().dropna().mean() * 100)
        else:
            stats['overall_growth'] = 0.0
            stats['avg_monthly_growth'] = 0.0

        chart_data = {
            'labels': [str(d)[:10] for d in dates],
            'values': [float(v) for v in sales]
        }

        df2 = df.copy()
        df2['month'] = pd.to_datetime(dates).dt.month
        monthly_avg = df2.groupby('month')[sales_col].mean()
        seasonal = {
            'labels': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
            'values': [float(monthly_avg.get(i, 0)) for i in range(1, 13)]
        }

        df2['year'] = pd.to_datetime(dates).dt.year
        yearly = df2.groupby('year')[sales_col].sum()
        yearly_data = {
            'labels': [str(int(y)) for y in yearly.index],
            'values': [float(v) for v in yearly.values]
        }

        return {
            'stats': stats,
            'chart_data': chart_data,
            'seasonal': seasonal,
            'yearly': yearly_data
        }

    def generate_sample_data(self):
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='ME')
        n = len(dates)
        trend = np.linspace(10000, 22000, n)
        seasonal = 3000 * np.sin(2 * np.pi * np.arange(n) / 12 - np.pi / 2) + 3000
        noise = np.random.normal(0, 600, n)
        sales = np.maximum(trend + seasonal + noise, 500).round(2)

        df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Sales': sales,
            'Product': np.random.choice(['Electronics', 'Clothing', 'Food', 'Furniture'], n),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], n)
        })
        os.makedirs('static/uploads', exist_ok=True)
        filepath = 'static/uploads/sample_sales_data.csv'
        df.to_csv(filepath, index=False)
        return filepath
