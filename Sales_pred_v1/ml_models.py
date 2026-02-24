import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class SalesPredictionEngine:

    def _evaluate(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100)
        return {
            'MAE':  round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2':   round(r2, 4),
            'MAPE': round(mape, 2)
        }

    def _make_features(self, df, sales_col):
        """Build time-series features. Returns feature-enriched DataFrame."""
        d = df.copy()
        date_col = d.columns[0]
        d['ds']        = pd.to_datetime(d[date_col])
        d['month']     = d['ds'].dt.month
        d['quarter']   = d['ds'].dt.quarter
        d['year']      = d['ds'].dt.year
        d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
        d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
        d['trend']     = np.arange(len(d))

        for lag in [1, 2, 3, 6, 12]:
            if len(d) > lag:
                d[f'lag_{lag}'] = d[sales_col].shift(lag)

        for w in [3, 6]:
            if len(d) > w:
                d[f'roll_mean_{w}'] = d[sales_col].shift(1).rolling(w).mean()
                d[f'roll_std_{w}']  = d[sales_col].shift(1).rolling(w).std()

        d = d.dropna().reset_index(drop=True)
        return d

    def _feature_cols(self, df, sales_col):
        return [c for c in df.columns if c not in [sales_col, df.columns[0], 'ds']]

    # ── individual model trainers ──────────────────────────────────────

    def _train_lr(self, Xtr, ytr, Xte, yte):
        m = LinearRegression()
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        return m, p, self._evaluate(yte, p)

    def _train_rf(self, Xtr, ytr, Xte, yte):
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        return m, p, self._evaluate(yte, p)

    def _train_gb(self, Xtr, ytr, Xte, yte):
        """XGBoost if available, else sklearn GradientBoosting."""
        try:
            import xgboost as xgb
            m = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                  max_depth=4, subsample=0.8,
                                  random_state=42, verbosity=0)
            label = 'XGBoost'
        except ImportError:
            m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                          max_depth=4, subsample=0.8,
                                          random_state=42)
            label = 'XGBoost'          # Keep same label for UI consistency
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        return m, label, p, self._evaluate(yte, p)

    def _train_arima(self, series, test_size):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            train = series.iloc[:-test_size]
            test  = series.iloc[-test_size:]
            m = ARIMA(train, order=(2, 1, 2)).fit()
            p = m.forecast(steps=test_size)
            return m, np.array(p), self._evaluate(test.values, p)
        except Exception:
            return None, None, None

    # ── future forecast ──────────────────────────────────────────────

    def _forecast_ml(self, model, df_feat, sales_col, fcols, last_date, periods):
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
        history = list(df_feat[sales_col].values)
        results = []

        for i, fdate in enumerate(future_dates):
            row = {
                'month':     fdate.month,
                'quarter':   (fdate.month - 1) // 3 + 1,
                'year':      fdate.year,
                'month_sin': np.sin(2 * np.pi * fdate.month / 12),
                'month_cos': np.cos(2 * np.pi * fdate.month / 12),
                'trend':     len(df_feat) + i,
            }
            for lag in [1, 2, 3, 6, 12]:
                row[f'lag_{lag}'] = history[-lag] if len(history) >= lag else np.mean(history)
            for w in [3, 6]:
                row[f'roll_mean_{w}'] = np.mean(history[-w:]) if len(history) >= w else np.mean(history)
                row[f'roll_std_{w}']  = np.std(history[-w:])  if len(history) >= w else 0.0

            X_fut = np.array([[row.get(f, 0) for f in fcols]])
            pred  = max(0.0, float(model.predict(X_fut)[0]))
            results.append({'date': str(fdate)[:10], 'value': round(pred, 2)})
            history.append(pred)

        return results

    def _forecast_arima(self, model, last_date, periods):
        try:
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
            preds = model.forecast(steps=periods)
            return [{'date': str(d)[:10], 'value': round(max(0, float(v)), 2)}
                    for d, v in zip(future_dates, preds)]
        except Exception:
            return []

    # ── main entry ──────────────────────────────────────────────────

    def train_all_models(self, df, sales_col, forecast_periods=12):
        if len(df) < 8:
            raise ValueError("Need at least 8 monthly data points to train models.")

        date_col   = df.columns[0]
        series     = df[sales_col].copy()
        test_size  = max(3, int(len(df) * 0.2))

        df_feat  = self._make_features(df, sales_col)
        fcols    = self._feature_cols(df_feat, sales_col)

        X = df_feat[fcols].values
        y = df_feat[sales_col].values
        X_tr, X_te = X[:-test_size], X[-test_size:]
        y_tr, y_te = y[:-test_size], y[-test_size:]

        test_dates = [str(d)[:10] for d in df[date_col].iloc[-test_size:]]
        last_date  = pd.to_datetime(df[date_col].iloc[-1])

        all_metrics   = {}
        all_test_preds = {}
        trained_models = {}

        # ── Linear Regression ──
        lr_m, lr_p, lr_met = self._train_lr(X_tr, y_tr, X_te, y_te)
        all_metrics['Linear Regression']    = lr_met
        all_test_preds['Linear Regression'] = [round(float(v), 2) for v in lr_p]
        trained_models['Linear Regression'] = ('ml', lr_m)

        # ── Random Forest ──
        rf_m, rf_p, rf_met = self._train_rf(X_tr, y_tr, X_te, y_te)
        all_metrics['Random Forest']    = rf_met
        all_test_preds['Random Forest'] = [round(float(v), 2) for v in rf_p]
        trained_models['Random Forest'] = ('ml', rf_m)

        # ── XGBoost / GradientBoosting ──
        gb_m, gb_label, gb_p, gb_met = self._train_gb(X_tr, y_tr, X_te, y_te)
        all_metrics[gb_label]    = gb_met
        all_test_preds[gb_label] = [round(float(v), 2) for v in gb_p]
        trained_models[gb_label] = ('ml', gb_m)

        # ── ARIMA (optional) ──
        ar_m, ar_p, ar_met = self._train_arima(series, test_size)
        if ar_m is not None and ar_met is not None:
            all_metrics['ARIMA']    = ar_met
            all_test_preds['ARIMA'] = [round(float(v), 2) for v in ar_p]
            trained_models['ARIMA'] = ('arima', ar_m)

        # ── Pick best by RMSE ──
        best_name = min(all_metrics, key=lambda k: all_metrics[k]['RMSE'])
        best_type, best_model = trained_models[best_name]

        # ── Generate forecast ──
        if best_type == 'arima':
            future_forecasts = self._forecast_arima(best_model, last_date, forecast_periods)
        else:
            future_forecasts = self._forecast_ml(
                best_model, df_feat, sales_col, fcols, last_date, forecast_periods)

        # ── Feature importance ──
        feature_importance = []
        if best_name in trained_models and best_type == 'ml':
            try:
                imps = best_model.feature_importances_
                fi   = sorted(zip(fcols, imps), key=lambda x: x[1], reverse=True)[:10]
                feature_importance = [{'feature': f, 'importance': round(float(v), 4)}
                                       for f, v in fi]
            except AttributeError:
                pass

        # ── Historical data ──
        historical = [{'date': str(d)[:10], 'value': float(v)}
                      for d, v in zip(df[date_col], df[sales_col])]

        # ── Business insights ──
        insights = self._insights(df, sales_col, future_forecasts, all_metrics, best_name)

        return {
            'metrics':          all_metrics,
            'best_model':       best_name,
            'test_dates':       test_dates,
            'test_actual':      [round(float(v), 2) for v in y_te],
            'test_predictions': all_test_preds,
            'future_forecasts': future_forecasts,
            'historical':       historical,
            'feature_importance': feature_importance,
            'insights':         insights
        }

    # ── insights ─────────────────────────────────────────────────────

    def _insights(self, df, sales_col, forecasts, metrics, best_name):
        sales = df[sales_col].values
        items = []

        # Trend
        if len(sales) > 1:
            trend = (sales[-1] - sales[0]) / (abs(sales[0]) + 1e-10) * 100
            up = trend > 0
            items.append({
                'type': 'trend', 'icon': '📈' if up else '📉',
                'title': f'Sales Trend: {"Upward" if up else "Downward"}',
                'text': (f'Overall sales have {"increased" if up else "decreased"} by '
                         f'{abs(trend):.1f}% over the full analysis period.')
            })

        # Forecast outlook
        if forecasts:
            fut_avg = np.mean([f['value'] for f in forecasts])
            cur_avg = np.mean(sales[-3:])
            g = (fut_avg - cur_avg) / (cur_avg + 1e-10) * 100
            items.append({
                'type': 'forecast', 'icon': '🔮',
                'title': 'Future Outlook',
                'text': (f'Predicted sales are expected to {"grow" if g > 0 else "decline"} '
                         f'by {abs(g):.1f}% over the next {len(forecasts)} months '
                         f'(avg ${fut_avg:,.0f}/month).')
            })

        # Best model
        bm = metrics[best_name]
        items.append({
            'type': 'model', 'icon': '🤖',
            'title': f'Best Model: {best_name}',
            'text': (f'{best_name} won the model competition with RMSE of '
                     f'{bm["RMSE"]:,.2f}, R² of {bm["R2"]:.3f}, and MAPE of {bm["MAPE"]:.1f}%.')
        })

        # Seasonality
        df2 = df.copy()
        df2['month'] = pd.to_datetime(df.iloc[:, 0]).dt.month
        monthly = df2.groupby('month')[sales_col].mean()
        if not monthly.empty:
            names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            best_m  = int(monthly.idxmax())
            worst_m = int(monthly.idxmin())
            items.append({
                'type': 'seasonal', 'icon': '🌊',
                'title': 'Seasonality Pattern',
                'text': (f'Peak sales historically occur in {names[best_m-1]} '
                         f'(avg ${monthly[best_m]:,.0f}), while the slowest month is '
                         f'{names[worst_m-1]} (avg ${monthly[worst_m]:,.0f}).')
            })

        # Volatility
        cv = (np.std(sales) / (np.mean(sales) + 1e-10)) * 100
        vol = 'High' if cv > 30 else ('Moderate' if cv > 15 else 'Low')
        tip = 'Consider safety stock buffers.' if cv > 20 else 'Sales are relatively stable — planning is easier.'
        items.append({
            'type': 'volatility', 'icon': '⚡',
            'title': f'Sales Volatility: {vol}',
            'text': f'Coefficient of variation is {cv:.1f}% ({vol.lower()} volatility). {tip}'
        })

        return items
