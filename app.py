# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import holidays

# Optional: for more flexible date parsing, you can uncomment the line below
# from dateutil.parser import parse

app = Flask(__name__)
CORS(app)

# Load model and scaler
try:
    model = joblib.load('power_forecast_model.pkl')
    scaler = joblib.load('power_forecast_scaler.pkl')
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model or scaler: {e}")


def create_features(start_date, end_date):
    """
    Create features for the date range between start_date and end_date.
    The function computes time-based features, holiday flags, and initializes
    lag/rolling features (filled with zeros) for consistency.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    feature_df = pd.DataFrame(index=date_range)

    # Time-based features
    feature_df['hour_sin'] = np.sin(2 * np.pi * feature_df.index.hour / 24)
    feature_df['hour_cos'] = np.cos(2 * np.pi * feature_df.index.hour / 24)
    feature_df['day_of_week'] = feature_df.index.dayofweek
    feature_df['month'] = feature_df.index.month
    feature_df['is_weekend'] = feature_df.index.dayofweek.isin([5, 6]).astype(int)

    # Holiday feature using France holidays (adjust as needed)
    fr_holidays = holidays.France(years=range(feature_df.index.year.min(), feature_df.index.year.max() + 1))
    feature_df['is_holiday'] = pd.Series(
        [date in fr_holidays for date in feature_df.index.date],
        index=feature_df.index
    ).astype(int)

    # Initialize lag and rolling features (for inference we don't have actual consumption)
    feature_df['lag_24h'] = np.nan
    feature_df['lag_48h'] = np.nan
    feature_df['rolling_7d'] = np.nan
    feature_df['rolling_30d'] = np.nan

    # Fill NaN values with 0 (or you could use another strategy)
    feature_df[['lag_24h', 'lag_48h', 'rolling_7d', 'rolling_30d']] = (
        feature_df[['lag_24h', 'lag_48h', 'rolling_7d', 'rolling_30d']].fillna(0)
    )

    return feature_df


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Request received")
        data = request.json
        print("Data received:", data)

        # Validate input: must have 'days'
        if not data or 'days' not in data:
            return jsonify({'error': 'Missing "days" in request body'}), 400

        # Validate days range
        days = int(data.get('days', 7))
        if days < 1 or days > 30:
            return jsonify({'error': 'Days must be between 1 and 30'}), 400

        # Determine start_date: use provided startDate or default to current time.
        if 'startDate' in data:
            try:
                # If your input format is ISO (e.g., "2025-01-01T00:00"), you can use:
                start_date = datetime.fromisoformat(data['startDate'])
                # Alternatively, use dateutil.parser:
                # start_date = parse(data['startDate'])
            except Exception as ex:
                return jsonify({'error': f'Invalid startDate format: {ex}'}), 400
        else:
            start_date = datetime.now()

        end_date = start_date + timedelta(days=days)
        print(f"Creating features from {start_date} to {end_date}")

        features = create_features(start_date, end_date)
        print("Features created with shape:", features.shape)

        # Check for NaN values (should be handled by our fillna, but double-check)
        if features.isnull().values.any():
            return jsonify({'error': 'Generated features contain NaN values'}), 500

        # Ensure feature count matches the scaler/model requirements
        if scaler.n_features_in_ != features.shape[1]:
            return jsonify({
                'error': f'Feature mismatch: scaler expects {scaler.n_features_in_} features, got {features.shape[1]}'
            }), 500

        # Scale features and make predictions
        scaled_features = scaler.transform(features)
        print("Features scaled successfully")

        predictions = model.predict(scaled_features)
        print("Predictions made successfully")

        return jsonify({
            'timestamps': features.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'predictions': predictions.tolist()
        })

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
