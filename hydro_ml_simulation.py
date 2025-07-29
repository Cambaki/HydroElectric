import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def simulate_microgrid(sim_days=7, start_time=None):
    """
    Simulate a microgrid with hydro, solar, wind, diesel, and battery storage
    """
    if start_time is None:
        start_time = datetime.now()
    
    # Create time series (15-minute intervals)
    time_points = sim_days * 24 * 4  # 4 intervals per hour
    timestamps = [start_time + timedelta(minutes=15*i) for i in range(time_points)]
    
    # Create realistic data patterns
    np.random.seed(42)  # For reproducible results
    
    data = []
    battery_soc = 50.0  # Start at 50% charge
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.weekday()
        
        # Hydro generation (more consistent, but varies with season/time)
        hydro_base = 25 + 10 * np.sin(2 * np.pi * hour / 24)
        hydro_kw = max(0, hydro_base + np.random.normal(0, 3))
        
        # Solar generation (peaks during day)
        if 6 <= hour <= 18:
            solar_factor = np.sin(np.pi * (hour - 6) / 12) ** 2
            solar_kw = 30 * solar_factor + np.random.normal(0, 5)
        else:
            solar_kw = 0
        solar_kw = max(0, solar_kw)
        
        # Wind generation (more variable)
        wind_base = 15 + 10 * np.sin(2 * np.pi * hour / 24 + np.pi/3)
        wind_kw = max(0, wind_base + np.random.normal(0, 8))
        
        # Demand pattern (higher during day, varies by day of week)
        demand_base = 40 + 20 * np.sin(2 * np.pi * hour / 24 - np.pi/2)
        if day_of_week < 5:  # Weekday
            demand_multiplier = 1.2
        else:  # Weekend
            demand_multiplier = 0.8
        demand_kw = max(10, demand_base * demand_multiplier + np.random.normal(0, 5))
        
        # Calculate total renewable generation
        renewable_gen = hydro_kw + solar_kw + wind_kw
        
        # Battery and diesel logic
        power_balance = renewable_gen - demand_kw
        
        if power_balance > 0:  # Excess power - charge battery
            if battery_soc < 100:
                charge_amount = min(power_balance, (100 - battery_soc) * 0.5)  # Battery capacity factor
                battery_flow_kw = charge_amount
                battery_soc += charge_amount * 0.1  # Charging efficiency
                battery_soc = min(100, battery_soc)
                diesel_used_kw = 0
            else:
                battery_flow_kw = 0
                diesel_used_kw = 0
        else:  # Deficit - use battery and/or diesel
            deficit = abs(power_balance)
            
            # Try to use battery first
            if battery_soc > 20:  # Keep minimum 20% charge
                battery_discharge = min(deficit, (battery_soc - 20) * 0.5)
                battery_flow_kw = -battery_discharge
                battery_soc -= battery_discharge * 0.1
                remaining_deficit = deficit - battery_discharge
            else:
                battery_flow_kw = 0
                remaining_deficit = deficit
            
            # Use diesel for remaining deficit
            diesel_used_kw = remaining_deficit
        
        # Add some noise and constraints
        battery_soc = np.clip(battery_soc, 0, 100)
        
        data.append({
            'timestamp': ts,
            'hydro_kw': round(hydro_kw, 2),
            'solar_kw': round(solar_kw, 2),
            'wind_kw': round(wind_kw, 2),
            'demand_kw': round(demand_kw, 2),
            'battery_soc': round(battery_soc, 2),
            'battery_flow_kw': round(battery_flow_kw, 2),
            'diesel_used_kw': round(diesel_used_kw, 2),
            'hour': hour,
            'day_of_week': day_of_week
        })
    
    return pd.DataFrame(data)


class DemandPredictor:
    """
    Machine learning model to predict energy demand
    """
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _create_features(self, df):
        """Create features for machine learning"""
        features = df[['hour', 'day_of_week', 'hydro_kw', 'solar_kw', 'wind_kw']].copy()
        
        # Add lag features
        if len(df) > 1:
            features['demand_lag1'] = df['demand_kw'].shift(1).fillna(df['demand_kw'].mean())
        else:
            features['demand_lag1'] = df['demand_kw'].iloc[0] if not df.empty else 40
            
        # Add time-based features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features
    
    def fit(self, df):
        """Train the demand prediction model"""
        features = self._create_features(df)
        target = df['demand_kw']
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(features_scaled, target)
        self.is_fitted = True
        
        return self
    
    def predict(self, df):
        """Predict demand for new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        features = self._create_features(df)
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
    def score(self, df):
        """Get model accuracy score"""
        if not self.is_fitted:
            return 0.0
        
        features = self._create_features(df)
        features_scaled = self.scaler.transform(features)
        return self.model.score(features_scaled, df['demand_kw'])
