import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'price', 'discount', 'promotion_intensity', 'footfall',
            'ad_spend', 'competitor_price', 'stock_level',
            'weather_index', 'customer_sentiment', 'return_rate'
        ]
    
    def train(self, csv_path='retail_sales.csv'):
        """Train the Random Forest model"""
        print("Loading data...")
        df = pd.read_csv(csv_path)
        
        X = df[self.feature_names]
        y = df['sales']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        score = self.model.score(X_test, y_test)
        print(f"Model trained! R² Score: {score:.4f}")
        
        # Save model
        joblib.dump(self.model, 'sales_model.joblib')
        return score
    
    def load(self):
        """Load trained model"""
        if os.path.exists('sales_model.joblib'):
            self.model = joblib.load('sales_model.joblib')
            return True
        return False
    
    def predict(self, features_dict):
        """Predict sales from features"""
        features = [features_dict[f] for f in self.feature_names]
        prediction = self.model.predict([features])[0]
        
        # Simple confidence interval
        std = prediction * 0.15
        return {
            'predicted_sales': round(prediction, 2),
            'confidence_lower': round(max(0, prediction - 1.96 * std), 2),
            'confidence_upper': round(prediction + 1.96 * std, 2)
        }
    
    def get_feature_importance(self):
        """Get top features affecting sales"""
        importances = self.model.feature_importances_
        features = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {'feature': name, 'importance': round(imp * 100, 2)}
            for name, imp in features[:5]
        ]

# Global instance
predictor = SalesPredictor()
