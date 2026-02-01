import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

class FootfallPredictor:
    def __init__(self):
        self.model = None
        # Features used to predict footfall (9 features, footfall is the target)
        self.feature_names = [
            'price', 'discount', 'promotion_intensity',
            'ad_spend', 'competitor_price', 'stock_level',
            'weather_index', 'customer_sentiment', 'return_rate'
        ]
        self.target = 'footfall'  # What we're predicting
    
    def train(self, csv_path='retail_sales.csv'):
        """
        Train model to predict footfall (customer visits)
        """
        print("\n" + "="*60)
        print("TRAINING FOOTFALL PREDICTION MODEL")
        print("="*60)
        
        # Load data
        print("\n1. Loading dataset...")
        df = pd.read_csv(csv_path)
        print(f"   Dataset shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Verify all columns exist
        print("\n2. Validating columns...")
        all_required = self.feature_names + [self.target]
        missing = [col for col in all_required if col not in df.columns]
        
        if missing:
            print(f"   ❌ Missing columns: {missing}")
            print(f"   Available: {df.columns.tolist()}")
            raise ValueError(f"Missing required columns: {missing}")
        print(f"   ✅ All required columns present")
        
        # Handle missing values
        print("\n3. Checking data quality...")
        missing_counts = df[all_required].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"   Found missing values:\n{missing_counts[missing_counts > 0]}")
            df = df.dropna(subset=all_required)
            print(f"   After cleaning: {len(df)} rows")
        else:
            print(f"   ✅ No missing values ({len(df)} rows)")
        
        # Prepare data
        print("\n4. Preparing features and target...")
        X = df[self.feature_names]
        y = df[self.target]
        
        print(f"   Features (X): {X.shape}")
        print(f"   Target (y - footfall): {y.shape}")
        print(f"   Footfall range: {y.min():.0f} to {y.max():.0f} customers")
        print(f"   Average footfall: {y.mean():.0f} customers")
        
        # Split data
        print("\n5. Splitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        # Train model
        print("\n6. Training Random Forest Regressor...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\n7. Model Performance...")
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"   Training R² Score: {train_score:.4f}")
        print(f"   Testing R² Score: {test_score:.4f}")
        
        if train_score - test_score > 0.1:
            print(f"   ⚠️  Overfitting detected (diff: {train_score - test_score:.4f})")
        else:
            print(f"   ✅ Good generalization!")
        
        # Feature importance
        print("\n8. Feature Importance (what drives customer visits):")
        importances = self.model.feature_importances_
        feature_imp = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (feature, importance) in enumerate(feature_imp, 1):
            bar = '█' * int(importance * 50)
            print(f"   {i}. {feature:25s}: {importance:6.2%} {bar}")
        
        # Save model
        print("\n9. Saving model...")
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target': self.target,
            'train_score': train_score,
            'test_score': test_score
        }
        joblib.dump(model_data, 'footfall_model.joblib')
        print("   ✅ Saved to 'footfall_model.joblib'")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60 + "\n")
        
        return {
            'train_r2': train_score,
            'test_r2': test_score,
            'target': self.target
        }
    
    def load(self):
        """Load pre-trained model"""
        if os.path.exists('footfall_model.joblib'):
            data = joblib.load('footfall_model.joblib')
            self.model = data['model']
            self.feature_names = data.get('feature_names', self.feature_names)
            self.target = data.get('target', 'footfall')
            print(f"✅ Model loaded (predicting: {self.target})")
            return True
        return False
    
    def predict(self, features_dict):
        """Predict footfall (customer visits)"""
        if self.model is None:
            raise ValueError("Model not loaded. Train first!")
        
        # Prepare features (excluding footfall itself)
        features = [features_dict[f] for f in self.feature_names]
        
        # Predict
        prediction = self.model.predict([features])[0]
        
        # Get prediction variance from individual trees
        tree_predictions = [tree.predict([features])[0] for tree in self.model.estimators_]
        std = np.std(tree_predictions)
        
        return {
            'predicted_footfall': round(float(prediction), 0),
            'confidence_lower': round(float(max(0, prediction - 1.96 * std)), 0),
            'confidence_upper': round(float(prediction + 1.96 * std), 0),
            'interpretation': f"Expected {int(prediction)} customer visits"
        }
    
    def get_feature_importance(self):
        """Get what features drive customer visits"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        importances = self.model.feature_importances_
        features = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                'feature': name,
                'importance_percent': round(float(imp * 100), 2),
                'rank': i + 1
            }
            for i, (name, imp) in enumerate(features)
        ]

# Global instance
predictor = FootfallPredictor()
