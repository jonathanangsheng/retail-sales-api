import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'price', 'discount', 'promotion_intensity', 'footfall',
            'ad_spend', 'competitor_price', 'stock_level',
            'weather_index', 'customer_sentiment', 'return_rate'
        ]
        self.target_column = None
    
    def detect_target_column(self, df):
        """
        Smart detection of the sales/target column
        Following Kaggle EDA best practices
        """
        # Possible target column names (case-insensitive)
        target_candidates = [
            'sales', 'Sales', 'SALES',
            'units_sold', 'Units_Sold', 'units sold',
            'quantity_sold', 'Quantity_Sold',
            'total_sales', 'Total_Sales',
            'revenue', 'Revenue',
            'demand', 'Demand'
        ]
        
        # First, try exact matches
        for candidate in target_candidates:
            if candidate in df.columns:
                print(f"✅ Found target column: '{candidate}'")
                return candidate
        
        # If not found, find numeric columns not in features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_feature_cols = [col for col in numeric_cols if col not in self.feature_names]
        
        if non_feature_cols:
            target = non_feature_cols[0]
            print(f"⚠️  Target column auto-detected as: '{target}'")
            print(f"   (First numeric column not in features)")
            return target
        
        # Last resort - check all columns
        all_non_features = [col for col in df.columns if col not in self.feature_names]
        if all_non_features:
            print(f"❌ Could not auto-detect target column")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"Non-feature columns: {all_non_features}")
            raise ValueError(f"Please manually specify target column from: {all_non_features}")
        
        raise ValueError(f"Dataset only has feature columns, no target found!")
    
    def train(self, csv_path='retail_sales.csv'):
        """
        Train Random Forest model following EDA best practices
        """
        print("\n" + "="*60)
        print("TRAINING PROCESS")
        print("="*60)
        
        # Load data
        print("\n1. Loading dataset...")
        df = pd.read_csv(csv_path)
        print(f"   Dataset shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Detect target
        print("\n2. Detecting target column...")
        self.target_column = self.detect_target_column(df)
        
        # Check for missing features
        print("\n3. Validating features...")
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            print(f"   ❌ Missing required features: {missing_features}")
            print(f"   Available columns: {df.columns.tolist()}")
            raise ValueError(f"Missing features: {missing_features}")
        print(f"   ✅ All {len(self.feature_names)} features present")
        
        # Handle missing values (EDA step)
        print("\n4. Handling missing values...")
        missing_counts = df[self.feature_names + [self.target_column]].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"   Found missing values:\n{missing_counts[missing_counts > 0]}")
            df = df.dropna(subset=self.feature_names + [self.target_column])
            print(f"   After dropping: {len(df)} rows")
        else:
            print("   ✅ No missing values")
        
        # Prepare data
        print("\n5. Preparing features and target...")
        X = df[self.feature_names]
        y = df[self.target_column]
        
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Target range: {y.min():.2f} to {y.max():.2f}")
        print(f"   Target mean: {y.mean():.2f}")
        
        # Split data
        print("\n6. Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")
        
        # Train model
        print("\n7. Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\n8. Evaluating model...")
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"   Training R² Score: {train_score:.4f}")
        print(f"   Testing R² Score: {test_score:.4f}")
        
        if train_score - test_score > 0.1:
            print(f"   ⚠️  Warning: Possible overfitting (diff: {train_score - test_score:.4f})")
        else:
            print(f"   ✅ Good generalization")
        
        # Feature importance
        print("\n9. Feature importance:")
        importances = self.model.feature_importances_
        feature_imp = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in feature_imp:
            print(f"   {feature:25s}: {importance:.4f} {'█' * int(importance * 50)}")
        
        # Save model
        print("\n10. Saving model...")
        model_data = {
            'model': self.model,
            'target_column': self.target_column,
            'feature_names': self.feature_names,
            'train_score': train_score,
            'test_score': test_score
        }
        joblib.dump(model_data, 'sales_model.joblib')
        print("    ✅ Model saved to 'sales_model.joblib'")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60 + "\n")
        
        return {
            'train_r2': train_score,
            'test_r2': test_score,
            'target_column': self.target_column
        }
    
    def load(self):
        """Load pre-trained model"""
        if os.path.exists('sales_model.joblib'):
            print("Loading saved model...")
            data = joblib.load('sales_model.joblib')
            self.model = data['model']
            self.target_column = data.get('target_column', 'sales')
            self.feature_names = data.get('feature_names', self.feature_names)
            print(f"✅ Model loaded (target: {self.target_column})")
            return True
        return False
    
    def predict(self, features_dict):
        """Make prediction with confidence interval"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load model first.")
        
        # Prepare features in correct order
        features = [features_dict[f] for f in self.feature_names]
        
        # Predict
        prediction = self.model.predict([features])[0]
        
        # Estimate confidence interval
        # Using tree variance method
        predictions_per_tree = []
        for tree in self.model.estimators_:
            predictions_per_tree.append(tree.predict([features])[0])
        
        std = np.std(predictions_per_tree)
        
        return {
            'predicted_sales': round(float(prediction), 2),
            'confidence_lower': round(float(max(0, prediction - 1.96 * std)), 2),
            'confidence_upper': round(float(prediction + 1.96 * std), 2),
            'std_dev': round(float(std), 2)
        }
    
    def get_feature_importance(self):
        """Get ranked feature importance"""
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
                'importance': round(float(imp * 100), 2),
                'rank': i + 1
            }
            for i, (name, imp) in enumerate(features)
        ]

# Global instance
predictor = SalesPredictor()
