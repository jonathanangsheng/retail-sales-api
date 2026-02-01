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
            "price", "discount", "promotion_intensity", "footfall",
            "ad_spend", "competitor_price", "stock_level",
            "weather_index", "customer_sentiment", "return_rate"
        ]
        self.target_column = None

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop accidental index columns like 'Unnamed: 0' that appear when saving CSV with index.
        """
        # Drop any Unnamed columns
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
        return df

    def detect_target_column(self, df: pd.DataFrame) -> str:
        """
        Detect target column safely.
        Prefer known target names first (especially 'sales').
        Never choose Unnamed/index/id-like columns as target.
        """
        # Normalize columns for matching
        cols = list(df.columns)

        # 1) Strong preference: 'sales' exact (most common for your dataset)
        preferred_exact = ["sales", "Sales", "SALES"]
        for c in preferred_exact:
            if c in cols:
                print(f"✅ Found target column: '{c}'")
                return c

        # 2) Other reasonable candidates
        target_candidates = [
            "units_sold", "Units_Sold", "units sold",
            "quantity_sold", "Quantity_Sold",
            "total_sales", "Total_Sales",
            "revenue", "Revenue",
            "demand", "Demand"
        ]
        for c in target_candidates:
            if c in cols:
                print(f"✅ Found target column: '{c}'")
                return c

        # 3) Fallback: pick numeric columns not in features, excluding junk
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        def is_junk(col_name: str) -> bool:
            name = str(col_name).strip().lower()
            return (
                name.startswith("unnamed")
                or name in {"index", "id"}
                or name.endswith("_id")
            )

        non_feature_numeric = [
            c for c in numeric_cols
            if c not in self.feature_names and not is_junk(c)
        ]

        if non_feature_numeric:
            target = non_feature_numeric[0]
            print(f"⚠️  Target column auto-detected as: '{target}'")
            print("   (First numeric column not in features, excluding junk/index columns)")
            return target

        # 4) If we still cannot find: give a clear error
        all_non_features = [c for c in df.columns if c not in self.feature_names]
        raise ValueError(
            "Could not detect a valid target column.\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Non-feature columns: {all_non_features}\n"
            "Please ensure your dataset includes a real target like 'sales'."
        )

    def train(self, csv_path: str = "retail_sales.csv"):
        """
        Train Random Forest model. Expects:
        - feature columns listed in self.feature_names
        - a target column (preferably 'sales')
        """
        print("\n" + "=" * 60)
        print("TRAINING PROCESS")
        print("=" * 60)

        # Load data
        print("\n1. Loading dataset...")
        df = pd.read_csv(csv_path)

        # Clean accidental columns
        df = self._clean_dataframe(df)

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

        # Handle missing values
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

        # Split
        print("\n6. Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")

        # Train
        print("\n7. Training Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0,
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
            print("   ✅ Good generalization")

        # Feature importance
        print("\n9. Feature importance:")
        importances = self.model.feature_importances_
        feature_imp = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        for feature, importance in feature_imp:
            bar = "█" * int(importance * 50)
            print(f"   {feature:25s}: {importance:.4f} {bar}")

        # Save model
        print("\n10. Saving model...")
        model_data = {
            "model": self.model,
            "target_column": self.target_column,
            "feature_names": self.feature_names,
            "train_score": train_score,
            "test_score": test_score,
        }
        joblib.dump(model_data, "sales_model.joblib")
        print("    ✅ Model saved to 'sales_model.joblib'")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60 + "\n")

        return {
            "train_r2": train_score,
            "test_r2": test_score,
            "target_column": self.target_column,
        }

    def load(self):
        """Load pre-trained model"""
        if os.path.exists("sales_model.joblib"):
            print("Loading saved model...")
            data = joblib.load("sales_model.joblib")
            self.model = data["model"]
            self.target_column = data.get("target_column", "sales")
            self.feature_names = data.get("feature_names", self.feature_names)
            print(f"✅ Model loaded (target: {self.target_column})")
            return True
        return False

    def predict(self, features_dict):
        """Make prediction with confidence interval"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load model first.")

        features = [features_dict[f] for f in self.feature_names]
        prediction = self.model.predict([features])[0]

        # Confidence via tree variance
        per_tree = [tree.predict([features])[0] for tree in self.model.estimators_]
        std = float(np.std(per_tree))

        return {
            "predicted_sales": round(float(prediction), 2),
            "confidence_lower": round(float(max(0, prediction - 1.96 * std)), 2),
            "confidence_upper": round(float(prediction + 1.96 * std), 2),
            "std_dev": round(float(std), 2),
        }

    def get_feature_importance(self):
        """Get ranked feature importance"""
        if self.model is None:
            raise ValueError("Model not loaded")

        importances = self.model.feature_importances_
        features = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"feature": name, "importance": round(float(imp * 100), 2), "rank": i + 1}
            for i, (name, imp) in enumerate(features)
        ]


predictor = SalesPredictor()
