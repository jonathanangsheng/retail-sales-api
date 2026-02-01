import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class FootfallPredictor:
    """
    Predict footfall (customer visits) from the other numeric drivers.
    """

    def __init__(self):
        self.model = None

        # IMPORTANT: footfall is the target, so it is not a feature
        self.feature_names = [
            "price",
            "discount",
            "promotion_intensity",
            "ad_spend",
            "competitor_price",
            "stock_level",
            "weather_index",
            "customer_sentiment",
            "return_rate",
        ]
        self.target = "footfall"

        # Stored metadata
        self.train_score = None
        self.test_score = None

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop accidental index columns like "Unnamed: 0"
        df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
        return df

    def train(self, csv_path: str):
        """
        Train model to predict footfall (customer visits).
        """
        print("\n" + "=" * 60)
        print("TRAINING FOOTFALL PREDICTION MODEL")
        print("=" * 60)

        print("\n1. Loading dataset...")
        df = pd.read_csv(csv_path)
        df = self._clean_dataframe(df)

        print(f"   Dataset path: {csv_path}")
        print(f"   Dataset shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")

        print("\n2. Validating columns...")
        required = self.feature_names + [self.target]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print("\n3. Handling missing values...")
        df = df.dropna(subset=required)
        print(f"   Rows after dropna: {len(df)}")

        print("\n4. Preparing X and y...")
        X = df[self.feature_names]
        y = df[self.target]

        if y.nunique() <= 1:
            raise ValueError("Target 'footfall' has no variation. Cannot train model.")

        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Footfall range: {y.min():.0f} to {y.max():.0f}")
        print(f"   Average footfall: {y.mean():.0f}")

        print("\n5. Splitting train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("\n6. Training Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        print("\n7. Evaluating...")
        self.train_score = float(self.model.score(X_train, y_train))
        self.test_score = float(self.model.score(X_test, y_test))
        print(f"   Train R²: {self.train_score:.4f}")
        print(f"   Test  R²: {self.test_score:.4f}")

        print("\n8. Feature importance:")
        ranked = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        for i, (f, imp) in enumerate(ranked, 1):
            bar = "█" * int(imp * 50)
            print(f"   {i}. {f:25s}: {imp:6.2%} {bar}")

        print("\n9. Saving model...")
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "target": self.target,
                "train_score": self.train_score,
                "test_score": self.test_score,
            },
            "footfall_model.joblib",
        )
        print("   Saved to footfall_model.joblib")

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60 + "\n")

        return {
            "train_r2": self.train_score,
            "test_r2": self.test_score,
            "target": self.target,
        }

    def load(self) -> bool:
        if os.path.exists("footfall_model.joblib"):
            data = joblib.load("footfall_model.joblib")
            self.model = data["model"]
            self.feature_names = data.get("feature_names", self.feature_names)
            self.target = data.get("target", "footfall")
            self.train_score = data.get("train_score", None)
            self.test_score = data.get("test_score", None)
            print(f"✅ Model loaded (target: {self.target})")
            return True
        return False

    def predict(self, features_dict: dict) -> dict:
        if self.model is None:
            raise ValueError("Model not loaded")

        features = [features_dict[f] for f in self.feature_names]
        pred = float(self.model.predict([features])[0])

        # Uncertainty proxy: std dev across trees
        tree_preds = [float(t.predict([features])[0]) for t in self.model.estimators_]
        std = float(np.std(tree_preds))

        lower = max(0.0, pred - 1.96 * std)
        upper = pred + 1.96 * std

        return {
            "predicted_footfall": int(round(pred)),
            "confidence_lower": int(round(lower)),
            "confidence_upper": int(round(upper)),
            "std_dev": float(round(std, 4)),
            "interpretation": f"Expected about {int(round(pred))} customer visits",
        }

    def get_feature_importance(self) -> list:
        if self.model is None:
            raise ValueError("Model not loaded")

        ranked = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"feature": name, "importance_percent": round(float(imp * 100), 2), "rank": i + 1}
            for i, (name, imp) in enumerate(ranked)
        ]


predictor = FootfallPredictor()
