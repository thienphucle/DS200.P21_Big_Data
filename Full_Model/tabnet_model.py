import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

try:
    from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("❌ TabNet not installed. Run: pip install pytorch-tabnet")

class TabNetModel:
    def __init__(self):
        if not TABNET_AVAILABLE:
            raise ImportError("TabNet not available.")
        self.regressor = None
        self.classifier = None
        self.scaler = RobustScaler()

    def fit_regression(self, X, y, max_epochs=100):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        self.regressor = TabNetRegressor(verbose=0)
        self.regressor.fit(X_train, y_train, eval_set=[(X_val, y_val)], patience=10)

        preds = self.regressor.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        return self.regressor, mse

    def fit_classification(self, X, y, max_epochs=100):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        self.classifier = TabNetClassifier(verbose=0)
        self.classifier.fit(X_train, y_train, eval_set=[(X_val, y_val)], patience=10)

        preds = self.classifier.predict(X_val)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='macro')
        return self.classifier, acc, f1
