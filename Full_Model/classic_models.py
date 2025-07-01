import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")


class ModernBaselineModels:
    def __init__(self):
        self.models = {
            # Linear models
            'linear_regression': LinearRegression(),
            'elastic_net': ElasticNet(),
            'ridge': Ridge(),
            'lasso': Lasso(),

            # Tree-based models
            'random_forest_reg': RandomForestRegressor(n_estimators=200, random_state=42),
            'extra_trees_reg': ExtraTreesRegressor(n_estimators=200, random_state=42),
            'xgboost_reg': xgb.XGBRegressor(n_estimators=200, random_state=42),
            'lightgbm_reg': lgb.LGBMRegressor(n_estimators=200, random_state=42),

            # Classification models
            'logistic_regression': LogisticRegression(max_iter=1000),
            'random_forest_cls': RandomForestClassifier(n_estimators=200, random_state=42),
            'xgboost_cls': xgb.XGBClassifier(n_estimators=200, random_state=42),
            'lightgbm_cls': lgb.LGBMClassifier(n_estimators=200, random_state=42),
        }

        if CATBOOST_AVAILABLE:
            self.models.update({
                'catboost_reg': cb.CatBoostRegressor(iterations=200, verbose=False),
                'catboost_cls': cb.CatBoostClassifier(iterations=200, verbose=False),
            })

        self.scalers = {}
        self.fitted_models = {}
        self.feature_importance = {}

    def prepare_features(self, training_data, text_features):
        feature_columns = [
            'user_nfollower', 'user_nfollowing', 'vid_duration_seconds',
            'post_hour', 'post_day_of_week', 'is_verified', 'music_popularity',
            'num_hashtags', 'caption_length', 'post_is_weekend', 'has_caption',
            'avg_view_growth_rate', 'avg_like_growth_rate', 'avg_comment_growth_rate',
            'avg_share_growth_rate', 'avg_save_growth_rate', 'avg_engagement_growth_rate',
            'std_view_growth_rate', 'std_like_growth_rate', 'std_comment_growth_rate',
            'std_share_growth_rate', 'std_save_growth_rate', 'std_engagement_growth_rate',
            'latest_hours_since_post', 'latest_views', 'latest_likes',
            'latest_comments', 'latest_shares', 'latest_saves'
        ]
        numeric = training_data[feature_columns].fillna(0).values
        text_subset = text_features[:, :100]  # reduce TF-IDF size
        combined = np.hstack([numeric, text_subset])
        return combined, feature_columns

    def train_baseline_models(self, X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls, feature_names):
        results = {}
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers['main'] = scaler

        # Regression Models
        reg_models = [name for name in self.models if name.endswith('_reg') or 'regression' in name]
        for name in reg_models:
            model = self.models[name]
            y_train_avg = np.mean(y_train_reg, axis=1)
            y_val_avg = np.mean(y_val_reg, axis=1)

            if 'linear' in name or name in ['elastic_net', 'ridge', 'lasso']:
                model.fit(X_train_scaled, y_train_avg)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train_avg)
                y_pred = model.predict(X_val)

            results[name] = {
                'mse': mean_squared_error(y_val_avg, y_pred),
                'mae': mean_absolute_error(y_val_avg, y_pred),
                'r2': r2_score(y_val_avg, y_pred),
                'predictions': y_pred
            }

            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(
                    feature_names + [f'tfidf_{i}' for i in range(100)],
                    model.feature_importances_
                ))

            self.fitted_models[name] = model

        # Classification Models
        cls_models = [name for name in self.models if name.endswith('_cls') or 'logistic' in name]
        for name in cls_models:
            model = self.models[name]
            y_train_cls_eng = y_train_cls[:, -1]
            y_val_cls_eng = y_val_cls[:, -1]

            if 'logistic' in name:
                model.fit(X_train_scaled, y_train_cls_eng)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train_cls_eng)
                y_pred = model.predict(X_val)

            results[name] = {
                'accuracy': accuracy_score(y_val_cls_eng, y_pred),
                'f1_macro': f1_score(y_val_cls_eng, y_pred, average='macro'),
                'f1_weighted': f1_score(y_val_cls_eng, y_pred, average='weighted'),
                'predictions': y_pred
            }

            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(
                    feature_names + [f'tfidf_{i}' for i in range(100)],
                    model.feature_importances_
                ))

            self.fitted_models[name] = model

        return results
