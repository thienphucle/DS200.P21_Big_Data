import numpy as np

try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False

try:
    import autosklearn.regression
    import autosklearn.classification
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False

try:
    from tpot import TPOTRegressor, TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False

try:
    import h2o
    from h2o.automl import H2OAutoML
    H2O_AVAILABLE = True
    h2o.init()
except:
    H2O_AVAILABLE = False

try:
    from pycaret.regression import setup as pycaret_reg_setup, compare_models as pycaret_compare_models_reg
    from pycaret.classification import setup as pycaret_cls_setup, compare_models as pycaret_compare_models_cls
    PYCARET_AVAILABLE = True
except:
    PYCARET_AVAILABLE = False


class AutoMLModels:
    def __init__(self):
        self.models = {}
        self.available = {
            'flaml': FLAML_AVAILABLE,
            'autosklearn': AUTOSKLEARN_AVAILABLE,
            'tpot': TPOT_AVAILABLE,
            'h2o': H2O_AVAILABLE,
            'pycaret': PYCARET_AVAILABLE
        }

    def train_flaml(self, X_train, y_train, task='regression', time_budget=30):
        if not FLAML_AVAILABLE:
            raise ImportError("FLAML not installed")
        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, task=task, time_budget=time_budget)
        return automl

    def train_autosklearn(self, X_train, y_train, task='regression'):
        if not AUTOSKLEARN_AVAILABLE:
            raise ImportError("AutoSklearn not installed")

        if task == 'regression':
            model = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=60, per_run_time_limit=30)
        else:
            model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60, per_run_time_limit=30)
        model.fit(X_train, y_train)
        return model

    def train_tpot(self, X_train, y_train, task='regression'):
        if not TPOT_AVAILABLE:
            raise ImportError("TPOT not installed")

        if task == 'regression':
            model = TPOTRegressor(generations=5, population_size=20, verbosity=2)
        else:
            model = TPOTClassifier(generations=5, population_size=20, verbosity=2)
        model.fit(X_train, y_train)
        return model

    def train_h2o(self, X_train, y_train, task='regression'):
        if not H2O_AVAILABLE:
            raise ImportError("H2O not installed or initialized")

        import pandas as pd
        train_df = pd.DataFrame(X_train)
        train_df['target'] = y_train
        h2o_frame = h2o.H2OFrame(train_df)

        aml = H2OAutoML(max_runtime_secs=60, seed=1)
        if task == 'regression':
            aml.train(x=h2o_frame.columns[:-1], y='target', training_frame=h2o_frame)
        else:
            h2o_frame['target'] = h2o_frame['target'].asfactor()
            aml.train(x=h2o_frame.columns[:-1], y='target', training_frame=h2o_frame)

        return aml

    def train_pycaret(self, X_train, y_train, task='regression'):
        if not PYCARET_AVAILABLE:
            raise ImportError("PyCaret not installed")

        import pandas as pd
        data = pd.DataFrame(X_train)
        data['target'] = y_train

        if task == 'regression':
            s = pycaret_reg_setup(data=data, target='target', silent=True, verbose=False)
            model = pycaret_compare_models_reg()
        else:
            s = pycaret_cls_setup(data=data, target='target', silent=True, verbose=False)
            model = pycaret_compare_models_cls()

        return model
