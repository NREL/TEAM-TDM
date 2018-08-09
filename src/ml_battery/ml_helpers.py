import sklearn
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.feature_selection
import sklearn.dummy
import bayes_opt
import functools
from ml_battery.utils import *
import ml_battery.log as log
import pickle
import joblib
import time

from collections import Iterable

import ml_battery.nhts_data as nhts_data

def fit_model_(model, X, y, **fit_params):
    ''' Fits a pickled model.  Handy for multiprocessing and such '''
    import ml_battery.log as log
    name, estimator = model
    log.info("training " + name)
    start = time.time()
    estimator = pickle.loads(estimator)
    estimator = estimator.fit(X,y,**fit_params) # in case of hyperparameter optimized thingies, this returns the underlying estimator
    estimator = pickle.dumps(estimator)
    end = time.time()
    log.info("finished training " + name)
    return ((name, estimator), (name, end-start))

def weighted_cv(model_class, X, y, weights, **hyperparameters):
    ''' cross validation with sample_weights '''
    model = model_class(**hyperparameters)
    return -1*cross_val_scores_weighted(model, X, y, weights, cv=5, metrics=[sklearn.metrics.mean_squared_error])

def train_n_score_weighted_pickled_model_(model, X, y, split, sample_weight=None, metrics=[sklearn.metrics.accuracy_score], **fit_params):
    ''' cross validation scoring of a pickled model... useful for multiprocessing model scoring '''
    estimator = pickle.loads(model)
    return train_n_score_weighted(estimator, X, y, split, sample_weight=sample_weight, metrics=metrics, **fit_params)
    
def train_n_score_weighted(model, X, y, split, sample_weight=None, metrics=[sklearn.metrics.accuracy_score], **fit_params):
    ''' cross validation scoring of a model.  Useful for hyperparameter selection '''
    model_clone = sklearn.base.clone(model)
    train_index, test_index = split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    if sample_weight is None:
        weights_train, weights_test = None, None
    else:
        weights_train, weights_test = sample_weight[train_index], sample_weight[test_index]
    model_clone.fit(X_train,y_train,sample_weight=weights_train)
    y_pred = model_clone.predict(X_test)
    scores = []
    for i, metric in enumerate(metrics):
        score = metric(y_test, y_pred, sample_weight = weights_test)
        scores.append(score)
    if hasattr(model_clone, "sess"): #clean up temporary estimators
        model_clone.sess.close()
    return scores
    
def cross_val_scores_weighted_parallel(model, X, y, sample_weight=None, kf=sklearn.model_selection.StratifiedKFold(n_splits=5), metrics=[sklearn.metrics.accuracy_score], **fit_params):
    ''' parallelized cross validation scores... Useful for hyperparameter selection, fast '''
    X,y = sklearn.utils.check_X_y(X,y)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
    kf.get_n_splits(X)
    splits = list(kf.split(X,y))
    estimator = pickle.dumps(model)
    scores = joblib.Parallel(n_jobs=32)(joblib.delayed(train_n_score_weighted_pickled_model_)(estimator, X, y, split, sample_weight=sample_weight, metrics=metrics, **fit_params) for split in splits)
    return np.array(scores)
    
def cross_val_scores_weighted(model, X, y, sample_weight=None, kf=sklearn.model_selection.StratifiedKFold(n_splits=5), metrics=[sklearn.metrics.accuracy_score], **fit_params):
    X,y = sklearn.utils.check_X_y(X,y)
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
    kf.get_n_splits(X)
    scores = []
    for split in kf.split(X,y): #### TODO: THIS NEEDS TO BE WEIGHTED!!!!
        score = train_n_score_weighted(model, X, y, split, sample_weight=sample_weight, metrics=metrics, **fit_params)
        scores.append(score)
    return np.array(scores)
    
class CodebookTransformer(object):
    ''' Transformer that one-hots categorical variables and scales numeric ones.
        Takes in a codebook pandas dataframe with index "Name" of data columns, and a column "Type" that has value "C" for categorical variables.
        X_possible_values is to pass in the entire dataset, so as to successfully one-hot variables even if they aren't in the training split.
        TODO: This should probably be refactored to a CategoricalTransformer and a ScalingTransformer and not take in a codebook the way it does. '''
    def __init__(self, codebook, sep="_____", scale_numeric=True, X_possible_values=None):
        self.codebook = codebook
        self.sep = sep
        self.scale = scale_numeric
        self.X_possible_values = X_possible_values
        
    def fit(self, X, y=None,**fit_params):
        # y is ignored
        log.info("fitting codebook")
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.numeric_columns = []
        encoder_X = self.X_possible_values if self.X_possible_values is not None else X
        for col in X.columns:
            if (col in self.codebook.index) and (self.codebook.loc[col]["Type"] == "C"):
                label_encoder = sklearn.preprocessing.LabelEncoder().fit(encoder_X[[col]].astype(str))
                onehot_encoder = sklearn.preprocessing.OneHotEncoder().fit(np.arange(len(label_encoder.classes_)).reshape((-1,1)))
                self.label_encoders[col] = label_encoder
                self.onehot_encoders[col] = onehot_encoder
            else:
                self.numeric_columns.append(col)
        if self.scale and self.numeric_columns:
        
            self.scaler = sklearn.preprocessing.MinMaxScaler()
            self.scaler.fit(X[self.numeric_columns])
        return self
    
    def transform(self, X):
        onehotted_dataframes = []
        for col in X.columns:
            log.info("transforming column: " + str(col))
            if (col in self.codebook.index) and (self.codebook.loc[col]["Type"] == "C"):
                onehot_encoded = self.onehot_encoders[col].transform(
                    self.label_encoders[col].transform(
                        X[[col]].astype(str)
                    ).reshape((-1,1))).toarray()
                onehotted_dataframes.append(
                    pd.DataFrame(
                        onehot_encoded, 
                        columns = [col + self.sep + cls for cls in self.label_encoders[col].classes_],
                        index = X.index
                    ))
        X = X.drop(self.onehot_encoders, axis=1)
        if self.scale and self.numeric_columns:
            X = pd.DataFrame(self.scaler.transform(X[self.numeric_columns]),columns=self.numeric_columns,index=X.index)
            
        return X.join(onehotted_dataframes)
    
    def fit_transform(self, X, y=None,**fit_params):
        self.fit(X,y, **fit_params)
        return self.transform(X)
        

def xy_accuracy(model,X,y,sample_weight=None):
    ''' sklearn.metrics.accuracy_score takes in y_pred and y_true.  This is a helper that does the prediction step from X and a model '''
    y_pred = model.predict(X)
    return sklearn.metrics.accuracy_score(y, y_pred, sample_weight=sample_weight)
    
def xy_mse(model,X,y,sample_weight=None):
    ''' sklearn.metrics.mean_squared_error takes in y_pred and y_true.  This is a helper that does the prediction step from X and a model '''
    y_pred = model.predict(X)
    return sklearn.metrics.mean_squared_error(y, y_pred, sample_weight=sample_weight)

def cv_weighted_instantiated_model(model, X, y, sample_weight=None, kf=sklearn.model_selection.StratifiedKFold(n_splits=5), metrics=[sklearn.metrics.accuracy_score], parallel=False, **hyperparameters):
    ''' does weighted cv scores for hyperparameter selection '''
    model = sklearn.base.clone(model)
    model.set_params(**hyperparameters)
    if parallel:
        return cross_val_scores_weighted_parallel(model, X, y, sample_weight=sample_weight, kf=kf, metrics=metrics).mean()
    else:
        return cross_val_scores_weighted(model, X, y, sample_weight=sample_weight, kf=kf, metrics=metrics).mean()
    
class HyperparameterOptimizedEstimator(object):
        ''' Class for doing hyperparameter selection.
            Parameters:
                model: an instantiated model.  Any model params are respected, unless they are passed to the hyperparameter_bounds parameters
                parallel: whether to do this in parallel TODO: broken
                kf: the cross validation kfold strategy to use.
                hyperparameter_bounds: any other numerical hyperparameters to `model` can be passed in here, as a tuple e.g. (1,80).
                    The fit method will do a search for the optimal hyperparameters within those bounds. 
                    NOTE: If you pass in integer bounds, it will try to find an integer solution.  Pass in float bounds if the hyperparameter is continuous e.g. (1.0, 7.0)''' 
        def __init__(self, model, parallel=False, kf=sklearn.model_selection.StratifiedKFold(n_splits=5), metric=sklearn.metrics.accuracy_score, **hyperparameter_bounds):
            self.hyperparameter_bounds = hyperparameter_bounds
            self.model = model
            self.parallel = parallel
            self.metric = metric
            self.kf=kf

        def fit(self, X, y, sample_weight=None, **fit_params):
            log.info("hyperparameter optimizing: " + str(self.model))
            bo = bayes_opt.BayesianOptimization(functools.partial(cv_weighted_instantiated_model, self.model, X, y, sample_weight, self.kf, [self.metric], self.parallel),
                                         self.hyperparameter_bounds)
            bo.maximize(init_points=5, n_iter = 30, acq="ei", xi=1e-4) #go greedy (low xi) b/c this takes a long time
            optimal_hyperparameters = {hyperparameter: bo.res["max"]["max_params"][hyperparameter].astype(type(self.hyperparameter_bounds[hyperparameter][0])) for hyperparameter in self.hyperparameter_bounds}
            log.info("optimal: " + str(optimal_hyperparameters))
            self.model.set_params(**optimal_hyperparameters)
            self.model.fit(X,y,sample_weight=sample_weight)
            return self.model
            
class HyperparameterOptimizedClassifier(HyperparameterOptimizedEstimator):
    pass
class HyperparameterOptimizedRegressor(HyperparameterOptimizedEstimator):
    def __init__(self, model, parallel=False, kf=sklearn.model_selection.KFold(n_splits=5), metric=sklearn.metrics.mean_squared_error, **hyperparameter_bounds):
        super().__init__(model, parallel, kf, metric, **hyperparameter_bounds)    
  
class IntegerRegressor(sklearn.base.BaseEstimator):
    ''' Regression that rounds to the nearest integer '''
    def __init__(self, model):
        self.model = model
    def predict_(self,X,*args,**kwargs):
        return np.round(self.model.predict(X,*args,**kwargs)).astype(np.int64)
    def predict(self,X,*args,**kwargs):
        probas = self.predict_proba(X,*args,**kwargs)
        return self.label_encoder.inverse_transform(np.argmax(probas, axis=1))
    def predict_proba(self,X,*args,**kwargs):
        predictions = self.predict_(X,*args,**kwargs)
        failures = predictions[np.isin(predictions, self.classes_, invert=True)]
        predictions[np.isin(predictions, self.classes_, invert=True)] = np.random.choice(self.classes_, size=failures.shape)
        pred_proba = self.onehot_encoder.transform(self.label_encoder.transform(predictions).reshape((-1,1))).toarray()
        return pred_proba
    def fit(self, X, y, **fit_params):
        self.model.fit(X,y,**fit_params)
        self.classes_ = np.unique(y)
        self.label_encoder = sklearn.preprocessing.LabelEncoder()
        self.label_encoder.classes_ = self.classes_
        self.onehot_encoder = sklearn.preprocessing.OneHotEncoder().fit(np.arange(len(self.label_encoder.classes_)).reshape((-1,1)))
        return self

class SelectFromModelPandas(sklearn.feature_selection.SelectFromModel):
    ''' Same as sklearn.model_selection.SelectFromModel, but supports pandas dataframes '''
    def transform(self, X, *args, **kwargs):
        log.info("performing feature selection")
        new_X = super().transform(X, *args, **kwargs)
        return pd.DataFrame(new_X, columns=X.columns[self.get_support()])

class PatchedDummy(sklearn.dummy.DummyClassifier):
    ''' Same as sklearn.dummy.DummyClassifier but supports pandas dataframes '''
    def fit(self,X,y,**fit_args):
        import sklearn.utils
        X,y = sklearn.utils.check_X_y(X,y)
        return super().fit(X,y,**fit_args)
        
class PatchedDummyRegressor(sklearn.dummy.DummyRegressor):
    ''' Same as sklearn.dummy.DummyClassifier but supports pandas dataframes '''
    def fit(self,X,y,**fit_args):
        import sklearn.utils
        X,y = sklearn.utils.check_X_y(X,y)
        return super().fit(X,y,**fit_args)


     


     
     

        


