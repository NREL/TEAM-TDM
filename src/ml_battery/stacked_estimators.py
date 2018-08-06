import pandas as pd
import numpy as np
import sklearn
import sklearn.utils.metaestimators
import time
import joblib
import functools
import pickle
from ml_battery.utils import *
from ml_battery.ml_helpers import *
from ml_battery.ordinal_regression import *
import ml_battery.permutation_importance as permutation_importance
import ml_battery.log as log
    
def pickle_estimators(estimators):
    return list(map(lambda x: (x[0], pickle.dumps(x[1])), estimators))
def unpickle_estimators(estimators):
    return list(map(lambda x: (x[0], pickle.loads(x[1])), estimators))

def classifier_score_battery(y_onehot, y_pred_proba, y_true, y_pred, sample_weight=None):
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    
    weighted_f1_score = sklearn.metrics.f1_score(y_true, y_pred, average="weighted", sample_weight=sample_weight)
    weighted_precision = sklearn.metrics.precision_score(y_true, y_pred, average="weighted", sample_weight=sample_weight)
    weighted_recall = sklearn.metrics.recall_score(y_true, y_pred, average="weighted", sample_weight=sample_weight)
    
    macro_f1_score = sklearn.metrics.f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)
    macro_precision = sklearn.metrics.precision_score(y_true, y_pred, average="macro", sample_weight=sample_weight)
    macro_recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro", sample_weight=sample_weight)
            
    log_loss = sklearn.metrics.log_loss(y_onehot, y_pred_proba)  

    confusion = sklearn.metrics.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    
    total_true_by_class, total_pred_by_class = np.sum(confusion, 1), np.sum(confusion, 0)
    absolute_error = np.abs(total_true_by_class - total_pred_by_class)
    macro_mean_absolute_market_share_percent_error = (absolute_error/total_true_by_class).sum()
    weighted_mean_absolute_market_share_percent_error = absolute_error.sum()/total_true_by_class.sum()

    #put it all together
    score_battery = [accuracy, weighted_f1_score, weighted_precision, weighted_recall, macro_f1_score, macro_precision, macro_recall, log_loss, macro_mean_absolute_market_share_percent_error, weighted_mean_absolute_market_share_percent_error]
    
    return score_battery, confusion

def regressor_score_battery(y_true, y_pred, sample_weight=None):
    evs = sklearn.metrics.explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    r2 = sklearn.metrics.r2_score(y_true, y_pred, sample_weight=sample_weight)

    #put it all together
    score_battery = [evs, mae, mse, r2]
    
    return score_battery

class MultiEstimator(sklearn.utils.metaestimators._BaseComposition):
    def __init__(self, estimators, parallel=False):
        self.estimators = estimators
        self.parallel = parallel

    def cleanup(self):
        for estimator in self.estimators: #clean up temporary estimators
            if hasattr(estimator, "sess"):
                estimator.sess.close()
        
    def get_params(self, deep=True):
        return self._get_params("estimators", deep=deep)
    def set_params(self, **kwargs):
        self._set_params("estimators", **kwargs)
        return self

    def fit(self, X, y, **fit_params):
        if self.parallel:
            fitted_models_and_times = joblib.Parallel(n_jobs=32)(joblib.delayed(fit_model_)(model, X, y, **fit_params) for model in pickle_estimators(self.estimators))
            fitted_models, times = zip(*fitted_models_and_times)
            self.estimators, self.fit_times = unpickle_estimators(fitted_models), dict(times) 
            return self
        else:
            self.fit_times = {}
            for i, (name, estimator) in enumerate(self.estimators):
                log.info("training: " + name)
                log.info("opened sessions: " + n_opened_sessions())
                start = time.time()
                self.estimators[i] = (name, estimator.fit(X,y,**fit_params)) # in case of hyperparameter optimized thingies, this returns the underlying estimator
                end = time.time()
                self.fit_times[name] = end-start
            return self

    def predict(self, X):
        prediction_battery = []
        for name, model in self.estimators:
            pred = model.predict(X)
            prediction_battery.append((name, pred))
        return prediction_battery
        
    
class MultiRegressor(MultiEstimator):
    def score_model_(self, model, X, y, sample_weight=None):
        y_true = y
        y_pred = model.predict(X)
    
        score_battery = regressor_score_battery(y_true, y_pred, sample_weight=sample_weight)
        
        base_score, score_decreases = permutation_importance.get_score_importances(functools.partial(xy_mse, model), X, y, sample_weight)
        feature_importances = np.mean(score_decreases, axis=0)
        
        return score_battery, feature_importances
        
    def score(self, X, y, sample_weight=None):
        score_battery = {}
        feature_importances = {}
        
        for name, model in self.estimators:
            score_battery[name], feature_importances[name] = self.score_model_(model, X, y, sample_weight=sample_weight)
            score_battery[name] += [self.fit_times[name]]
       
        return (
            pd.DataFrame(score_battery, index=["explained variance", "mean absolute error", "mean squared error", "R^2", "training time"]),
            pd.DataFrame(feature_importances, index=X.columns)
        )
    
class MultiClassifier(MultiEstimator):
    def predict_proba(self, X):
        proba_battery = []
        for name, model in self.estimators:
            # try to do probability things.  Some methods do not give probabilities, in which case, we'll just say they give 100% for the predicted class and 0% for other classes.
            try:
                pred_proba = model.predict_proba(X)
            except:
                label_encoder = sklearn.preprocessing.LabelEncoder()
                label_encoder.classes_ = model.classes_
                onehot_encoder = sklearn.preprocessing.OneHotEncoder().fit(np.arange(len(label_encoder.classes_)).reshape((-1,1)))
                pred = model.predict(X)
                pred_proba = onehot_encoder.transform(label_encoder.transform(pred).reshape((-1,1))).toarray()
            proba_battery.append((name, pred_proba))
        return proba_battery
        
    def score_model_(self, model, X, y, sample_weight=None): 
        pred = model.predict(X)
        # try to do probability things.  Some methods do not give probabilities, in which case, we'll just say they give 100% for the predicted class and 0% for other classes.
        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.classes_ = model.classes_
        onehot_encoder = sklearn.preprocessing.OneHotEncoder().fit(np.arange(len(label_encoder.classes_)).reshape((-1,1)))
        y_onehot = onehot_encoder.transform(label_encoder.transform(y).reshape((-1,1)))
        try:
            pred_proba = model.predict_proba(X)
        except:
            pred_proba = onehot_encoder.transform(label_encoder.transform(pred).reshape((-1,1))).toarray()

        score_battery, confusion = classifier_score_battery(y_onehot, pred_proba, y, pred, sample_weight=sample_weight)
            
        base_score, score_decreases = permutation_importance.get_score_importances(functools.partial(xy_accuracy, model), X, y, sample_weight)
        feature_importances = np.mean(score_decreases, axis=0)
        
        return score_battery, confusion, feature_importances
        
    def score(self, X, y, sample_weight=None):
        score_battery = {}
        confusion_battery = {}
        feature_importances = {}
        
        for name, model in self.estimators:
            score_battery[name], confusion_battery[name], feature_importances[name] = self.score_model_(model, X, y, sample_weight=sample_weight)
            score_battery[name] += [self.fit_times[name]]
       
        return (
            pd.DataFrame(score_battery, index=["accuracy", "weighted f1", "weighted precision", "weighted recall", "macro f1", "macro precision", "macro recall", "log loss", "macro mean absolute market share error", "weighted mean absolute market share error", "training time"]), 
            {name: pd.DataFrame(confusion_battery[name], index=model.classes_, columns=model.classes_) for name in confusion_battery},
            pd.DataFrame(feature_importances, index=X.columns)
        )
    
 
class StackedEstimator(sklearn.base.BaseEstimator):
    def __init__(self, multiestimator=None, metaestimator=None, kfold=sklearn.model_selection.KFold(n_splits=5)):
        self.multiestimator = multiestimator
        self.metaestimator = metaestimator
        self.kfold = kfold
        super().__init__()
        
    def predict(self, X):
        stacked_features = self.get_stacked_features_(self.multiestimator, X)
        return self.metaestimator.predict(np.hstack((X, stacked_features)))
               
    def fit(self, X, y, sample_weight=None):
        start = time.time()
    
        self.set_output_shape_(y)
        #first, fit the multiestimator, to do the whole hyperparameter thing
        log.info("training multiestimator")
        self.multiestimator.fit(X,y,sample_weight=sample_weight)
        
        #check X,y,sample_weight.... the built in cv splitting stuff doesn't like pandas shit
        X,y = sklearn.utils.check_X_y(X,y)
        
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            
        self.kfold.get_n_splits(X)
        #for each cv split, train the models on the train splits, predict on the test split, and keep those as new features for the meta estimator
        new_features = np.zeros((X.shape[0], len(self.multiestimator.estimators)*self.output_shape_))
        for train_index, test_index in self.kfold.split(X,y):
            X_meta_train, X_meta_test = X[train_index], X[test_index]
            y_meta_train, y_meta_test = y[train_index], y[test_index]
            if sample_weight is not None:
                sample_weight_meta_train, sample_weight_meta_test = sample_weight[train_index], sample_weight[test_index]
            else:
                sample_weight_meta_train, sample_weight_meta_test = None, None
            cloned_multi = sklearn.base.clone(self.multiestimator)
            log.info("cv training")
            cloned_multi.fit(X_meta_train, y_meta_train, sample_weight=sample_weight_meta_train)
            new_features[test_index] = self.get_stacked_features_(cloned_multi, X_meta_test) 
            cloned_multi.cleanup() #clean up temporary estimators

        new_X = np.hstack((X,new_features))
        log.info("training metaestimator")
        self.metaestimator.fit(new_X,y,sample_weight)
        log.info("finished training metaestimator")
        
        end = time.time()
        self.fit_time = end-start
        
        return self   


class StackedRegressor(StackedEstimator):
    def get_stacked_features_(self, multiestimator, X):
        new_features = np.zeros((X.shape[0], len(multiestimator.estimators)*self.output_shape_))
        predictions = multiestimator.predict(X)
        for i, (name, model) in enumerate(predictions):
            new_features[:,i*self.output_shape_:(i+1)*self.output_shape_] = predictions[i][1].reshape((-1, self.output_shape_))
        return new_features

    def set_output_shape_(self, y):
        if len(y.shape) > 1:
            self.output_shape_ = y.shape[1]
        else:
            self.output_shape_ = 1        
            
    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)

        score_battery = regressor_score_battery(y, pred, sample_weight = sample_weight)
        score_battery += [self.fit_time]
            
        base_score, score_decreases = permutation_importance.get_score_importances(functools.partial(xy_mse, self), X, y, sample_weight=sample_weight)
        feature_importances = np.mean(score_decreases, axis=0)
                
        return (
            pd.Series(score_battery, index=["explained variance", "mean absolute error", "mean squared error", "R^2", "training time"]),
            pd.Series(feature_importances, index=X.columns)
        ), self.multiestimator.score(X,y,sample_weight=sample_weight)
        
        
class StackedClassifier(StackedEstimator):
    def __init__(self, multiestimator=None, metaestimator=None, kfold=sklearn.model_selection.StratifiedKFold(n_splits=5)):
        super().__init__(multiestimator, metaestimator, kfold)
        
    def get_stacked_features_(self, multiestimator, X):
        new_features = np.zeros((X.shape[0], len(multiestimator.estimators)*len(self.classes_)))
        predictions = multiestimator.predict_proba(X)
        for i, (name, model) in enumerate(predictions):
            new_features[:,i*len(self.classes_):(i+1)*len(self.classes_)] = predictions[i][1]
        return new_features
            
    def set_output_shape_(self, y):
        self.classes_ = np.unique(y)
        self.output_shape_ = len(self.classes_)
        self.label_encoder = sklearn.preprocessing.LabelEncoder()
        self.label_encoder.classes_ = self.classes_
        self.onehot_encoder = sklearn.preprocessing.OneHotEncoder().fit(np.arange(len(self.label_encoder.classes_)).reshape((-1,1)))
            
    def predict_proba(self, X):
        stacked_features = self.get_stacked_features_(self.multiestimator, X)
        return self.metaestimator.predict_proba(np.hstack((X, stacked_features)))
    
    def score(self,X,y,sample_weight=None):
        pred = self.predict(X)
        y_onehot = self.onehot_encoder.transform(self.label_encoder.transform(y).reshape((-1,1)))
        try:
            pred_proba = self.predict_proba(X)
        except:
            pred_proba = self.onehot_encoder.transform(self.label_encoder.transform(pred).reshape((-1,1))).toarray()

        score_battery, confusion = classifier_score_battery(y_onehot, pred_proba, y, pred, sample_weight = sample_weight)
        score_battery += [self.fit_time]
        
        base_score, score_decreases = permutation_importance.get_score_importances(functools.partial(xy_accuracy, self), X, y, sample_weight=sample_weight)
        feature_importances = np.mean(score_decreases, axis=0)
                
        return (
            pd.Series(score_battery, index=["accuracy", "weighted f1", "weighted precision", "weighted recall", "macro f1", "macro precision", "macro recall", "log loss", "macro mean absolute market share error", "weighted mean absolute market share error", "training time"]), 
            pd.DataFrame(confusion, index=self.label_encoder.classes_, columns=self.label_encoder.classes_),
            pd.Series(feature_importances, index=X.columns)
        ), self.multiestimator.score(X,y,sample_weight=sample_weight)