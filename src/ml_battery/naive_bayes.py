import sklearn
import sklearn.naive_bayes
import numpy as np

class MixedNB(sklearn.base.BaseEstimator):
        
    def fit(self,X,y,sample_weight=None):
        X,y = sklearn.utils.check_X_y(X,y)
        self.dcs = [col for col in range(X.shape[1]) if np.array_equal(np.unique(X[:,col]),[0,1])]
        self.ccs = [col for col in range(X.shape[1]) if col not in self.dcs]
        
        if self.ccs:
            self.gnb = sklearn.naive_bayes.GaussianNB()
            self.gnb.fit(X[:,self.ccs],y,sample_weight=sample_weight)
            self.classes_ = self.gnb.classes_
            
        if self.dcs:
            self.mnb = sklearn.naive_bayes.MultinomialNB()
            self.mnb.fit(X[:,self.dcs],y,sample_weight=sample_weight)
            self.classes_ = self.mnb.classes_
            
        return self
        
    def predict(self,X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
    
    def predict_proba(self,X):
        X = sklearn.utils.check_array(X)

        if self.ccs and not self.dcs:
            return self.gnb.predict_proba(X)
        if self.dcs and not self.ccs:
            return self.mnb.predict_proba(X)
        if self.ccs and self.dcs:
            proba = self.gnb.predict_proba(X[:,self.ccs]) * self.mnb.predict_proba(X[:,self.dcs])
            probasums = proba.sum(axis=1)
            for col in range(proba.shape[1]):
                proba[:,col] /= probasums
            return proba
        
    def score(self,X,y,sample_weight=None):
        pred = self.predict(X)
        return sklearn.metrics.accuracy_score(y,pred,sample_weight=sample_weight)