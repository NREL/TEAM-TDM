import sklearn
import numpy as np
from ml_battery.utils import *

        
# nesting_structure should be a list of list of list of... defining a tree T such that set(T.flatten()) == (np.unique(y)) 
# if set(T) == set(np.unique(y)), then the NestedClassifier is identical to classifier
class NestedClassifier(sklearn.base.BaseEstimator):
    def __init__(self, classifier=None, nesting_structure=None):
        self.nesting_structure = nesting_structure
        self.classifier = classifier
     
    def figure_out_nests_(self):
        self.nests = []
        self.targets = []
        
        for target_value in self.nesting_structure:
            if isinstance(target_value, (list, tuple)):
                self.nests.append(NestedClassifier(self.classifier, target_value))
            else:
                self.targets.append(target_value)
        
        self.targets = np.array(self.targets)        
        self.target_sorter_ = np.argsort(self.targets) # self.targets[self.target_sorter_] is sorted
        self.target_unsorter_ = self.target_sorter_[np.argsort(self.target_sorter_)] # sorted(self.targets)[self.target_unsorter] == self.targets
        self.classifier = sklearn.base.clone(self.classifier)
        
    def flat_targets(self):
        if not hasattr(self, "classes_"):
            self.classes_ = np.array(list(flatten(self.nesting_structure)))
        return self.classes_

    def fit(self, X, y, sample_weight=None, **fit_params):
        self.figure_out_nests_()
        X,y = sklearn.utils.check_X_y(X,y)
        if sample_weight is not None: sample_weight = np.array(sample_weight)
        this_model_y = np.copy(y)

        #encode the target values
        for i, target in enumerate(self.targets):
            target_indices = (y == target)
            this_model_y[target_indices] = i
            
        #ecode the nest values and fit nests
        self.nest_classes_ = np.arange(len(self.nests)) + len(self.targets) #give these unique labels not already taken
        for i, nest in enumerate(self.nests):
            nest_indices = np.nonzero(np.isin(y, nest.flat_targets()))
            X_nest, y_nest = X[nest_indices], y[nest_indices]
            if sample_weight is None:
                weight_nest = None
            else:
                weight_nest = sample_weight[nest_indices]
            nest.fit(X_nest, y_nest, sample_weight = weight_nest, **fit_params)
            this_model_y[nest_indices] = self.nest_classes_[i]
            
        #fit the classifier now that all the ys are properly encoded
        self.classifier.fit(X, this_model_y, sample_weight=sample_weight, **fit_params)

        return self
        
    def predict(self, X):
        proba = self.predict_proba(X)
        argmax = np.argmax(proba, axis=1)
        inverted = self.flat_targets()[argmax]
        return inverted
    
    def predict_proba(self, X):
        X = sklearn.utils.check_array(X)
        top_model_probas = self.classifier.predict_proba(X)
        target_probas = top_model_probas[:,:len(self.targets)][:,self.target_unsorter_] #now the target probas are in the same order as the nesting structure
        nest_probas = top_model_probas[:,len(self.targets):] #the nests should be in the same order as the nesting structure to begin with
        probas = np.zeros((X.shape[0], len(self.flat_targets())))
        nest_i = 0
        target_i = 0
        j = 0
        # the plan here is to get a vector of probas for each target class.
        # top_model_probas includes some target classes, but also the probas of nests, in the order given in self.nesting_structure.
        # the plan, thus, is to iterate through the top_model_probas, replacing a nest proba with y class probas when we get to them
        for i, target_value in enumerate(self.nesting_structure):
            if isinstance(target_value, (list, tuple)):
                # we reached a nest!
                this_nest_probas = self.nests[nest_i].predict_proba(X)
                # get the nest probas, multiply them by the probability of choosing that nest in the top model
                probas[:,j:j+len(self.nests[nest_i].flat_targets())] = np.multiply(this_nest_probas, nest_probas[:,nest_i:nest_i+1])
                nest_i += 1
                j += len(self.nesting_structure[i])
            else:
                # not a nest, so just shove the probabilities in there.
                probas[:,j] = target_probas[:,target_i]
                target_i += 1
                j += 1

        return np.array(probas)
        