import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import keras.wrappers.scikit_learn
import sklearn
import random
import tensorflow as tf
from ml_battery.utils import *
import pdb

####TODO!!!! THIS WHOLE THING IS BROKED FOR PARALLEL PURPOSES, DUE TO KERAS DEPENDING HEAVILY ON A GLOBAL SESSION!  maybe we can fix it one day.... until then, use the tensorflow versions.

def sessioned(f):
    ''' This creates a session for the model when needed upon calling a method TODO: broke '''
    def sessioned_f(self, *args, **kwargs):
        if not hasattr(self, "sess"):
            self.sess = tf.Session()
        print(f)
        with self.sess.as_default():
            result = f(self, *args, **kwargs)
        print(self)
        return result
    return sessioned_f

class PickleableKerasModel(object):
    ''' Handy mixin for pickling keras models '''
    TEMP_MODEL_FILE_ = "a_temporary_file_for_storing_a_keras_model.h5"
    
    @sessioned
    def __getstate__(self):
        d = dict(self.__dict__)
        if hasattr(self, "model"):
            try:
                name = PickleableKerasModel.TEMP_MODEL_FILE_ + str(random.random())
                self.model.save(name)
                with open(name,"rb") as f:
                    serial_model_data = f.read()
                d["model"] = serial_model_data
            finally:
                os.remove(name)
        return d
    
    @sessioned
    def __setstate__(self, d):
        if "model" in d:
            name = PickleableKerasModel.TEMP_MODEL_FILE_ + str(random.random())
            try:
                with open(name, "wb") as f:
                    f.write(d["model"])
                d["model"] = keras.models.load_model(name)
            finally:
                os.remove(name)
        self.__dict__.update(d)
        
        
class MLPClassifier(keras.wrappers.scikit_learn.KerasClassifier, PickleableKerasModel):
    def __init__(self, hidden_layer_sizes=(100,), n_epochs=100, regularization=0.01, dropout=0.5, batch_size=None, lr=0.001, **kwargs):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_epochs = n_epochs
        self.regularization = regularization
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        super().__init__(**kwargs)
         
    @sessioned
    def fit(self, X, y, sample_weight=None, **kwargs):
        # get the shape of X and one hot y
        X,y = sklearn.utils.check_X_y(X,y)
        if sample_weight is not None: sample_weight = np.array(sample_weight)
        self.input_shape = X.shape[-1]
        self.label_encoder = sklearn.preprocessing.LabelEncoder()
        self.label_encoder.fit(y)
        self.output_shape = len(self.label_encoder.classes_)
        label_encoded = self.label_encoder.transform(y).reshape((-1,1))
        y_onehot = sklearn.preprocessing.OneHotEncoder().fit_transform(label_encoded).toarray()
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))
        if sample_weight is not None: assert not np.any(np.isnan(sample_weight))
        pdb.set_trace()
        super().fit(X,y_onehot,epochs=integerify(self.n_epochs),verbose=1,batch_size=self.batch_size,sample_weight=sample_weight,**kwargs)
        return self
    
    def check_params(self, params):
        #fuckit
        pass
    
    def predict(self, X, **kwargs):
        # gotta inverse the onehot encoding
        pred = self.predict_proba(X,**kwargs)
        best_pred = np.argmax(pred, axis=1)
        return self.label_encoder.inverse_transform(best_pred)
    
    @sessioned
    def predict_proba(self, X, **kwargs):
        return super().predict_proba(X, **kwargs)
    
    @sessioned
    def __call__(self): # the build_fn thing
        # create model
        model = keras.models.Sequential()
        layer_sizes = list(map(integerify, self.hidden_layer_sizes)) + [self.output_shape]
        for i in range(len(layer_sizes)):
            if i < len(layer_sizes)-1:
                activation = "relu"
            else:
                activation = "softmax"
            if i == 0:
                model.add(keras.layers.Dense(layer_sizes[i], input_dim=self.input_shape, kernel_initializer="normal", activation=activation, kernel_regularizer=keras.regularizers.l2(self.regularization)))
            else:
                model.add(keras.layers.Dense(layer_sizes[i], kernel_initializer='normal', activation=activation, kernel_regularizer=keras.regularizers.l2(self.regularization)))
            if i < len(layer_sizes)-1:
                model.add(keras.layers.Dropout(self.dropout))
        # Compile model
        optimizer = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

class OneHiddenLayerNNClassifier(sklearn.base.BaseEstimator, MLPClassifier): #inherit from BaseEstimator so that get_params works right
    def __init__(self, n_hidden_neurons=100, n_epochs=100, regularization=0.01, dropout=0.5, batch_size=None, lr=0.001):
        self.n_hidden_neurons=n_hidden_neurons
        super().__init__(hidden_layer_sizes = (self.n_hidden_neurons,), n_epochs=n_epochs, regularization=regularization, dropout=dropout, batch_size=batch_size, lr=lr)
