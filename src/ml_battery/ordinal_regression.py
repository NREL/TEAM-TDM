import tensorflow as tf
import sklearn
import numpy as np
import os
import zipfile
import random
from ml_battery.utils import *
from ml_battery.tensorflow_models import *
import ml_battery.log as log
     
class OrdinalRegression(sklearn.base.BaseEstimator, PickleableTFModel, TFClassifierMixin):
    def __init__(self, inverse_link_function="logit", n_epochs=1000, batch_size=None):
        self.inverse_link_function = inverse_link_function
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        super().__init__()

    def fit_(self, X, y, sample_weight=None, feed_dict_extras={}):
        if len(y.shape) == 1:
            y = y.reshape((-1,1))
        return TFEstimatorMixin.fit_(self,X,y,sample_weight=sample_weight, feed_dict_extras=feed_dict_extras)
        
    def inverse_link_function_(self):
        if self.inverse_link_function == "logit": return tf.sigmoid
        if self.inverse_link_function == "probit": return tf.distributions.Normal(0.,1.).cdf  

    def build_model_(self):
        # create model
        n_features = list(self.input_shape_)
        self.model.theta = tf.Variable(tf.truncated_normal([len(self.classes_)-1], stddev=0.5), dtype="float", name="theta")
        self.model.sorted_theta = tf.contrib.framework.sort(self.model.theta)
        self.model.weights = tf.Variable(tf.truncated_normal(n_features + [1], stddev=0.1), dtype="float", name="weights")
        output_range = tf.constant(self.classes_, dtype="float")
        self.model.x = tf.placeholder("float", shape=[None] + n_features)
        self.model.y = tf.placeholder("float", shape=(None, 1))
        self.model.sample_weight = tf.placeholder("float", shape=(None,))
        unlinked = self.model.sorted_theta-tf.matmul(self.model.x,self.model.weights)
        linked = self.inverse_link_function_()(unlinked)
        
        link_upper_padding = tf.constant([[0,0],[0,1]], dtype="int32") # because P(y<=\inf)=1
        link_lower_padding = tf.constant([[0,0],[1,0]], dtype="int32") # because P(y<=-\inf)=0
        linked = tf.pad(linked, link_upper_padding, mode="CONSTANT", constant_values=1.)
        linked = tf.pad(linked, link_lower_padding, mode="CONSTANT", constant_values=0.)
        self.model.predict_proba = linked[:,1:] - linked[:,:-1]
        
        y_mask = tf.equal(output_range, self.model.y)
        
        loss_high = tf.boolean_mask(linked[:,1:], y_mask)
        loss_low = tf.boolean_mask(linked[:,:-1], y_mask)
        self.model.loss = -tf.log(tf.nn.relu(loss_high - loss_low)+0.001)*self.model.sample_weight #relu to prevent nans... this enforces the order on the thetas

        optimizer = tf.train.AdamOptimizer(0.01)
        self.model.train_step = optimizer.minimize(self.model.loss)

        return self.model