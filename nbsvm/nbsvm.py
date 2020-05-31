# https://www.kaggle.com/jhoward/nb-svm-strong-linear-
import os
import gc
import sys
import time
import pickle
import hashlib
import traceback
import logging
import logging.config
from scipy import sparse
from datetime import datetime
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.sparse import isspmatrix_csc, isspmatrix_csr
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer


class NBFeaturer(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def preprocess_x(self, x, r):
        return x.multiply(r)

    def transform(self, x):
        x_nb = self.preprocess_x(x, self._r)
        return x_nb

    def fit(self, x, y):
        assert set(y) == set([0, 1]), "NBFeaturer only support binary label (0 and 1)"
        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) / self.pr(x, 0, y)))
        return self

    def pr(self, x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + self.alpha) / ((y == y_i).sum() + self.alpha)

class NBSVM(ClassifierMixin):
    save_obj = ['class_num', 'nb_ratios', 'models']
    logistic_reg_params = {'solver':'lbfgs'}

    def __init__(self, logistic_reg_params={}, n_jobs=1, logger=None):
        self.logistic_reg_params.update(logistic_reg_params)
        self.n_jobs = n_jobs
        self.md5 = None
        self.logger = logger and logger or logging.getLogger('nbsvm_logger')
        self.classes_ = None
        self.nb_ratios = []
        self.models = []
        self.class_num = None


    def __repr__(self):
        return f'<NBSVM: classes={self.classes_}>'

    def _chcek_class_num(self, fit_y):
        assert len(set(fit_y)) == self.class_num, \
                f'Fitting Target is not compatible with class num={self.class_num}'

    def _preprocess_x(self, x):
        if not (isspmatrix_csc(x) or isspmatrix_csr(x)):
            x = sparse.csr_matrix(x)
        return x

    def fit(self, x, y):
        x = self._preprocess_x(x)
            
        self.logger.info("Shape of Training Set: {}".format(x.shape))

        y = np.array(y) 
        assert y.ndim == 1, f"Not support multi-label"
        
        self.classes_ = np.array(sorted(set(y)))

        start = time.time()
        for step, (model, nb_ratio) in enumerate(
            Parallel(n_jobs=self.n_jobs)(delayed(
            self._get_lr_mdl)(x, (y == idx).astype(int)) for idx in self.classes_)):

            self.models.append(model)
            self.nb_ratios.append(nb_ratio)

        self.logger.info(f'Training comsuming time: {time.time() - start:2f} s')
        self.logger.info(f"Train set acc: {self.evaluate(x, y)}")
        return self

    def _get_lr_mdl(self, x, y):
        factor = sparse.csr_matrix(np.log(self._pr(1, y, x) / self._pr(0, y, x)))
        m = LogisticRegression(**self.logistic_reg_params)
        x_nb = x.multiply(factor)
        return m.fit(x_nb, y), factor

    def _pr(self, y_i, y, x):
        p = x[y==y_i].sum(0)
        return (p + 1) / ((y == y_i).sum() + 1)

    def predict(self, x):
        '''
        x: list of string
        '''
        preds = self.predict_proba(x)
        return self.classes_[preds.argmax(axis=1)]

    def predict_proba(self, x):
        x = self._preprocess_x(x)

        preds = np.zeros((x.shape[0], self.classes_.shape[0]))
        for idx in range(self.classes_.shape[0]):
            model, factor = self.models[idx], self.nb_ratios[idx]
            preds[:, idx] = model.predict_proba(x.multiply(factor))[:, 1]
        return preds

    def evaluate(self, x, y_true):
        y_pred = self.predict(x)
        acc = accuracy_score(y_pred, y_true)
        return acc

    def save(self, models_path='./model/nbsvm.model'):
        model_package = {
            obj_name: self.__dict__[obj_name]
            for obj_name in self.save_obj
        }
        model_dir = os.path.dirname(models_path)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        joblib.dump(model_package, models_path)
        md5 = self.get_md5(models_path)
        self.logger.info(f"save models to {models_path}, md5={md5}")

    def load(self, models_path):
        isSucceeded = False
        try:
            self.md5 = self.get_md5(models_path)
            self.logger.info(f"Load models from :{models_path}, md5={self.md5}")
            model_package= joblib.load(models_path)
            self.__dict__.update(model_package)
        except:
            self.logger.error(f'Fail to load model:{traceback.format_exc()}')
        return isSucceeded

    def get_md5(self, file_path):
        res = None
        try:
            md5 = hashlib.md5()
            md5.update(open(file_path, 'rb').read())
            res = md5.hexdigest()
        except:
            self.logger.warning('Get md5 fail: {}'.format(traceback.format_exc()))
        return res