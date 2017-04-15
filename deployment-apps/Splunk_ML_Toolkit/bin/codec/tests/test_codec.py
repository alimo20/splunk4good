#!/usr/bin/env python

import json
from pytest import raises

from codec import *


class TestCodec():
    def ndarray_util(self, a):
        j = json.dumps(a, cls=MLSPLEncoder)
        o = json.loads(j, cls=MLSPLDecoder)
        assert (a == o).all()

        # Alternative syntax.
        j = MLSPLEncoder().encode(a)
        o = MLSPLDecoder().decode(j)
        assert (a == o).all()

    def clusterer_util(self, clusterer):
        import numpy as np
        m = clusterer()
        X = np.random.randn(30, 5)
        m.fit(X)
        j = json.dumps(m, cls=MLSPLEncoder)
        o = json.loads(j, cls=MLSPLDecoder)
        assert m.predict(X[0]) == o.predict(X[0])

    def transformer_util(self, transformer):
        import numpy as np
        m = transformer()
        X = np.random.randn(30, 5)
        m.fit(X)
        j = json.dumps(m, cls=MLSPLEncoder)
        o = json.loads(j, cls=MLSPLDecoder)
        assert (m.transform(X[0]) == o.transform(X[0])).all()

    def estimator_util(self, estimator, X, y):
        m = estimator()
        m.fit(X, y)
        j = json.dumps(m, cls=MLSPLEncoder)
        o = json.loads(j, cls=MLSPLDecoder)
        assert m.predict(X[0]) == o.predict(X[0])

    def selector_util(self, estimator, X, y):
        m = estimator(mode='k_best', param=1)
        m.fit(X, y)
        j = json.dumps(m, cls=MLSPLEncoder)
        o = json.loads(j, cls=MLSPLDecoder)
        print o.transform(X[0])
        assert m.transform(X[0]) == o.transform(X[0])

    def regressor_util(self, estimator):
        self.estimator_util(estimator, [[1, 2], [3, 4]], [5, 6])

    def classifier_util(self, estimator):
        self.estimator_util(estimator, [[1, 2], [3, 4]], ['red', 'blue'])
        self.estimator_util(estimator, [[1, 2], [3, 4]], ['world', u'hello\u0300'])
        self.estimator_util(estimator, [[1, 2], [3, 4]], [u'hello\u0300', 'world'])
        self.estimator_util(estimator, [[1, 2], [3, 4]], [5, 6])

    def pod_util(self, obj):
        j = json.dumps(obj, cls=MLSPLEncoder)
        o = json.loads(j, cls=MLSPLDecoder)
        assert cmp(obj, o) == 0

    def test_ndarray(self):
        import numpy as np

        print self
        print self.ndarray_util

        self.ndarray_util(np.array([[1, 2], [3, 4]]))
        self.ndarray_util(np.array([1.5, 2.5, 3.5]))
        self.ndarray_util(np.array([True, False, True, False]))
        self.ndarray_util(np.array(['hello', 'world']))
        self.ndarray_util(np.array([u'hello\u0300', u'w\u0300rld']))

    def test_PCA(self):
        from algos.PCA import PCA
        PCA.register_codecs()
        from sklearn.decomposition import PCA
        self.transformer_util(PCA)

    def test_KMeans(self):
        from algos.KMeans import KMeans
        KMeans.register_codecs()
        from sklearn.cluster import KMeans
        self.clusterer_util(KMeans)

    def test_Birch(self):
        from algos.Birch import Birch
        Birch.register_codecs()
        from sklearn.cluster import Birch as clusterer
        self.clusterer_util(clusterer)

    def test_DecisionTreeClassifier(self):
        from algos.DecisionTreeClassifier import DecisionTreeClassifier
        DecisionTreeClassifier.register_codecs()
        from sklearn.tree import DecisionTreeClassifier
        self.classifier_util(DecisionTreeClassifier)

    def test_RandomForestClassifier(self):
        from algos.RandomForestClassifier import RandomForestClassifier
        RandomForestClassifier.register_codecs()
        from sklearn.ensemble import RandomForestClassifier
        self.classifier_util(RandomForestClassifier)

    def test_GaussianNB(self):
        from algos.GaussianNB import GaussianNB
        GaussianNB.register_codecs()
        from sklearn.naive_bayes import GaussianNB
        self.classifier_util(GaussianNB)

    def test_BernoulliNB(self):
        from algos.BernoulliNB import BernoulliNB
        BernoulliNB.register_codecs()
        from sklearn.naive_bayes import BernoulliNB
        self.classifier_util(BernoulliNB)

    def test_DecisionTreeRegressor(self):
        from algos.DecisionTreeRegressor import DecisionTreeRegressor
        DecisionTreeRegressor.register_codecs()
        from sklearn.tree import DecisionTreeRegressor
        self.regressor_util(DecisionTreeRegressor)

    def test_RandomForestRegressor(self):
        from algos.RandomForestRegressor import RandomForestRegressor
        RandomForestRegressor.register_codecs()
        from sklearn.ensemble import RandomForestRegressor
        self.regressor_util(RandomForestRegressor)

    def test_Lasso(self):
        from algos.Lasso import Lasso
        Lasso.register_codecs()
        from sklearn.linear_model import Lasso
        self.regressor_util(Lasso)

    def test_ElasticNet(self):
        from algos.ElasticNet import ElasticNet
        ElasticNet.register_codecs()
        from sklearn.linear_model import ElasticNet
        self.regressor_util(ElasticNet)

    def test_Ridge(self):
        from algos.Ridge import Ridge
        Ridge.register_codecs()
        from sklearn.linear_model import Ridge
        self.regressor_util(Ridge)

    def test_LinearRegression(self):
        from algos.LinearRegression import LinearRegression
        LinearRegression.register_codecs()
        from sklearn.linear_model import LinearRegression
        self.regressor_util(LinearRegression)

    def test_LogisticRegression(self):
        from algos.LogisticRegression import LogisticRegression
        LogisticRegression.register_codecs()
        from sklearn.linear_model import LogisticRegression
        self.regressor_util(LogisticRegression)

    def test_GenericUnivariateSelect(self):
        from algos.FieldSelector import FieldSelector
        FieldSelector.register_codecs()
        from sklearn.feature_selection import GenericUnivariateSelect
        self.selector_util(GenericUnivariateSelect, [[1, 2], [3, 4]], ['red', 'blue'])
        self.selector_util(GenericUnivariateSelect, [[1, 2], [3, 4]], ['world', u'hello\u0300'])
        self.selector_util(GenericUnivariateSelect, [[1, 2], [3, 4]], [u'hello\u0300', 'world'])
        self.selector_util(GenericUnivariateSelect, [[1, 2], [3, 4]], [5, 6])

    def test_SVC(self):
        from algos.SVM import SVM
        SVM.register_codecs()
        from sklearn.svm import SVC
        self.regressor_util(SVC)

    def test_OneClassSVM(self):
        from algos.OneClassSVM import OneClassSVM
        OneClassSVM.register_codecs()
        from sklearn.svm.classes import OneClassSVM
        self.clusterer_util(OneClassSVM)

    def test_TFIDF(self):
        from algos.TFIDF import TFIDF
        TFIDF.register_codecs()
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = ['the quick brown fox jumps over the lazy dog']
        m = TfidfVectorizer()
        m.fit(X)
        j = json.dumps(m, cls=MLSPLEncoder)
        o = json.loads(j, cls=MLSPLDecoder)
        print m.transform(X)
        assert (m.transform(X) == o.transform(X)).toarray().all()

    def test_StandardScaler(self):
        from algos.StandardScaler import StandardScaler
        StandardScaler.register_codecs()
        from sklearn.preprocessing import StandardScaler
        self.transformer_util(StandardScaler)

    def test_KernelRidge(self):
        from algos.KernelRidge import KernelRidge
        KernelRidge.register_codecs()
        from sklearn.kernel_ridge import KernelRidge
        self.regressor_util(KernelRidge)

    def test_pod(self):
        self.pod_util({"foo": 1})
        self.pod_util({u"foo": 1})
        self.pod_util({"foo": u"bar"})
        self.pod_util({u"foo": u"bar"})
        self.pod_util([1, 2, 3])
        self.pod_util([1.5, 2.5, 3.5])
        self.pod_util([{"foo": "bar"}])
        self.pod_util({"foo": ["bar", 5]})

        # JSON objects always have a string key
        with raises(AssertionError):
            self.pod_util({1: "foo"})

    def test_np_builtin(self):
        import numpy as np
        self.pod_util(np.int64(42))
        self.pod_util(np.int32(42))
        self.pod_util(np.int16(42))
        self.pod_util(np.int8(42))
        self.pod_util(np.uint64(42))
        self.pod_util(np.uint32(42))
        self.pod_util(np.uint16(42))
        self.pod_util(np.uint8(42))
        self.pod_util(np.float16(42))
        self.pod_util(np.float32(42))
        self.pod_util(np.float64(42))
        # self.pod_util(np.float128(42))
        self.pod_util(np.complex64(42))
        self.pod_util(np.complex128(42))
        # self.pod_util(np.complex256(42))

    def test_pandas(self):
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({'a': range(10), 'b': np.arange(10) * 0.1})

        j = json.dumps(df, cls=MLSPLEncoder)
        o = json.loads(j, cls=MLSPLDecoder)

        assert (df == o).all().all()
