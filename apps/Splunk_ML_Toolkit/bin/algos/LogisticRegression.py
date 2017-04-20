#!/usr/bin/env python

from sklearn.linear_model import LogisticRegression as _LogisticRegression
import pandas as pd
from codec import codecs_manager
from base import EstimatorMixin
from util.param_util import convert_params, is_truthy
import numpy as np


class LogisticRegression(EstimatorMixin):
    def __init__(self, options):
        self.handle_options(options)

        out_params = convert_params(
            options.get('params', {}),
            bools=['fit_intercept', 'probabilities'],
        )

        if 'probabilities' in out_params:
            del out_params['probabilities']
        self.estimator = _LogisticRegression(class_weight='auto', **out_params)
        self.is_classifier = True

    def summary(self):
        df = pd.DataFrame()

        n_classes = len(self.estimator.classes_)
        limit = 1 if n_classes == 2 else n_classes

        for i, c in enumerate(self.estimator.classes_[:limit]):
            cdf = pd.DataFrame({'feature': self.columns,
                                'coefficient': self.estimator.coef_[i].ravel()})
            if not isinstance(self.estimator.intercept_, float):
                cdf = cdf.append(
                    pd.DataFrame({'feature': ['_intercept'], 'coefficient': [self.estimator.intercept_[i]]}))
            cdf['class'] = c
            df = df.append(cdf)

        return df

    def predict(self, X, options=None, output_name=None):
        self.drop_unused_fields(X, self.explanatory_variables)
        self.filter_non_numeric(X)
        X = pd.get_dummies(X, prefix_sep='=', sparse=True)
        self.drop_na_columns(X)
        self.drop_unused_fields(X, self.columns)
        self.assert_any_fields(X)
        self.fill_missing_fields(X, self.columns)
        self.sort_fields(X)
        assert set(X.columns) == set(self.columns), 'Internal error: column mismatch'
        # Allocate output DataFrame
        if output_name is None:
            output_name = 'predicted(%s)' % self.response_variable
        output = pd.DataFrame({output_name: np.empty(len(X))})
        output[output_name] = np.nan

        if options is not None:
            out_params = convert_params(
                options.get('params', {}),
                bools=['probabilities', 'fit_intercept'],  # adding fit_intercept also; otherwise will cause error
            )

            if 'probabilities' in out_params:
                probabilities = is_truthy(out_params['probabilities'])
                if probabilities:  # i.e., probabilities=t
                    output_predict = pd.DataFrame({output_name: np.empty(len(X))})
                    output_predict[output_name] = np.nan
                    # need to obtain class names which we will use in the column header for class probabilities
                    class_names = list(self.estimator.classes_)  # class names for header
                    for i in range(0, len(class_names)):
                        class_names[i] = 'probability(' + str(self.response_variable) + '=' + str(class_names[i]) + ')'
                    output_predict_proba = pd.DataFrame(np.empty([len(X), len(class_names)]))
                    output_predict_proba.columns = class_names
                    output_predict_proba.ix[:, :] = np.nan
                    # prediction will start now
                    nans = self.drop_na_rows(X)  # removing rows containing missing values
                    y_hat_predict = self.estimator.predict(X.values)  # predicted response without class probabilities
                    y_hat_proba = self.estimator.predict_proba(X.values)  # predicted class probabilities
                    # filling DataFrames with respective prediction results
                    output_predict.ix[~nans, output_name] = y_hat_predict
                    output_predict_proba.ix[~nans, :] = y_hat_proba

                    self.rename_columns(output_predict, options)
                    output = pd.concat([output_predict, output_predict_proba], axis=1)
                    # return output_predict, output_predict_proba
                    return output

                else:  # i.e., probabilities=f
                    nans = self.drop_na_rows(X)
                    y_hat = self.estimator.predict(X.values)

                    output.ix[~nans, output_name] = y_hat
                    self.rename_columns(output, options)

                    return output

            else:
                nans = self.drop_na_rows(X)
                y_hat = self.estimator.predict(X.values)

                output.ix[~nans, output_name] = y_hat
                self.rename_columns(output, options)

                return output

    @staticmethod
    def register_codecs():
        from codec.codecs import SimpleObjectCodec
        codecs_manager.add_codec('algos.LogisticRegression', 'LogisticRegression', SimpleObjectCodec)
        codecs_manager.add_codec('sklearn.linear_model.logistic', 'LogisticRegression', SimpleObjectCodec)
