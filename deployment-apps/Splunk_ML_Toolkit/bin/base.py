#!/usr/bin/env python
import json
from distutils.version import StrictVersion

import pandas as pd
import numpy as np

import cexc
import conf

from codec import MLSPLEncoder, MLSPLDecoder
from util.param_util import convert_params, is_truthy
from util.base_util import match_field_globs


class BaseMixin(object):
    def encode(self):
        if StrictVersion(np.version.version) >= StrictVersion('1.10.0'):
            return MLSPLEncoder().encode(self)
        else:
            raise RuntimeError('Python for Scientific Computing version 1.1 or later is required to save models.')
            # import pickle
            # return pickle.dumps(self)

    @staticmethod
    def decode(payload):
        if StrictVersion(np.version.version) >= StrictVersion('1.10.0'):
            return MLSPLDecoder().decode(payload)
        else:
            raise RuntimeError('Python for Scientific Computing version 1.1 or later is required to load models.')
            # import pickle
            # return pickle.dumps(self)

    @staticmethod
    def filter_non_numeric(df, max_values=100):
        """Filter out non-numeric columns with too many unique factors."""
        dropcols = []
        scols = list(df.dtypes[df.dtypes == 'object'].index)

        # TODO: Profile this loop.
        for scol in scols:
            if df[scol].nunique() > max_values:
                dropcols.append(scol)

        if len(dropcols) > 0:
            cexc.messages.warn('Dropping field(s) with too many distinct values: %s', ', '.join(dropcols))
            df.drop(dropcols, inplace=True, axis=1)

        if len(df.columns) == 0:
            raise RuntimeError('No valid fields to fit or apply model to.')

    @staticmethod
    def assert_field_present(df, field):
        """Make sure field is present."""
        if field not in df:
            raise Exception('Field "%s" not present.' % field)

    @staticmethod
    def assert_not_too_many_values(df, field, max_values=100):
        """Limit the number of categories for classifiers."""
        n = df[field].nunique()
        if n > max_values:
            raise Exception('Field "%s" has too many distinct values: %d (max %d)' % (
                field, n, max_values))
        nans = df[field].isnull()
        df[field] = df[field].astype(str)
        df.ix[nans, field] = np.nan

    @staticmethod
    def drop_unused_fields(df, requested_fields):
        """Drop fields the user didn't ask for."""
        dropcols = set(df.columns).difference(requested_fields)
        df.drop(dropcols, inplace=True, axis=1)

    @staticmethod
    def warn_on_missing_fields(df, requested_fields):
        """Raise user-visible warning for missing fields."""
        missing_columns = set(requested_fields).difference(df.columns)
        if len(missing_columns) > 0:
            cexc.messages.warn('Missing field(s): %s', ', '.join(missing_columns))

    @staticmethod
    def split_response_out(df, response_variable):
        return df.drop(response_variable, axis=1), df[response_variable]

    @staticmethod
    def drop_na_columns(df):
        start_columns = df.columns
        df.dropna(axis=1, how='all', inplace=True)
        end_columns = df.columns
        dropped_columns = set(start_columns).difference(end_columns)
        if len(dropped_columns) > 0:
            cexc.messages.warn('Dropped field(s) with all null values: %s', ', '.join(dropped_columns))

    @staticmethod
    def drop_na_rows(df):
        nans = df.isnull().any(axis=1).values
        df.dropna(axis=0, how='any', inplace=True)
        return nans

    @staticmethod
    def assert_any_fields(df):
        if len(df.columns) == 0:
            raise RuntimeError('No valid fields to fit or apply model to.')

    @staticmethod
    def assert_any_rows(df):
        if len(df) == 0:
            raise RuntimeError('No valid events; check for null or non-numeric values in numeric fields')

    @staticmethod
    def sort_fields(df):
        df.sort_index(inplace=True, axis=1)
        return list(df.columns)

    @staticmethod
    def fill_missing_fields(df, requested_fields):
        """Fill missing fields with 0's."""
        missing_fields = set(requested_fields).difference(set(df.columns))
        if len(missing_fields) > 0:
            cexc.logger.debug('Filling missing fields(s): %s', ', '.join(missing_fields))
        for col in missing_fields:
            df[col] = 0

    def handle_options(self, options):
        if len(options.get('variables', [])) == 0 or len(options.get('explanatory_variables', [])) > 0:
            raise RuntimeError('Syntax error: expected "<field> ..."')

        self.variables = options['variables']

    @staticmethod
    def apply_in_chunks(df, func, n=1000, options=None, output_name=None):
        def bechunk(df_, n_):
            return [df_[i:i + n_] for i in xrange(0, len(df_), n_)]

        dfs = [func(x, options, output_name) for x in bechunk(df, n)]

        df = pd.DataFrame()
        df = df.append(dfs).copy()
        df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def find_unseen_ind_X(X, columns):
        """Find the X axis indices (row numbers) where new values are present."""
        new_cat_cols = np.setdiff1d(X.columns, columns)
        if len(new_cat_cols) == 0:
            return (new_cat_cols, None)
        new_cat_ind = np.where(X[new_cat_cols].any(axis=1).values == True)[0]
        return (new_cat_ind, new_cat_cols)

    @staticmethod
    def skip_unseen_X(X, y, row_ind, columns):
        """Remove rows with unseen categorical value(s) from X"""
        X.drop(row_ind, axis=0, inplace=True)
        X.drop(columns, axis=1, inplace=True)
        if y is not None:
            y.drop(row_ind, axis=0, inplace=True)
        cexc.messages.warn(
            'Some events containing unseen categorical explanatory values have been skipped while updating the model.')

    @staticmethod
    def skip_unseen_y(X, y, classes):
        """Remove unseen categorical values from y. Also remove rows which corresponds to removed y values from X"""
        new_cat_cols_y = np.setdiff1d(np.unique(y), classes)
        if len(new_cat_cols_y) == 0:
            return
        new_cat_ind = [i for i, x in enumerate(y) if x in new_cat_cols_y]
        X.drop(new_cat_ind, axis=0, inplace=True)
        y.drop(new_cat_ind, axis=0, inplace=True)
        cexc.messages.warn(
            'Some events containing unseen categorical target values have been skipped while updating the model.')

    @staticmethod
    def rename_columns(p, options):
        # Rename output columns
        if options is not None:
            if len(p.columns) == 1 and 'output_name' in options:
                p.columns = [options['output_name']]
            elif len(p.columns) > 1 and 'output_name' in options:
                p.columns = [options['output_name'] + '_%d' % (i + 1)
                             for i in range(len(p.columns))]

    def get_relevant_variables(self):
        """Return the union of explanatory & normal variables.

        Returns:
            list: list of relevant fields
        """
        # TODO: consistent variable naming in base.py
        relevant_variables = []
        if 'variables' in self.__dict__:
            relevant_variables.extend(self.variables)

        if 'explanatory_variables' in self.__dict__:
            relevant_variables.extend(self.explanatory_variables)

        # just a string
        if 'response_variable' in self.__dict__:
            relevant_variables.append(self.response_variable)
        return relevant_variables

    def handle_categorical(self, X, y, handle_new_cat, columns, classes=None):
        action_unseen = {'stop', 'default', 'skip'}
        try:
            assert (handle_new_cat in action_unseen)
        except AssertionError:
            raise Exception('Invalid value for "unseen_value": %s' % handle_new_cat)
        # Fill in empty columns if input has less categorical values than the ones the existing model was trained with
        self.fill_missing_fields(X, columns)
        if handle_new_cat == 'skip':
            # remove rows containing new categorical values in X
            new_cat_ind_X, new_cat_cols = self.find_unseen_ind_X(X, columns)
            if len(new_cat_ind_X) > 0:
                self.skip_unseen_X(X, y, new_cat_ind_X, new_cat_cols)
            # remove rows containing new categorical values in y
            if classes is not None:
                self.skip_unseen_y(X, y, classes)

        elif handle_new_cat == 'default':
            # set columns that corresponds to new categorical value(s) to 0 for applicable rows
            new_cat_ind_X, new_cat_cols = self.find_unseen_ind_X(X, columns)
            if len(new_cat_ind_X) > 0:
                X.drop(new_cat_cols, axis=1, inplace=True)
                cexc.messages.warn(
                    'Columns correspond to unseen categorical explanatory variable value(s): %s are omitted' % new_cat_cols)
            # remove rows containing new categorical values in y
            if classes is not None:
                self.skip_unseen_y(X, y, classes)

        else:
            # stops when encountering rows containing new categorical values (X or y)
            new_col_in_X = np.setdiff1d(X.columns, columns)
            if len(new_col_in_X) > 0:
                raise RuntimeError(
                    'New categorical value for explanatory variables in training data: %s' % new_col_in_X)
            if classes is not None:
                new_class_in_y = np.setdiff1d(y, classes)
                if len(new_class_in_y) > 0:
                    raise RuntimeError('New target values in training data: %s' % new_class_in_y)


class EstimatorMixin(BaseMixin):
    def handle_options(self, options):
        if len(options.get('variables', [])) != 1 or len(options.get('explanatory_variables', [])) == 0:
            raise RuntimeError('Syntax error: expected "<target> FROM <field> ..."')

        self.response_variable = options['variables'][0]
        self.explanatory_variables = options['explanatory_variables']

    def preprocess_fit(self, X):
        self.assert_field_present(X, self.response_variable)
        if 'is_classifier' in dir(self):
            self.assert_not_too_many_values(X, self.response_variable)
        self.explanatory_variables = match_field_globs(X.columns, self.explanatory_variables)
        self.drop_unused_fields(X, [self.response_variable] + list(self.explanatory_variables))
        self.drop_na_columns(X)
        self.drop_na_rows(X)
        self.assert_field_present(X, self.response_variable)
        self.warn_on_missing_fields(X, self.explanatory_variables)
        X, y = self.split_response_out(X, self.response_variable)
        self.filter_non_numeric(X)
        X = pd.get_dummies(X, prefix_sep='=', sparse=True)
        self.assert_any_fields(X)
        self.assert_any_rows(X)
        columns = self.sort_fields(X)
        return (X, y, columns)

    def fit(self, X):
        X, y, self.columns = self.preprocess_fit(X)
        self.estimator.fit(X.values, y.values)
        if 'is_classifier' in dir(self):
            self.classes = np.unique(y.values)

    def preprocess_predict(self, X):
        self.drop_unused_fields(X, self.explanatory_variables)
        self.filter_non_numeric(X)
        X = pd.get_dummies(X, prefix_sep='=', sparse=True)
        self.drop_na_columns(X)
        self.drop_unused_fields(X, self.columns)
        self.assert_any_fields(X)
        self.fill_missing_fields(X, self.columns)
        self.sort_fields(X)
        assert set(X.columns) == set(self.columns), 'Internal error: column mismatch'
        return X

    def predict(self, X, options=None, output_name=None):
        X = self.preprocess_predict(X)

        # Allocate output DataFrame
        output_name = 'predicted(%s)' % self.response_variable
        output = pd.DataFrame({output_name: np.empty(len(X))})
        output[output_name] = np.nan

        nans = self.drop_na_rows(X)
        y_hat = self.estimator.predict(X.values)

        output.ix[~nans, output_name] = y_hat
        self.rename_columns(output, options)

        return output


class TreeEstimatorMixin(EstimatorMixin):
    def summary(self, options=None):
        if options:
            out_params = convert_params(
                options.get('params', {}),
                ints=["limit"],
                bools=["json"]
            )
            if "json" in out_params:
                return_json = out_params["json"]
            if "limit" in out_params:
                depth_limit = out_params["limit"]

        if 'return_json' not in locals():
            return_json = is_truthy(conf.get_mlspl_prop('summary_return_json', self.__class__.__name__, 'f'))
        if 'depth_limit' not in locals():
            depth_limit = int(conf.get_mlspl_prop('summary_depth_limit', self.__class__.__name__, -1))
        if depth_limit <= 0:
            raise ValueError('Limit = %d. Value for limit should be greater than 0.' % depth_limit)

        root = 0
        depth = 0
        if return_json:
            output_data = [json.dumps(self.summaryDict(depth_limit, root, depth), sort_keys=True)]
        else:
            output_data = self.summaryStr(depth_limit, root, depth)
        return pd.DataFrame({'Decision Tree Summary': output_data})

    def summaryStr(self, depth_limit, root, depth):
        t = self.estimator.tree_
        features = self.columns

        left_child = t.children_left[root]
        right_child = t.children_right[root]

        n_nodes = t.n_node_samples[root]
        impurity = t.impurity[root]
        if 'is_classifier' in dir(self):
            classes = self.estimator.classes_
            value = t.value[root][0]
            class_value = classes[value.argmax()]
            value_str = "class:%s  " % class_value
        else:
            value_str = "value:%.3f  " % t.value[root][0][0]
        indent = '----' * depth + ' '
        if left_child > 0 or right_child > 0:
            feature = features[t.feature[root]]
            if feature in self.explanatory_variables:
                feature_val = t.threshold[root]
                split_str = "split:%s<=%.3f" % (feature, feature_val)
            else:
                split_str = "split:%s" % feature
        else:
            split_str = "split:N/A - Leaf node"
        output_str = "|--" + indent + "count:%d  %s  %simpurity:%.3f" % (
            n_nodes, split_str, value_str, impurity)
        output = [output_str]

        if depth_limit >= 1:
            depth += 1
            depth_limit -= 1
            if left_child > 0:
                output.extend(self.summaryStr(depth_limit, left_child, depth))
            if right_child > 0:
                output.extend(self.summaryStr(depth_limit, right_child, depth))
        return output

    def summaryDict(self, depth_limit, root, depth):
        t = self.estimator.tree_
        features = self.columns

        left_child = t.children_left[root]
        right_child = t.children_right[root]

        output = {}
        output["count"] = int(t.n_node_samples[root])

        if 'is_classifier' in dir(self):
            classes = self.estimator.classes_
            value = t.value[root][0]
            output["class"] = classes[value.argmax()]
        else:
            output["value"] = round(t.value[root][0][0], 3)

        if left_child > 0 or right_child > 0:
            feature = features[t.feature[root]]
            if feature in self.explanatory_variables:
                feature_val = t.threshold[root]
                output["split"] = "%s<=%.3f" % (feature, feature_val)
            else:
                output["split"] = "split:%s" % feature
        else:
            output["split"] = "split:N/A - Leaf node"

        output["impurity"] = round(t.impurity[root], 3)

        if depth_limit >= 1:
            depth += 1
            depth_limit -= 1
            if left_child > 0:
                output["left child"] = self.summaryDict(depth_limit, left_child, depth)
            if right_child > 0:
                output["right child"] = self.summaryDict(depth_limit, right_child, depth)
        return output


class ClustererMixin(BaseMixin):
    def preprocess_fit(self, X):
        self.variables = match_field_globs(X.columns, self.variables)
        self.drop_unused_fields(X, self.variables)
        self.drop_na_columns(X)
        self.drop_na_rows(X)
        self.warn_on_missing_fields(X, self.variables)
        self.filter_non_numeric(X)
        X = pd.get_dummies(X, prefix_sep='=', sparse=True)
        self.assert_any_fields(X)
        self.assert_any_rows(X)
        columns = self.sort_fields(X)
        return (X, columns)

    def fit(self, X):
        X, self.columns = self.preprocess_fit(X)
        self.estimator.fit(X.values)

    def preprocess_predict(self, X):
        self.drop_unused_fields(X, self.variables)
        self.drop_na_columns(X)
        self.warn_on_missing_fields(X, self.variables)
        self.filter_non_numeric(X)
        X = pd.get_dummies(X, prefix_sep='=', sparse=True)
        self.drop_unused_fields(X, self.columns)
        self.assert_any_fields(X)
        self.fill_missing_fields(X, self.columns)
        self.sort_fields(X)
        assert set(X.columns) == set(self.columns), 'Internal error: column mismatch'
        return X

    def predict(self, X, options=None, output_name=None):
        X = self.preprocess_predict(X)

        # Allocate output DataFrame
        output_name = 'cluster'
        output = pd.DataFrame({output_name: np.empty(len(X))})
        output[output_name] = np.nan

        nans = self.drop_na_rows(X)
        y_hat = self.estimator.predict(X.values)
        y_hat = ['' if np.isnan(v) else str('%.0f' % v) for v in y_hat]

        output.ix[~nans, output_name] = y_hat
        self.rename_columns(output, options)

        return output


class SelectorMixin(EstimatorMixin):
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

        length = len(X)
        nans = self.drop_na_rows(X)

        # Transform.
        y_hat = self.estimator.transform(X.values)
        mask = self.estimator.get_support()
        columns_select = np.array(self.columns)[mask]
        width = len(columns_select)

        if width == 0:
            cexc.messages.warn('No fields pass the current configuration. Consider changing your parameters.')

        output_names = ['fs_%s' % x for x in columns_select]
        output = pd.DataFrame(np.zeros((length, width)), columns=output_names)
        output.ix[:, output_names] = np.nan
        output.ix[~nans, output_names] = y_hat
        self.rename_columns(output, options)

        return output


class TransformerMixin(BaseMixin):
    def fit(self, X):
        self.assert_field_present(X, self.variable)
        self.drop_unused_fields(X, [self.variable])
        self.drop_na_rows(X)
        self.assert_any_fields(X)
        self.assert_any_rows(X)

        if type(X[self.variable][X[self.variable].axes[0][0]]) != str:
            raise RuntimeError('Invalid type: "%s" is of %s. String expected.' % (
                self.variable, type(X[self.variable][X[self.variable].axes[0][0]])))

        # Fit the model.
        self.estimator.fit(X[self.variable])

    def predict(self, X, options=None, output_name=None):
        self.variable = self.variable.encode('utf-8')
        self.assert_field_present(X, self.variable)
        self.drop_unused_fields(X, [self.variable])
        self.assert_any_fields(X)

        length = len(X)
        nans = self.drop_na_rows(X)

        y_hat = self.estimator.transform(X[self.variable])

        # Allocate output DataFrame
        width = y_hat.shape[1]
        columns = [self.variable + '_tfidf_' + str(index) + '_' + word
                   for (index, word) in enumerate(self.estimator.get_feature_names())]

        output = pd.DataFrame(np.zeros((length, width)), columns=columns)
        output.ix[:, columns] = np.nan
        output.ix[~nans, columns] = y_hat.toarray()
        self.rename_columns(output, options)

        return output


class TSMixin(BaseMixin):
    def handle_options(self, options):
        if len(options.get('variables', [])) == 0 \
                or len(options.get('variables', [])) > 2 \
                or len(options.get('explanatory_variables', [])) > 0:
            raise RuntimeError('Syntax error: expected " _time, <field>."')
        self.variables = options['variables']
        if len(self.variables) == 2:
            if '_time' in self.variables:
                self.time_series = self.variables[1 - self.variables.index('_time')]
            else:
                raise RuntimeError('Syntax error: if two fields given, one should be _time.')
        elif len(self.variables) == 1:
            self.time_series = options['variables'][0]
