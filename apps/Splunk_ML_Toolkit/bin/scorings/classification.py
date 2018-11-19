#!/usr/bin/env python

import pandas as pd
import numpy as np

import cexc
from base_scoring import (
    BaseScoring,
    ClassificationScoringMixin,
    ROCMixin,
)
from util.param_util import convert_params
from util.scoring_util import (
    check_pos_label_and_average_against_data,
    initial_check_pos_label_and_average,
    add_default_params,
    get_and_check_fields_two_1d_arrays,
)

messages = cexc.get_messages_logger()


class AccuracyScoring(ClassificationScoringMixin, BaseScoring):
    """ Implements sklearn.metrics.accuracy_score """

    @staticmethod
    def convert_param_types(params):
        out_params = convert_params(
            params,
            bools=['normalize']
        )
        _meta_params = {}
        return out_params, _meta_params


class PrecisionScoring(ClassificationScoringMixin, BaseScoring):
    """ Implements sklearn.metrics.precision_score """

    @staticmethod
    def convert_param_types(params):
        out_params = convert_params(
            params,
            strs=['average', 'pos_label']
        )
        out_params = add_default_params(out_params, {'average': 'binary'})
        out_params = initial_check_pos_label_and_average(out_params)
        # class_variable_headers is True when average=None
        _meta_params = {'class_variable_headers': True} if out_params['average'] is None else {}
        out_params = add_default_params(out_params, {'pos_label': '1'})  # add positive label after checking average
        return out_params, _meta_params

    def check_params_with_data(self, actual_df, predicted_df):
        check_pos_label_and_average_against_data(actual_df, predicted_df, self.params)


class RecallScoring(ClassificationScoringMixin, BaseScoring):
    """Implements sklearn.metrics.recall_score """

    @staticmethod
    def convert_param_types(params):
        out_params = convert_params(
            params,
            strs=['average', 'pos_label']
        )
        out_params = add_default_params(out_params, {'average': 'binary'})
        out_params = initial_check_pos_label_and_average(out_params)
        # class_variable_headers is True when average=None
        _meta_params = {'class_variable_headers': True} if out_params['average'] is None else {}
        out_params = add_default_params(out_params, {'pos_label': '1'})  # add positive label after checking average
        return out_params, _meta_params

    def check_params_with_data(self, actual_df, predicted_df):
        check_pos_label_and_average_against_data(actual_df, predicted_df, self.params)


class F1Scoring(ClassificationScoringMixin, BaseScoring):
    """ Implements sklearn.metrics.f1_score """

    @staticmethod
    def convert_param_types(params):
        out_params = convert_params(
            params,
            strs=['average', 'pos_label']
        )
        out_params = add_default_params(out_params, {'average': 'binary'})
        out_params = initial_check_pos_label_and_average(out_params)
        # class_variable_headers is True when average=None
        _meta_params = {'class_variable_headers': True} if out_params['average'] is None else {}
        out_params = add_default_params(out_params, {'pos_label': '1'})  # add positive label after checking average
        return out_params, _meta_params

    def check_params_with_data(self, actual_df, predicted_df):
        check_pos_label_and_average_against_data(actual_df, predicted_df, self.params)


class PrecisionRecallFscoreSupportScoring(ClassificationScoringMixin, BaseScoring):
    """ Implements sklearn.metrics.precision_recall_fscore_support

    - note that multi-field comparisons is only enabled when average != None
    """

    @staticmethod
    def convert_param_types(params):
        out_params = convert_params(
            params,
            strs=['pos_label', 'average'],
            floats=['beta']
        )
        out_params = add_default_params(out_params, {'average': 'None'})
        out_params = initial_check_pos_label_and_average(out_params)
        # class_variable_headers is True when average=None
        _meta_params = {'class_variable_headers': True} if out_params['average'] is None else {}
        out_params = add_default_params(out_params, {'pos_label': '1'})  # add positive label after checking average

        # For precision_recall_fscore_support, "support" is undefined when average != None; warn appropriately
        average = out_params.get('average', None)
        if average is not None:
            msg = '"support" metric is not defined when average is not None (found average="{}")'
            cexc.messages.warn(msg.format(average))
        return out_params, _meta_params

    def check_params_with_data(self, actual_df, predicted_df):
        # assert that if average=None, arrays are comprised of exactly 1 field
        if self.params.get('average') is None and predicted_df.shape[1] > 1:
            msg = ('Value error: multi-field comparisons not supported when average=None. Single fields must be '
                   'specified as "..| score {} <actual_field> against <predicted_field>".')
            raise RuntimeError(msg.format(self.scoring_name, self.scoring_name))

        check_pos_label_and_average_against_data(actual_df, predicted_df, self.params)

    def create_output(self, scoring_name, results):
        """ Output dataframe differs from parent.

        The output shape of precision_recall_fscore_support depends on the
        average value. If average!=None, the output shape is
        n-comparisons x (4-metrics + 2 field identifiers). If average=None,
        the output is
        actual-class-variable-cardinality x (4-metrics + 2 field identifiers)
        """
        # Labels is populated when average=None. In this case, metrics are computed for each class variable.
        class_variables = self.params.get('labels', None)

        if class_variables is not None:
            # In this case, the dataframe headers are unique class-labels + field identifiers
            results_array = np.vstack(results)  # 4 x n-classes
            row_labels = np.array(['precision', 'recall', 'fbeta_score', 'support']).reshape(-1, 1)  # 4 x 1
            output_array = np.hstack((row_labels, results_array))  # 4 x (n-classes + 1)
            col_labels = ['scored({})'.format(i) for i in class_variables]
            output_df = pd.DataFrame(data=output_array, columns=['Metric'] + col_labels)  # 4 x (n-classes + 1)
        else:
            result_array = np.array(results).reshape(len(results), -1)  # n-comparisons x 4
            col_labels = ['precision', 'recall', 'fbeta_score', 'support']
            output_df = pd.DataFrame(data=result_array, columns=col_labels)  # n-comparisons x 4
            # Add compared-fields information to the output df
            for k, v in self._meta_params['field_identifiers'].iteritems():  # n-comparisons x (4 + 2)
                output_df[k] = v

        return output_df


class ConfusionMatrixScoring(ClassificationScoringMixin, BaseScoring):
    """Implements sklearn.metrics.confusion_matrix"""

    def handle_options(self, options):
        """ Only single-field against single-field comparisons supported. """
        params = options.get('params', {})
        params, _meta_params = self.convert_param_types(params)
        actual_fields, predicted_fields = get_and_check_fields_two_1d_arrays(
            options, self.scoring_name, a_field_alias='actual_field', b_field_alias='predicted_field'
        )
        return params, actual_fields, predicted_fields, _meta_params

    @staticmethod
    def convert_param_types(params):
        out_params = convert_params(params)
        _meta_params = {'class_variable_headers': True}  # Confusion matrix populates rows & cols with class-variables
        return out_params, _meta_params

    def score(self, df, options):
        """ Confusion matrix requires arrays to be reshaped. """
        # Prepare ground-truth and predicted labels
        actual_array, predicted_array = self.prepare_input_data(df, self.actual_fields, self.predicted_fields, options)
        # Get the scoring result
        result = self.scoring_function(actual_array, predicted_array)
        # Create the output df
        df_output = self.create_output(self.scoring_name, result)
        return df_output

    def create_output(self, scoring_name, result):
        """Output dataframe differs from parent.

        The indices of confusion matrix columns/rows should correspond.
        Columns represent predicted results, rows represent ground-truth.
        """
        class_variables = self.params['labels']  # labels = union of predicted & actual classes
        # Predicted (column) and ground-truth (row) labels
        col_labels = ['Label'] + ['predicted({})'.format(i) for i in class_variables]
        row_labels = pd.DataFrame(['actual({})'.format(i) for i in class_variables])
        # Create output df
        result_df = pd.DataFrame(result)
        output_df = pd.concat((row_labels, result_df), axis=1)
        output_df.columns = col_labels
        return output_df


class ROCCurveScoring(ROCMixin, BaseScoring):
    """ Implements sklearn.metrics.roc_curve"""

    def handle_options(self, options):
        """ Only single-field against single-field comparisons supported. """
        params = options.get('params', {})
        params, _meta_params = self.convert_param_types(params)
        actual_fields, predicted_fields = get_and_check_fields_two_1d_arrays(
            options, self.scoring_name, a_field_alias='actual_field', b_field_alias='predicted_field'
        )
        return params, actual_fields, predicted_fields, _meta_params

    @staticmethod
    def convert_param_types(params):
        out_params = convert_params(
            params,
            strs=['pos_label'],
            bools=['drop_intermediate']
        )
        _meta_params = {'pos_label': out_params.pop('pos_label', '1')}
        return out_params, _meta_params

    def create_output(self, scoring_name, result):
        """ Outputs false-positive rate, true-positive rate and thresholds.

        - Note that roc_curve only works on a pair of 1d columns,
            and so 'result' contains exactly 1 element
        """
        fpr, tpr, thresholds = result[0]
        return pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})


class ROCAUCScoring(ROCMixin, BaseScoring):
    """Implements sklearn.metrics.roc_auc_score"""
    pass
