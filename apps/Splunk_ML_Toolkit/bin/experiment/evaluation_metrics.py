from copy import deepcopy
from StringIO import StringIO

from numpy import sqrt
import pandas as pd

import cexc
from scorings.classification import (
    PrecisionRecallFscoreSupportScoring,
    AccuracyScoring,
)

from scorings.regression import (
    R2Scoring,
    MeanSquaredErrorScoring,
)

logger = cexc.get_logger('LogexperimentStatistics')
messages = cexc.get_messages_logger()


def get_statistics_metadata(experiment, body):
    """ Load experiment metadata and input data.

    Args:
        experiment (dict): fetched experiment from REST
        body (csv str): csv string data payload from CEXC

    Returns:
        exp_metadata (dict or None): experiment metadata, if was accessible.
            - experiment type (str)
            - ground-truth field name (str)
            - predicted field name (str)
            - data containing ground-truth/predicted data (pd.dataframe)
         If not accessible, returns None.
    """
    debugging_msg_prefix = 'Statistics could not be computed -- {}.'

    exp_type = experiment.get('type')
    # Get 'main' search stage, as it contains the desired target/feature variables
    search_stages = experiment.get('searchStages', [])
    if len(search_stages) == 0:
        logger.debug(debugging_msg_prefix.format('experiment has no searchStages'))
        return None

    # We assume that each 'stage' is a python dictionary (search stages is a list of dictionaries)
    main_search_stages = [stage for stage in search_stages if stage.get('role') == 'main']
    if len(main_search_stages) != 1:
        logger.debug(debugging_msg_prefix.format("require exactly one 'main' search stages."))
        return None

    # Get the actual and predicted fields
    main_search_stage = main_search_stages[0]
    actual_field = main_search_stage.get('targetVariable')
    if actual_field is None:
        logger.debug(debugging_msg_prefix.format('No target variable was found in main search stage.'))
        return None

    predicted_field = 'predicted({})'.format(actual_field)  # TODO: Don't hard-code this dependency
    sio = StringIO(body)
    sio.seek(0)
    try:  # Ensure that the data contains the actual and predicted fields
        applied_data = pd.read_csv(sio, usecols=[str(actual_field), str(predicted_field)])
    except ValueError as e:
        msg = "data must contain the actual ({}) and predicted ({}) fields"
        logger.debug(msg.format(actual_field, predicted_field))
        logger.debug(e)
        return None

    # Return the metadata required for scoring
    exp_metadata = {'type': exp_type, 'actual': actual_field, 'predicted': predicted_field, 'data': applied_data}
    return exp_metadata


def get_scoring_results(scoring_class, scoring_opts, data):
    """ Call the scoring function and compute the results.

    Args:
        scoring_class (scoring obj): object capable of computing score
        scoring_opts (dict): options passed to scoring object
        data (pd.dataframe): data containing predicted and ground-truth data

    Returns:
        results (pd.dataframe): results of applying scoring function
    """
    scorer = scoring_class(scoring_opts)
    results = scorer.score(data, scoring_opts)
    return results


def compute_pcf_statistics(exp_metadata, ndigits=2):
    """ Compute the statistics for a PCF experiment.

    Computes precision, recall, f1 and accuracy scores.

    -For precision, recall and f1, a 'weighted' averaging
        scheme is used.

    Args:
        exp_metadata (dict): Metadata of an experiment, containing:
            - experiment type (str)
            - ground-truth field name (str)
            - predicted field name (str)
            - data containing ground-truth/predicted data (pd.dataframe)
        ndigits (int): Number of digits to keep after the decimal place

    Returns:
        statistics_dict (dict): Dictionary of computed statistics
    """
    opts_skeleton = {
        'scoring_name': '',
        'params': {},
        'variables': [],
        'a_variables': [exp_metadata['actual']],
        'b_variables': [exp_metadata['predicted']],
    }

    try:
        # Get the precision, recall and f1-score
        opts = deepcopy(opts_skeleton)
        opts['scoring_name'] = 'precision_recall_fscore_support'
        opts['params']['average'] = 'weighted'
        p_r_f_s = get_scoring_results(PrecisionRecallFscoreSupportScoring, opts, exp_metadata['data'])

        # Get the accuracy score
        opts = deepcopy(opts_skeleton)
        opts['scoring_name'] = 'accuracy_score'
        accuracy = get_scoring_results(AccuracyScoring, opts, exp_metadata['data'])

        # Create statistics dictionary; keys must be compatible with experiment schema
        statistics_dict = {
            'stats_precision': round(float(p_r_f_s['precision'][0]), ndigits),
            'stats_recall': round(float(p_r_f_s['recall'][0]), ndigits),
            'stats_f1': round(float(p_r_f_s['fbeta_score'][0]), ndigits),
            'stats_accuracy': round(float(accuracy['accuracy_score'][0]), ndigits),
        }
    except Exception as e:
        msg = 'PCF statistics could not be computed -- failed to evaluate scoring metrics on experiment metadata.'
        logger.debug(msg)
        logger.debug(e)
        statistics_dict = {}

    return statistics_dict


def compute_pnf_statistics(exp_metadata, ndigits_rmse=2, ndigits_r2=4):
    """ Compute the statistics for a PNF experiment.

    - Computes r^2 (coefficient of determination) and
        RMSE (root mean squared error)

    Args:
        exp_metadata (dict): Metadata of an experiment, containing:
            - experiment type (str)
            - ground-truth field name (str)
            - predicted field name (str)
            - data containing ground-truth/predicted data (pd.dataframe)
        ndigits_rmse (int): Number of digits to keep after the decimal place
            for root-mean-squared-error metric
        ndigits_r2 (int): Number of digits to keep after the decimal place
            for r^2 metric
    Returns:
        statistics_dict (dict): Dictionary of computed statistics
    """
    opts_skeleton = {
        'scoring_name': '',
        'params': {},
        'a_variables': [exp_metadata['actual']],
        'b_variables': [exp_metadata['predicted']],
        'variables': []
    }

    try:
        # Get the r^2 statistic
        opts = deepcopy(opts_skeleton)
        opts['scoring_name'] = 'r2_score'
        r2 = get_scoring_results(R2Scoring, opts, exp_metadata['data'])

        # Get the RMSE statistic
        opts = deepcopy(opts_skeleton)
        opts['scoring_name'] = 'mean_squared_error'
        mse = get_scoring_results(MeanSquaredErrorScoring, opts, exp_metadata['data'])

        # Create statistics dictionary; keys must be compatible with experiment schema
        statistics_dict = {
            # For r^2, we round to 4 decimal places for historical reasons
            'stats_rSquared': round(float(r2['r2_score'][0]), ndigits_r2),
            # Take the square root to obtain RMSE
            'stats_RMSE': round(sqrt(float(mse['mean_squared_error'][0])), ndigits_rmse),
        }

    except Exception as e:
        msg = 'PNF statistics could not be computed -- failed to evaluate scoring metrics on experiment metadata.'
        logger.debug(msg)
        logger.debug(e)
        statistics_dict = {}

    return statistics_dict


def _merge_exp_metadata(exp_metadata_list):
    """
    Merge the list of experiment metadata into a single entry

    Args:
        exp_metadata_list (list): list of experiment metadata dicts
            (each metadata dict is the output of get_statistics_metadata())

    Returns:
        exp_metadata (dict): Single experiment metadata with merged 'data' field
    """
    data_fieldname = 'data'
    data_list = [exp_metadata.get(data_fieldname) for exp_metadata in exp_metadata_list]

    # Fix the index after merging the data field
    data_df = pd.concat(data_list)
    data_df.index = range(len(data_df))

    # All metadata entries should have the same fields and values except for the data.
    exp_metadata = exp_metadata_list[-1]
    exp_metadata[data_fieldname] = data_df
    return exp_metadata


def compute_statistics(exp_metadata_list):
    """ Compute the statistics for the given prediction problem.

    Accepted prediction problems are:
        - Predict categorical fields (PCF)

    Args:
        exp_metadata_list (list): list of experiment metadata dicts
            (each metadata dict is the output of get_statistics_metadata())

    Returns:
        statistics (dict): PCF dictionary of statistics results.
            Empty dictionary returned if results could not be calculated.
    """
    # If any chunk is None, don't compute statistics
    if not all(exp_metadata_list):
        return {}

    exp_metadata = _merge_exp_metadata(exp_metadata_list)

    exp_type = exp_metadata.get('type')
    if exp_type == 'predict_categorical_fields':
        statistics_dict = compute_pcf_statistics(exp_metadata)
    elif exp_type == 'predict_numeric_fields':
        statistics_dict = compute_pnf_statistics(exp_metadata)
    elif exp_type == 'cluster_numeric_events':
        # cluster_numeric_events are not implemented yet
        statistics_dict = {}
    else:
        logger.debug("Cannot compute experiment statistics on experiment of type: {}.".format(exp_type))
        statistics_dict = {}
    return statistics_dict
