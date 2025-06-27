# Copyright contributors to the TSFM project
#
"""Code adapted from https://github.com/TheDatumOrg/TSB-UAD"""

import numpy as np
from numpy import percentile
from sklearn.metrics import precision_score
from sklearn.utils import check_consistent_length, column_or_1d


def _pprint(params, offset=0, printer=repr):
    # noinspection PyPep8
    """Pretty print the dictionary 'params'

    See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    and sklearn/base.py for more information.

    :param params: The dictionary to pretty print
    :type params: dict

    :param offset: The offset in characters to add at the begin of each line.
    :type offset: int

    :param printer: The function to convert entries to strings, typically
        the builtin str or repr
    :type printer: callable

    :return: None
    """

    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = []
    this_line_length = offset
    line_sep = ",\n" + (1 + offset // 2) * " "
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = "%s=%s" % (k, str(v))
        else:
            # use repr of the rest
            this_repr = "%s=%s" % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + "..." + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or "\n" in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(", ")
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = "".join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = "\n".join(l.rstrip(" ") for l in lines.split("\n"))
    return lines


def precision_n_scores(y, y_pred, n=None):
    """Utility function to calculate precision @ rank n.
    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).
    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.
    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.
    Returns
    -------
    precision_at_rank_n : float
        Precision at rank n score.
    """

    # turn raw prediction decision scores into binary labels
    y_pred = get_label_n(y, y_pred, n)

    # enforce formats of y and labels_
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    return precision_score(y, y_pred)


def get_label_n(y, y_pred, n=None):
    """Function to turn raw outlier scores into binary labels by assign 1
    to top n outlier scores.
    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).
    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.
    n : int, optional (default=None)
        The number of outliers. if not defined, infer using ground truth.
    Returns
    -------
    labels : numpy array of shape (n_samples,)
        binary labels 0: normal points and 1: outliers
    Examples
    --------
    >>> from pyod.utils.utility import get_label_n
    >>> y = [0, 1, 1, 0, 0]
    >>> y_pred = [0.1, 0.5, 0.3, 0.2, 0.7]
    >>> get_label_n(y, y_pred)
    array([0, 1, 0, 0, 1])
    """

    # enforce formats of inputs
    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)

    check_consistent_length(y, y_pred)
    y_len = len(y)  # the length of targets

    # calculate the percentage of outliers
    if n is not None:
        outliers_fraction = n / y_len
    else:
        outliers_fraction = np.count_nonzero(y) / y_len

    threshold = percentile(y_pred, 100 * (1 - outliers_fraction))
    y_pred = (y_pred > threshold).astype("int")

    return y_pred
