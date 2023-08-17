# stdlib
from typing import List

# third party
import numpy as np
from pydantic import validate_call


@validate_call(config={"arbitrary_types_allowed": True})
def get_label_scores(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """
    Gets label quality scores for each data point.

    Args:
        labels (np.ndarray): True labels
        pred_probs (np.ndarray): The predicted probabilities for each class

    Returns:
        np.ndarray: label quality scores, between 0 and 1, where lower scores indicate labels less likely to be correct.
    """
    return np.array([pred_probs[i, l] for i, l in enumerate(labels)])


@validate_call(config={"arbitrary_types_allowed": True})
def get_conf_thresholds(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> np.ndarray:
    """
    The conf threshold for a class is the mean predicted probability of this class, according to the model, averaged
    against the examples with that class label.

    Args:
        labels (np.ndarray): True labels
        pred_probs (np.ndarray): The predicted probabilities for each class

    Returns:
        np.ndarray: Returns expected (average) "self-confidence" for each class, as an array of shape (K,)
    """

    all_classes = range(pred_probs.shape[1])
    unique_classes = set(labels)
    """
    confident_thresholds[k] is the average of all predicted probabilities for class k, where k is the true label.
    If there are more classes being predicted in pred_probs than in labels, then we set the confident threshold to
    the arbitrary large value of 10e8.
    This will ensure that the label cannot be considered "confident" for this class.
    """
    confident_thresholds = [
        np.mean(pred_probs[:, k][labels == k]) if k in unique_classes else 10e8
        for k in all_classes
    ]
    return np.asarray(confident_thresholds)


@validate_call(config={"arbitrary_types_allowed": True})
def get_conf_learning_error_indices(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> List[int]:
    """
    Calculates the confident joint distribution of true and noisy labels.

    Args:
        labels (np.ndarray): True labels
        pred_probs (np.ndarray): The predicted probabilities for each class

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, list]]: An array representing counts of examples
        for which we are confident about their given and true label
    """

    # The threshold for considering a class prediction as `confident`.
    thresholds = get_conf_thresholds(labels, pred_probs)

    # Create a boolean mask on pred_probs for confident predictions
    conf_above_threshold = pred_probs >= thresholds - 1e-6

    # Create a boolean mask for predictions for examples with the specified number of classes being confidently predicted
    num_confident_bins = conf_above_threshold.sum(axis=1)
    at_least_one_confident = num_confident_bins > 0
    more_than_one_confident = num_confident_bins > 1
    most_conf_pred = pred_probs.argmax(axis=1)
    confident_argmax = conf_above_threshold.argmax(axis=1)
    true_label_guess = np.asarray(
        [
            most_conf_pred[i] if more_than_one_confident[i] else confident_argmax[i]
            for i in range(len(pred_probs))
        ]
    )

    # drop the rows with no confident predictions from labels and confident guesses
    true_labels_confident = true_label_guess[at_least_one_confident]
    labels_confident = labels[at_least_one_confident]

    # Find the indices where the true label and confident guess are not equal
    conf_label_guess_given_labels_mismatch = true_labels_confident != labels_confident

    indices = np.arange(len(labels))[at_least_one_confident][
        conf_label_guess_given_labels_mismatch
    ]

    return indices.tolist()


@validate_call(config={"arbitrary_types_allowed": True})
def num_mislabelled_data_points(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> int:
    """Calculate the number of mis-labeled data points.

    Args:
        labels (np.ndarray): True labels
        pred_probs (np.ndarray): The predicted probabilities for each class

    Returns:
        int: The number of mis-labeled data points.
    """

    conf_learning_error_indices = get_conf_learning_error_indices(
        labels=labels,
        pred_probs=pred_probs,
    )

    # Create a boolean mask for the indices of the mislabelled data points
    mislabelling_mask = np.zeros(len(labels), dtype=bool)
    for idx in conf_learning_error_indices:
        mislabelling_mask[idx] = True
    pred = pred_probs.argmax(axis=1)

    # Remove mislabelled value if given label is the same as model prediction
    for i, pred_label in enumerate(pred):
        if pred_label == labels[i]:
            mislabelling_mask[i] = False

    num_mislabelled = np.sum(mislabelling_mask)

    return num_mislabelled
