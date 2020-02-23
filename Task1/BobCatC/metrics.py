import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    true_positive = np.sum(np.logical_and(prediction, ground_truth))
    true_negative = np.sum(np.logical_not(np.logical_or(prediction, ground_truth)))
    false_negative = np.sum(np.logical_and(np.logical_not(prediction), ground_truth))
    false_positive = np.sum(np.logical_and(prediction, np.logical_not(ground_truth)))

    precision = None
    recall = None
    accuracy = None
    f1 = None

    if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)

    if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)

    if true_positive + true_negative + false_positive + false_negative > 0:
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    if precision and recall and precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return np.sum(prediction == ground_truth) / ground_truth.shape[0]
