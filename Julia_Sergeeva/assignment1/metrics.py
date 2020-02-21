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
    precision = 0
    recall = 0
    f1 = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(prediction.shape[0]):
        if prediction[i]:
            if ground_truth[i]:
                tp += 1
            else:
                fp += 1
        else:
            if ground_truth[i]:
                fn += 1
            else:
                tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if tp + fp != 0:
        precision = tp / (tp + fp)
    if tp + fn != 0:
        recall = tp / (tp + fn)
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)

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
    confusion = (10, 10)
    confusion = np.zeros(confusion, dtype=int)
    for i in range(prediction.shape[0]):
        confusion[ground_truth[i]][prediction[i]] += 1

    accuracy = 0

    for i in range(10):
        accuracy += confusion[i][i]

    accuracy /= prediction.shape[0]

    return accuracy
