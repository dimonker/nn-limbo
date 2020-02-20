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

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    num_samples = prediction.shape[0]
    for i in range(num_samples):
        tp += prediction[i] and ground_truth[i]
        fn += not prediction[i] and ground_truth[i]
        fp += prediction[i] and not ground_truth[i]
        tn += not prediction[i] and not ground_truth[i]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)

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

    true_prediction = 0
    num_samples = prediction.shape[0]
    for i in range(num_samples):
        if prediction[i] == ground_truth[i]:
            true_prediction += 1
    print(true_prediction)
    return true_prediction / num_samples
