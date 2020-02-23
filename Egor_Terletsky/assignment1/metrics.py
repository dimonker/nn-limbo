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
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    n_samples = ground_truth.shape[0]
    
    tp = float((prediction * ground_truth).sum())
    fp = float((prediction[np.nonzero(ground_truth == 0)[0]]).sum())
    fn = float((ground_truth[np.nonzero(prediction == 0)[0]]).sum())
    tn = float(n_samples - tp - fn - fp)

    accuracy = (tp + tn) / n_samples

    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if (recall + precision) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
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
    counter = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            counter += 1
    acc = counter / len(prediction)
    return acc

