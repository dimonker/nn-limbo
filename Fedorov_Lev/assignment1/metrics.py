def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = tn = fp = fn = 0

    for p, g in zip(prediction, ground_truth):
        if p:
            if g:
                tp += 1
            else:
                fp += 1
        else:
            if not g:
                tn += 1
            else:
                fn += 1

    precision = tp / (tp + fp) if tp else 0
    recall = tp / (tp + fn) if tp else 0
    accuracy = (tp + tn) / prediction.shape[0] if prediction.shape[0] else 0

    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0
    
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
    return sum([p == g for p, g in zip(prediction, ground_truth)]) / prediction.shape[0] if prediction.shape[0] else 0
