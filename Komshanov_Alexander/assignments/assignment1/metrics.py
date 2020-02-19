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

    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(prediction)):
        if prediction[i]:
            if prediction[i] == ground_truth[i]: tp += 1
            else: fp += 1
        else:
            if prediction[i] == ground_truth[i]: tn += 1 
            else: fn += 1
    
    accuracy = (tp + tn) / len(prediction) if len(prediction) != 0 else 0
    precision = tp / (tp + fp) if (tp+fp) != 0 else 0
    recall = tp / (tp + fn) if (tp+fn) != 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) != 0 else 0
    
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

    
    return sum([1 if pred==tr else 0 for pred, tr in zip(prediction, ground_truth)])/len(prediction)
