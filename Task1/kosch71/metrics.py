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

    tp, tn, fp, fn = 0, 0, 0, 0
    
    for i in range(ground_truth.shape[0]):
        if prediction[i] == True:
            if prediction[i] == ground_truth[i]:
                tp += 1
            else:
                fp += 1
                
        else:
            if prediction[i] == ground_truth[i]:
                tn += 1
            else:
                fn += 1
        
        if prediction.shape[0] != 0:
            accuracy = (tp + tn) / prediction.shape[0]
        else:
            accuracy = 0

        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if (tp + fn) != 0:
            recall = tp / (tp + fn)

        else:
            recall = 0

        if (recall + precision) != 0:
            f1 = 2 * recall * precision / (recall + precision)
        else:
            f1 = 0
    
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
    tp = 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            tp += 1

    if prediction.shape[0] != 0:
        accuracy = tp / prediction.shape[0]
    else:
        accuracy = 0
    return accuracy
