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

    true_p, true_n, false_p, false_n = 0, 0, 0, 0
    
    for i in range(ground_truth.shape[0]):
        if prediction[i] == True:
            if prediction[i] == ground_truth[i]:
                true_p += 1
            else:
                true_p += 1
                
        else:
            if prediction[i] == ground_truth[i]:
                true_n += 1
            else:
                false_n += 1
        
        if prediction.shape[0] != 0:
            accuracy = (true_p + true_n) / prediction.shape[0]
        else:
            accuracy = 0
            
        if (true_p + false_p) != 0:
            precision = true_p / (true_p + false_p)
        else:
            precision = 0
        
        if (true_p + fn) != 0:
            recall = true_p / (true_p + false_n)

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
    acc = 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            acc += 1

    if prediction.shape[0] != 0:
        accuracy = acc / prediction.shape[0]
    else:
        accuracy = 0
    return accuracy
