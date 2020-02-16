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

    tp_count, tn_count, fp_count, fn_count = 0, 0, 0, 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] == True:
            if prediction[i] == ground_truth[i]:
                tp_count += 1
            else:
                fp_count += 1
        else:
            if prediction[i] == ground_truth[i]:
                tn_count += 1
            else:
                fn_count += 1
    
    if prediction.shape[0] != 0:
        accuracy = (tp_count + tn_count) / prediction.shape[0]
    else:
        accuracy = 0
    if (tp_count + fp_count) != 0:
        precision = tp_count / (tp_count + fp_count)
    else:
        precision = 0
    if (tp_count + fn_count) != 0:
        recall = tp_count / (tp_count + fn_count)
    else:
        recall = 0
    if (recall + precision) != 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0
            
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
    correct = 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            correct += 1

    if prediction.shape[0] != 0:
        accuracy = correct / prediction.shape[0]
    else:
        accuracy = 0

    return accuracy
