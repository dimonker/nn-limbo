def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    len_predict = len(prediction)

    for i in range(len_predict):
        if prediction[i] == ground_truth[i]:
            if prediction[i]:
                tp += 1
            else:
                tn += 1
        elif prediction[i]:
            fp += 1
        else:
            fn += 1

    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    accuracy = (tp + tn) / len_predict

    if (precision + recall) != 0:
        f1 = (2 * recall * precision) / (precision + recall)
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
    # TODO: Implement computing accuracy

    accuracy = 0
    len_predict = len(prediction)

    for i in range(len_predict):
        if prediction[i] == ground_truth[i]:
            accuracy += 1

    return accuracy / len_predict

