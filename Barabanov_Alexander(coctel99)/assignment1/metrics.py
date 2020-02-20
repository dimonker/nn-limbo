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

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(prediction.shape[0]):
        if prediction[i]:
            if prediction[i] == ground_truth[i]:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if prediction[i] == ground_truth[i]:
                true_neg += 1
            else:
                false_neg += 1

    if true_pos + false_pos == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)

    if true_pos + false_neg == 0:
        recall = 0
    else:
        recall = true_pos / (true_pos + false_neg)

    if recall + precision == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)

    if prediction.shape[0] == 0:
        accuracy = 0
    else:
        accuracy = (true_pos + true_neg) / prediction.shape[0]

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
    correct = 0
    # TODO: Implement computing accuracy
    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            correct += 1

    if prediction.shape[0] == 0:
        accuracy = 0
    else:
        accuracy = correct / prediction.shape[0]

    return accuracy
