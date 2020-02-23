def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(ground_truth.shape[0]):
        if prediction[i]:
            if ground_truth[i]:
                tp += 1
            else:
                fp += 1
        else:
            if not ground_truth[i]:
                tn += 1
            else:
                fn += 1

    accuracy = (tp + tn) / prediction.shape[0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * recall * precision / (recall + precision)

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
    correctly_predicted = 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            correctly_predicted += 1

    accuracy = correctly_predicted / prediction.shape[0]
    return accuracy
