def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = sum([x[1] == ground_truth[x[0]] for x in enumerate(prediction == True)])
    fn = sum([x[1] == ground_truth[x[0]] for x in enumerate(prediction == False)])
    tn = sum([x[1] != ground_truth[x[0]] for x in enumerate(prediction == True)])
    fp = sum([x[1] != ground_truth[x[0]] for x in enumerate(prediction == False)])
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if len(prediction) != 0:
        accuracy = sum(prediction == ground_truth)/len(prediction)
    else:
        accuracy = 0
    if precision + recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
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
    return sum(prediction == ground_truth)/len(prediction)
