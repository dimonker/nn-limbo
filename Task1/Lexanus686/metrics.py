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

    truepos = (prediction * ground_truth).sum()
    falsepos = (prediction[(ground_truth != 0)[0]]).sum()
    falseneg = (ground_truth[(prediction != 0)[0]]).sum()
    trueneg = ground_truth.shape[0] - truepos - falseneg - falsepos

    accuracy = (truepos + trueneg) / ground_truth.shape[0]

    precision = 0 if (truepos + falsepos) == 0 else truepos / (truepos + falsepos)

    recall = 0 if (truepos + falseneg) == 0 else truepos / (truepos + falseneg)

    f1 = 0 if (recall + precision) == 0 else 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1, accuracy
    
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
    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]: accuracy += 1
    return accuracy / ground_truth.shape[0]
