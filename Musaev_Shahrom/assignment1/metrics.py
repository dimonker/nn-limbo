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
    
    true_pos, true_neg, folse_pos, folse_neg = 0, 0, 0, 0
    
    for i in range(ground_truth.shape[0]):

        if prediction[i] == True:

            if prediction[i] == ground_truth[i]:

                true_pos += 1

            else:

                folse_pos += 1

        else:

            if prediction[i] == ground_truth[i]:

                true_neg += 1

            else:

                folse_neg += 1

    

    if prediction.shape[0] != 0:

        accuracy = (true_pos + true_neg) / prediction.shape[0]

    else:

        accuracy = 0

    if (true_pos + folse_pos) != 0:

        precision = true_pos / (true_pos + folse_pos)

    else:

        precision = 0

    if (true_pos + folse_neg) != 0:

        recall = true_pos / (true_pos + folse_neg)

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
    # TODO: Implement computing accuracy
    true = 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            true += 1

    if prediction.shape[0] != 0:
        accuracy = true / prediction.shape[0]
    else:
        accuracy = 0

    return accuracy
