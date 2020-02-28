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
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for i in range(prediction.shape[0]):
        if prediction[i] == True and ground_truth[i] == True:
            true_positives += 1
        if prediction[i] == True and ground_truth[i] == False:
            false_positives += 1
        if prediction[i] == False and ground_truth[i] == True:
            false_negatives += 1
        if prediction[i] == False and ground_truth[i] == False:
            true_negatives += 1
    
    if (true_positives + false_positives) != 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
    
    if (true_positives + false_negatives) != 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0
    
    if (precision + recall) != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    if (true_positives + true_negatives + false_positives + false_negatives) != 0:
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    else:
        accuracy = 0
    
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
    good_count = 0
    for i in range(prediction.shape[0]):
#         print(prediction[i], ground_truth[i])
        if prediction[i] == ground_truth[i]:
            good_count += 1

    return good_count / prediction.shape[0]
