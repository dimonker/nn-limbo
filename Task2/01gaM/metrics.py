def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy

    num_samples = prediction.shape[0]
    
    if num_samples == 0:
        return 0
    
    TP_TN = 0
    for i in range(num_samples):
        if (ground_truth[i] == prediction[i]):
            TP_TN += 1
            
    accuracy = TP_TN / num_samples
    return accuracy
