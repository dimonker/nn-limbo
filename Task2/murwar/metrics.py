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
    curr_accuracy = 0
    num_test = prediction.shape[0]
    for j in range(num_test):
        if prediction[j] == ground_truth[j]: 
            curr_accuracy += 1    
    accuracy = curr_accuracy / num_test
    
    return accuracy

