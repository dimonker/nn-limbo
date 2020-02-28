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
    accuracy = 0
    num_samples = prediction.shape[0]
    
    for i in range(num_samples):
        if prediction[i] == ground_truth[i]:
            accuracy += 1
            
    accuracy /= num_samples

    return accuracy
