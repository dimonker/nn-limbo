def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    correct = 0

    for i in range(ground_truth.shape[0]):
        if prediction[i] == ground_truth[i]:
            correct += 1

    accuracy = correct / prediction.shape[0] if prediction.shape[0] != 0 else 0
    return accuracy
