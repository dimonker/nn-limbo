import numpy as np

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

    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    arr = np.zeros((2, 2), np.float32)
    for i in range(prediction.shape[0]):
        if prediction[i] == ground_truth[i] and prediction[i] == True:
            arr[0,0] += 1
        elif prediction[i] == ground_truth[i] and prediction[i] == False:
            arr[1,1] += 1
        elif prediction[i] == 1 and ground_truth[i] == False:
            arr[1,0] += 1
        else:
            arr[0,1] += 1

    precision = arr[0,0] / (arr[0,0] + arr[1,1])
    recall = arr[0,0] / (arr[0,0] + arr[0,1])
    accuracy = (arr[0,0] + arr[1,1]) / (arr[0,0] + arr[0,1] + arr[1,0] + arr[1,1])
    f1 = 2 * precision * recall / (precision + recall)

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

    arr = np.zeros((10,10), np.int)
    for i in range(prediction.shape[0]):
        p = prediction[i]
        t = ground_truth[i]
        if p == t:
            arr[p,p] += 1
        else:
            arr[t,p] += 1

    def get_precisions():
        precisions = []
        for i in range(arr.shape[0]):
            precision = arr[i,i] / np.sum(arr[i])
            precisions.append(precision)
        return np.array(precisions, dtype=np.float32)
    precision = np.mean(get_precisions())

    def get_recalls():
        recalls = []
        recalls_sum = np.apply_along_axis(lambda v: np.sum(v), 0, arr)
        for i in range(arr.shape[0]):
            recall = arr[i,i] / recalls_sum[i]
            recalls.append(recall)
        return np.array(recalls, dtype=np.float32)
    recall = np.mean(get_recalls())
    accuracy = np.sum(arr.diagonal()) / prediction.shape[0]
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, accuracy
