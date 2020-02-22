import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions (retrieved)
    ground_truth, np array of bool (num_samples) - true labels (relevant)

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    # TP - true positive (prediction = 1, ground_truth = 1)
    # TN - true negative (prediction = 0, ground_truth = 0)
    # FP - false positive (prediction = 1, ground_truth = 0)
    # FN - false negative (prediction = 0, ground_truth = 1)
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    TP = np.sum((prediction[i] == True and ground_truth[i] == True) for i in range(prediction.shape[0]))
    TN = np.sum((prediction[i] == False and ground_truth[i] == False) for i in range(prediction.shape[0]))
    FP = np.sum((prediction[i] == True and ground_truth[i] == False) for i in range(prediction.shape[0]))
    FN = np.sum((prediction[i] == False and ground_truth[i] == True) for i in range(prediction.shape[0]))
    
    #print("TP = ", TP)
    #print("TN = ", TN)
    #print("FP = ", FP)
    #print("FN = ", FN)
    if TP + FP != 0:
        precision = TP / (TP + FP) 
    if TP + FN != 0:
        recall = TP / (TP + FN)
    if TP + TN + FP + FN != 0:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    if precision + recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    #print(precision, recall, accuracy, f1)
    #display(prediction)
    #display(ground_truth)
    
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
    TP_or_TN = np.sum((prediction[i] == ground_truth[i]) for i in range(prediction.shape[0]))
    accuracy = TP_or_TN / len(prediction)
    return accuracy
