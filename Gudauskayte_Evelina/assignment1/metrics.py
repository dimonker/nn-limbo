def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    # print(ground_truth)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]: 
            if prediction[i]: tp+=1
            else: tn += 1
        elif prediction[i]: fp+=1
        else: fn+=1 
    
    precision = tp/(tp+fp) if (tp+fp)!=0 else 0
    recall = tp/(tp+fn) if (tp+fp)!=0 else 0
    accuracy = (tp+tn)/len(prediction)

    f1 = 2*recall*precision/(precision+recall) if (precision+recall)!=0 else 0

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
    acc = 0
    for i in range(len(prediction)):
        if prediction[i]==ground_truth[i]: acc+=1
    return acc/len(prediction)
