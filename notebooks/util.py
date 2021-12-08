import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, cohen_kappa_score,\
                            roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

#evaluation functions
def model_evaluation(model, X_tr, y_tr, X_te, y_te, labels): 
    '''
    Evaluates model and returns scores cross-val-score, accuracy, precision, recall, f1, and cohen-kappa,  
    then plots a confusion matrix and ROC-AUC graph. 
    
    'code from https://inblog.in/AUC-ROC-score-and-curve-in-multiclass-classification-problems-2ja4jOHb2X'
    
    Inputs: 
        model: model fitted to X_tr
        X_tr: Training variable
        y_tr: Training target
        X_te: Test variable
        y_te: Test target
        labels: target labeles for confusion matrix
        df: data frame used to keep track of scores
    '''
    #determine multiclass or binary model
    classes = len(labels)
       
    #predictions
    preds = model.predict(X_te)
    preds_proba = model.predict_proba(X_te)

    #adjustments for binary vs. multi-class evluations
    if classes == 2:
        average = None
        roc_auc = roc_auc_score(y_te, preds_proba)
    else: 
        average = 'macro'
        roc_auc = roc_auc_score(y_te, preds_proba, multi_class='ovo', average='weighted')

    #score model from metrics
    avg_cros_val = cross_val_score(model, X_tr, y_tr, cv=3).mean()
    accuracy = accuracy_score(y_te, preds)
    precision = precision_score(y_te, preds, average=average)
    recall = recall_score(y_te, preds, average=average)
    f1 = f1_score(y_te, preds, average=average)
    cohen_kappa = cohen_kappa_score(y_te, preds)
    
    #store scores
    scores = [[model, avg_cros_val, accuracy, precision, recall, f1, cohen_kappa, roc_auc]]
    scores_df = pd.DataFrame(scores, columns=['model', 'cross_val', 'accuracy','precision','recall','f1','cohen-kappa', 'roc-auc'])

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}

    for i in range(classes):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_te, preds_proba[:,i], pos_label=i)

    #plot evaluation graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

    ConfusionMatrixDisplay.from_predictions(y_te, preds, display_labels=labels, ax=ax1)
    ax1.set_title('Confusion Matrix')

    for i in range(classes):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_te, preds_proba[:,i], pos_label=i)

        #plot binary roc-auc
    if classes == 2: 
        ax2.plot(fpr[0], tpr[0], linestyle='--',color='orange', label=str('Conservative'))
        ax2.plot(fpr[1], tpr[1], linestyle='--',color='green', label=str('Liberal'))
    
    #plot mulit-class roc
    else: 
        ax2.plot(fpr[0], tpr[0], linestyle='--',color='orange', label=str('Center'))
        ax2.plot(fpr[1], tpr[1], linestyle='--',color='green', label=str('Conservative'))
        ax2.plot(fpr[2], tpr[2], linestyle='--',color='blue', label=str('Liberal'))

    # plotting
    ax2.plot([0,1], [0,1], linestyle='--', color='grey')    
    ax2.set_title('ROC curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive rate')
    ax2.legend(loc='best')

    plt.show();

    return scores_df