from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            plot_confusion_matrix, f1_score, cohen_kappa_score
from sklearn.model_selection import cross_val_score
import pandas as pd

#evaluation functions
def model_evaluation(model, X_tr, y_tr, X_te, y_te, labels): 
    '''
    Evaluates model and returns accuracy, precision, recall, F1, roc-auc scores, 
    plots a confusion matrix and ROC-AUC graph.
    
    Inputs: 
    model: model fitted to X_tr
    X_tr: Training variable
    y_tr: Training target
    X_te: Test variable
    y_te: Test target
    labels: target labeles for confusion matrix
    '''

    preds = model.predict(X_te)

    #score model from metrics
    avg_cros_val = cross_val_score(model, X_tr, y_tr, cv=3).mean()
    accuracy = accuracy_score(y_te, preds)
    precision = precision_score(y_te, preds, average='micro')
    recall = recall_score(y_te, preds, average='micro')
    f1 = f1_score(y_te, preds, average='micro')
    cohen_kappa = cohen_kappa_score(y_te, preds)

    #store scores
    scores = [[model, avg_cros_val, accuracy, precision, recall, f1, cohen_kappa]]
    scores_df = pd.DataFrame(scores, columns=['model', 'cross_val', 'accuracy','precision','recall','f1','cohen-kappa'])
    
    #plot confusion matrix
    plot_confusion_matrix(model, X_te, y_te, display_labels=labels, normalize='true')

    return scores_df