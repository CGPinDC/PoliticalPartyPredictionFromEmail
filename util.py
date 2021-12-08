import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            f1_score, cohen_kappa_score, RocCurveDisplay,\
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
    '''
    #determine multiclass or binary model
    classes = len(labels)
       
    #predictions
    preds = model.predict(X_te)
    preds_proba = model.predict_proba(X_te)

    #adjustments for binary vs. multi-class evluations
    if classes == 2:
        average = 'binary'
        preds_proba = model.predict_proba(X_te)[:,1]
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

    #plot evaluation graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

    ConfusionMatrixDisplay.from_predictions(y_te, preds, display_labels=labels, ax=ax1, normalize='true')
    ax1.set_title('Confusion Matrix')

    #plot binary roc-auc
    if classes == 2: 
        fpr, tpr, thresh = roc_curve(y_te, preds_proba) 
        ax2.plot(fpr, tpr, linestyle='--',color='orange', label='ROC curve (area = %0.2f)'%roc_auc)
    
    #plot mulit-class roc
    else:
        for i in range(classes):    
            fpr[i], tpr[i], thresh[i] = roc_curve(y_te, preds_proba[:,i], pos_label=i) 
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

def feature_importance_df(feature_names, feature_importances, number, coef=True):
    '''
    Creates dataframe of feature importances. 

    Inputs:
        vectorizer: the fitted model, or step in the piplene, that has attribute of .get_feature_names()
        classifier: the fitted model, or step in the piplene, that has attribute of .feature_importances() or .coef_
        coef: boolean, True if the model returns coefficients, False if not.
        number: number of top words to return
    
    Output:
        Dataframe of features, sorted by most important
    '''

    #Finding the feature names
    features = feature_names.get_feature_names_out()

    if coef==True:
        importances = feature_importances.coef_.flatten()
    
    if coef==False: 
        importances = feature_importances.feature_importances_

    #zip features and importances values to form a dataframe
    feature_importances = pd.DataFrame(zip(features, importances), columns=['features', 'values'])

    #find absolute value of values
    feature_importances['abs_val'] = feature_importances['values'].apply(lambda x: abs(x))

    #sort values
    feature_importances = feature_importances.sort_values(by='abs_val', ascending=False)[:number]

    return feature_importances

def model_evaluation_withoutproba(model, X_tr, y_tr, X_te, y_te, labels): 
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
    '''
    #determine multiclass or binary model
    classes = len(labels)
       
    #predictions
    preds = model.predict(X_te)
    
    #score model from metrics
    avg_cros_val = cross_val_score(model, X_tr, y_tr, cv=3).mean()
    accuracy = accuracy_score(y_te, preds)
    precision = precision_score(y_te, preds, average='binary')
    recall = recall_score(y_te, preds, average='binary')
    f1 = f1_score(y_te, preds, average='binary')
    cohen_kappa = cohen_kappa_score(y_te, preds)
    roc_auc = roc_auc_score(y_te, preds)
    
    #store scores
    scores = [[model, avg_cros_val, accuracy, precision, recall, f1, cohen_kappa, roc_auc]]
    scores_df = pd.DataFrame(scores, columns=['model', 'cross_val', 'accuracy','precision','recall','f1','cohen-kappa', 'roc-auc'])

    #plot evaluation graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

    ConfusionMatrixDisplay.from_predictions(y_te, preds, display_labels=labels, ax=ax1, normalize='true')
    ax1.set_title('Confusion Matrix')

    RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax2)
    
    return scores_df