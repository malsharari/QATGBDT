from sklearn import metrics

import numpy as np
import catboost
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from Functions.precisioninteger import update_to_PTQ, quantized_data, dequantize
from scipy.special import expit, logit




def evaluate_model_performance(model_path, X_test, y_test, S_threshold, Z_threshold, S_leaf, Z_leaf, N, Flag, Save_data=False):
    '''
    # Usage
    # model_integer_post = 'path_to_model'
    # accuracy, avg_auc = evaluate_model_performance(model_integer_post, X_test, y_test, S_threshold, Z_threshold, S_leaf, Z_leaf, N, Flag)
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Average AUC: {avg_auc:.4f}")

'''
    
    # Load the CatBoostClassifier model
    cls = catboost.CatBoostClassifier()
    cls.load_model(model_path, format="json")

    # Quantize test data
    Xq_test = quantized_data(X_test, S_threshold, Z_threshold, N)
    if Save_data:
        np.savetxt(f'data/X_test_{N}.dat', Xq_test, fmt='%s')
    # Predict the classes
    y_pred_classes = cls.predict(Xq_test, prediction_type='RawFormulaVal')
    
    y_pred_prob_try = expit(dequantize(y_pred_classes, S_leaf, Z_leaf,N))


    y_pred = (y_pred_prob_try > 0.8).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # roc_auc = []

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_try)
    roc_auc=(auc(fpr, tpr))
    # Calculate average AUC
    avg_auc = roc_auc
    #print(avg_auc)
    
    # Optimal threshold is the one where the sum of sensitivity (TPR) and specificity (1 - FPR) is maximized
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    # print(f'optimal_threshold = {optimal_threshold:.4f}')
    # print(f'accuracy = {accuracy:.4f}')
    # Use the optimal threshold for classification
    y_pred_optimal = (y_pred_prob_try >= optimal_threshold).astype(int)

    # Calculate accuracy with the optimal threshold
    accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
    print(f'optimal_threshold = {optimal_threshold:.4f}')
    print(f'Optimal Accuracy = {accuracy_optimal:.4f}')
    
    
    return accuracy_optimal, avg_auc,roc_auc

def evaluate_model_performance_without_quantization(model_path, X_test, y_test):
    '''
    # Usage
    # accuracy, avg_auc = evaluate_model_performance_without_quantization(model_cat, X_test, y_test)
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Average AUC: {avg_auc:.4f}")
    '''
    
    # Load the CatBoostClassifier model
    cls = catboost.CatBoostClassifier()
    cls.load_model(model_path, format="json")

    # Predict the classes
    y_pred_classes = cls.predict(X_test, prediction_type='RawFormulaVal')
    y_pred_prob_try = expit(y_pred_classes)

    indices_of_highest = (y_pred_prob_try >= 0.8).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, indices_of_highest)
    
    # Calculate AUC for each class
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_try)
    roc_auc=(auc(fpr, tpr))
    # Calculate average AUC
    avg_auc = roc_auc
    #print(avg_auc)
    
    # Optimal threshold is the one where the sum of sensitivity (TPR) and specificity (1 - FPR) is maximized
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Use the optimal threshold for classification
    y_pred_optimal = (y_pred_prob_try >= optimal_threshold).astype(int)

    # Calculate accuracy with the optimal threshold
    accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
    print(f'optimal_threshold = {optimal_threshold:.4f}')
    print(f'Optimal Accuracy = {accuracy_optimal:.4f}')
    
    return accuracy_optimal, avg_auc,roc_auc

