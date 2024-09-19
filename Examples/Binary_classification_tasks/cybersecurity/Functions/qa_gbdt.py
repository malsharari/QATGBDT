from catboost import CatBoostClassifier
from Functions.metrics import evaluate_model_performance, evaluate_model_performance_without_quantization
from Functions.precisioninteger import update_combined_precision, quantized_data,dequantize_data,dequantize
import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from scipy.special import expit, logit

def qa_train_model(X_train, y_train, X_validation, y_validation,X_test,y_test,Flag=[ True , True , True , True], N=0, NIQ=3, learning_rate=0.5, depth=3, tree_size=99, Binary=False):
    
    
    global model_name1, model_name2
    
    model1 = CatBoostClassifier(iterations=NIQ, learning_rate=learning_rate, depth=depth)
    model2 = CatBoostClassifier()
    num_iterations = int(tree_size/NIQ)
    itr=NIQ

    
    def get_losses(model):
        evals_result = model.get_evals_result()
        learn_loss = evals_result.get('learn', {}).get('Logloss', [None])
        test_loss = evals_result.get('validation', {}).get('Logloss', [None])
        return learn_loss, test_loss
    
    learn_losses = []
    test_losses = []
    model_name1 = f'model_dummy1_{N}.json'
    model_name2 = f'model_dummy2_{N}.json'
    
    print('Model training steps to intilize S and Z of Float Dataset')
    model1.fit(X_train, y_train, verbose=None)
#     learn_loss, _ = get_losses(model1)
#     test_loss=0
#     if learn_loss is not None and test_loss is not None:
#         learn_losses.append(learn_loss)
#         #test_losses.append(test_loss)
    model1.save_model(model_name1, format="json")


    
    
    
    S_threshold, Z_threshold, S_leaf, Z_leaf =update_combined_precision(model_name1, model_name2,Flag, N, Binary)
    # print(S_threshold,Z_threshold)
    if not Binary:
        X_train_quantized = quantized_data(X_train,S_threshold,Z_threshold, N)

        X_train_quantized = dequantize_data(X_train_quantized,S_threshold,Z_threshold,N)

        X_validation_quantized = quantized_data(X_validation,S_threshold,Z_threshold, N)

        X_validation_quantized = dequantize_data(X_validation,S_threshold,Z_threshold,N) 


    else:
        X_train_quantized = X_train

        X_train_quantized = X_train_quantized

        X_validation_quantized = X_validation

        X_validation_quantized = X_validation   
        
        
        
        
    model1 = CatBoostClassifier(iterations=NIQ, learning_rate=learning_rate, depth=depth)
    model2 = CatBoostClassifier()
    
    
    
    learn_losses = []
    test_losses = []

    
    print('Model training steps on Quantised Dataset')
    model1.fit(X_train_quantized, y_train, verbose=1)
    learn_loss, _ = get_losses(model1)
    test_loss=0
    if learn_loss is not None and test_loss is not None:
        learn_losses.append(learn_loss)
        #test_losses.append(test_loss)
        
    
    model1.save_model(model_name1, format="json")

    
    S_threshold, Z_threshold, S_leaf, Z_leaf =update_combined_precision(model_name1, model_name2,Flag, N,Binary)
    # print(S_threshold,Z_threshold)
    model2.load_model(model_name2, format="json")

    # Keep track of the processed tree indices
    processed_tree_indices = set()

    count=NIQ

    recent_loss_diffs = []  # Store recent loss differences

    master_S_threshold=[]
    master_Z_threshold=[]
    master_S_leaf=[]
    master_Z_leaf=[]

    while int(model2.tree_count_/NIQ) <= num_iterations:
        print(count)

        # Calculate newly added trees indices
        with open(model_name1, "r") as f:
            model_json = json.load(f)
            all_tree_indices = set(range(len(model_json['oblivious_trees'])))
            #print(all_tree_indices)
        
        # Get indices of the trees that have not been processed yet
        new_tree_indices = all_tree_indices - processed_tree_indices
        print('Training Trees num:',new_tree_indices)
        # Update precision only for the newly added trees
        if new_tree_indices:

            if Binary:
                S_threshold, Z_threshold, S_leaf, Z_leaf = update_combined_precision(model_name1, model_name2,Flag, N, Binary)
                
                X_test_quantized = X_test

                
            else:
                S_threshold, Z_threshold, S_leaf, Z_leaf = update_combined_precision(model_name1, model_name2,Flag, N,Binary)
                
                X_train_quantized = quantized_data(X_train,S_threshold,Z_threshold, N)

                X_train_quantized = dequantize_data(X_train_quantized,S_threshold,Z_threshold,N)

                X_validation_quantized = quantized_data(X_validation,S_threshold,Z_threshold, N)

                X_validation_quantized = dequantize_data(X_validation,S_threshold,Z_threshold,N)             
                
                X_test_quantized = quantized_data(X_test,S_threshold,Z_threshold, N)

                X_test_quantized = dequantize_data(X_test_quantized,S_threshold,Z_threshold,N)
        
        # Add the newly processed trees to the set of processed trees
        processed_tree_indices.update(new_tree_indices)

        
        model2.load_model(model_name2, format="json")
        # Load the modified model
        

#         # Predict the classes
#         y_pred_classes = model2.predict(X_test_quantized, prediction_type='RawFormulaVal')

#         y_pred_prob_try = expit(y_pred_classes)

#         fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_try)
        
#         # Optimal threshold is the one where the sum of sensitivity (TPR) and specificity (1 - FPR) is maximized
#         optimal_idx = np.argmax(tpr - fpr)
#         optimal_threshold = thresholds[optimal_idx]
#         roc_auc=(auc(fpr, tpr))
        
        
#         indices_of_highest = (y_pred_prob_try >= optimal_threshold).astype(int)
#         # Calculate accuracy
#         accuracy = accuracy_score(y_test, indices_of_highest)        
#         # Calculate average AUC
#         avg_auc = roc_auc
        
        accuracy, avg_auc,_ = evaluate_model_performance_without_quantization(model_name2, X_test_quantized, y_test)

        print(f"Optimal Test Accuracy: {accuracy:.4f}")
        print(f"Test AUC: {avg_auc:.4f}")
        
        master_S_threshold.append(S_threshold)
        master_Z_threshold.append(Z_threshold)
        master_S_leaf.append(S_leaf)
        master_Z_leaf.append(Z_leaf)        
        
        if count >= tree_size:
            
            break           
        
        count+=itr        
        model1.fit(X_train_quantized, y_train, init_model=model2)
        learn_loss, _ = get_losses(model1)
        # print(learn_loss)
        if learn_loss is not None and test_loss is not None:
            learn_losses.append(learn_loss)
            test_losses.append(test_loss)
            
        # Save model to JSON before processing newly added trees
        model1.save_model(model_name1, format="json")       

    
    count+=itr

    master_S_threshold=np.array((master_S_threshold))
    master_Z_threshold=np.array((master_Z_threshold))
    master_S_leaf=np.array((master_S_leaf))
    master_Z_leaf=np.array((master_Z_leaf))

    return model_name2, model1, model2, learn_losses, test_losses,master_S_threshold, master_Z_threshold, master_S_leaf, master_Z_leaf










