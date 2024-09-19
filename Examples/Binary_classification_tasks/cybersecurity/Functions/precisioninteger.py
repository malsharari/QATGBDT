import json
import numpy as np




def dequantize(q, S, Z,N):

    r = S * (np.array(q) - Z )
    return r

def dequantize_data(q, S, Z,N):

    r = S * (q - Z)
    return r


def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max

def linear_quantization(r, N, Symmetric):
    # Calculate the quantization range for Signed int

    q_min,q_max=get_quantized_range(N)

    # Flatten the input array
    r_flat = np.array(r).flatten()
    
    r_max = max(r_flat)   
    r_min = min(r_flat) 

    
    # Compute the scale
    # Compute the initial zero point
    # Adjust the zero point to be an integer within the quantization range
    if Symmetric:
        Z = 0
        r_max=max(max(abs(r_flat)), 5e-16)
        S = (r_max/q_max)

    else:

        S = (r_max - r_min) / (q_max - q_min)
        Z = q_min - (r_min / S)
        
        
    # convert from float to int using round()        
    Z = np.round(Z)    
    # clip the zero_point to fall in [quantized_min, quantized_max]
    #Z = np.clip(Z, q_min, q_max)

        
    # Quantize the input array
    q = (np.round(r_flat / S)) + Z
    q = np.clip(q, q_min, q_max)

    return q, S, Z


def quantize(r,S,Z, N):
    
    q_min,q_max=get_quantized_range(N)
    q = (np.round(np.array(r).flatten() / S)) + Z
    q = np.clip(q, q_min, q_max)

    return q

def quantized_data(r,S,Z, N):
    
    q_min,q_max=get_quantized_range(N)
    q = (np.round(r / S)) + Z
    q = np.clip(q, q_min, q_max)

    return q

def quantized_to_int(r,S,Z, N):

    q_min,q_max=get_quantized_range(N) 
    q = (np.round(r / S)) + Z
    q = np.clip(q, q_min, q_max)

    return q


def bwn(x, bta=1,Beta=None):
    # Compute the scaling factor alpha
    
    x=np.array(x).flatten()
    alpha = np.mean(np.abs(x))
    if Beta is not None:
        alpha = Beta
    # Binarize x using the sign function instead of np.where(x <= 0, -1, 1)
    binarized_x = np.sign(x)
    # Apply the scaling
    if bta == 1:
        return alpha * binarized_x,alpha
    elif bta == 0:
        return np.where(x <= 0, -1, 1),alpha    
    else:
        return np.where(x <= 0, -bta/2, bta/2),alpha

    

def update_combined_precision(input_filepath, output_filepath, 
                              Flag=[ True , True , True , True], 
                              N=16,Binary=False):
    
    with open(input_filepath, "r") as f1:
        model = json.load(f1)    
    bta=1    
    # Initialize lists or dictionaries for S and Z values for each tree
    S_threshold_per_tree = []
    Z_threshold_per_tree = []
    S_leaf_per_tree = []
    Z_leaf_per_tree = []
    all_threshold_values = []
    all_leaf_values = []
    
    
    for index in set(range(len(model['oblivious_trees']))):
        # Process each tree
        tree = model['oblivious_trees'][index]

        # Collect leaf values from the tree
        if 'leaf_values' in tree:
            all_leaf_values.extend(tree['leaf_values'])

    #Extract feature borders and add to all_threshold_values
    float_features = model['features_info']['float_features']
    feature_borders = [feature.get('borders') for feature in float_features]
    all_threshold_values.extend([border for sublist in feature_borders if sublist is not None for border in sublist])

    # Compute the S and Z from the collected list for this tree
    if Binary:
        _, S_leaf= bwn(np.array(all_leaf_values), bta)
        Z_leaf =0        
        S_threshold=1
        Z_threshold = 0

    else:
        _, S_threshold, Z_threshold = linear_quantization(np.array(all_threshold_values), N, Symmetric= Flag[1])
        _, S_leaf, Z_leaf = linear_quantization(np.array(all_leaf_values), N, Symmetric= Flag[3])

    
    for index in set(range(len(model['oblivious_trees']))):
        # Update tree splits and leaf values using computed S and Z
        tree = model['oblivious_trees'][index]
        if not Binary: 
            if tree.get('splits') is not None:
                threshold_value = np.zeros_like(tree['splits'], dtype=np.float)
                for i, split in enumerate(tree['splits']):
                    if 'float_feature_index' in split and split['border'] is not None:
                        threshold_value[i] = float(split['border'])
                if not Binary:
                    threshold_value = dequantize(quantize(np.array(threshold_value), S_threshold, Z_threshold,N), S_threshold, Z_threshold,N).tolist()

                for i, split in enumerate(tree['splits']):
                    if 'float_feature_index' in split and split['border'] is not None:
                        split['border'] = float(threshold_value[i])

        if 'leaf_values' in tree:
            leaf_values = tree['leaf_values']
            if Binary:
                tree['leaf_values']= bwn(np.array(leaf_values),bta,S_leaf)[0].tolist()

            else:
                tree['leaf_values'] = dequantize(quantize(np.array(leaf_values), S_leaf, Z_leaf, N), S_leaf, Z_leaf,N).tolist()

    # Append S and Z values for this tree
    S_threshold_per_tree.append(S_threshold)
    Z_threshold_per_tree.append(Z_threshold)
    S_leaf_per_tree.append(S_leaf)
    Z_leaf_per_tree.append(Z_leaf)

    # Update feature borders using computed S and Z
    if not Binary:
        filtered_borders = [border for sublist in feature_borders if sublist is not None for border in sublist]

        processed_borders = dequantize(quantize(np.array(filtered_borders), S_threshold, Z_threshold, N), S_threshold, Z_threshold,N)
        pb_index = 0
        updated_borders = []
        for border in feature_borders:
            if border is not None:

                updated_borders.append(processed_borders[pb_index:pb_index+len(border)].tolist())

                pb_index += len(border)
            else:
                updated_borders.append(None)

        for i, feature in enumerate(float_features):
            feature['borders'] = updated_borders[i]

    # Save the updated model
    if output_filepath is None:
        raise ValueError("An output_filepath must be specified.")
    with open(output_filepath, "w") as f2:
        json.dump(model, f2, indent=2)

    # Return lists of S and Z values for this iteration
    return S_threshold_per_tree, Z_threshold_per_tree, S_leaf_per_tree, Z_leaf_per_tree


def update_precision_with_precomputed_SZ(input_filepath, output_filepath, S_threshold_per_tree, Z_threshold_per_tree, S_leaf_per_tree, Z_leaf_per_tree,Flag=[ True , True , True , True],N=4, Binary=False,bta=0):
    with open(input_filepath, "r") as f1:
        model = json.load(f1)

    for index in set(range(len(model['oblivious_trees']))):
        # Retrieve the S and Z values for the current tree
        S_threshold = S_threshold_per_tree[-1:][0]
        Z_threshold = Z_threshold_per_tree[-1:][0]
        S_leaf = S_leaf_per_tree[-1:][0]
        Z_leaf = Z_leaf_per_tree[-1:][0]
        #print(index)
        # Process each tree
        tree = model['oblivious_trees'][index]

        if not Binary: 
            # Extract feature borders and add to all_threshold_values
            float_features = model['features_info']['float_features']
            feature_borders = [feature.get('borders') for feature in float_features]
            #Quantize threshold values using the precomputed S and Z
            if tree.get('splits') is not None:
                threshold_value = np.zeros_like(tree['splits'], dtype=np.float)
                for i, split in enumerate(tree['splits']):
                    if 'float_feature_index' in split and split['border'] is not None:
                        threshold_value[i] = float(split['border'])
                #print(threshold_value, S_threshold, Z_threshold)
                #print('threshold_value')
                threshold_value = quantized_to_int(threshold_value, S_threshold, Z_threshold,N ).tolist()
                for i, split in enumerate(tree['splits']):
                    if 'float_feature_index' in split and split['border'] is not None:
                        split['border'] = float(threshold_value[i])

        # Quantize leaf values using the precomputed S and Z
        if 'leaf_values' in tree:
            leaf_values = tree['leaf_values']
            #print('leaf_values')
            if Binary:
                tree['leaf_values']= bwn(np.array(leaf_values),bta,S_leaf)[0].tolist()
                
            else:     
                tree['leaf_values'] = quantized_to_int(leaf_values, S_leaf, Z_leaf,N).tolist()
            
            
            
    if not Binary:       
        # Update feature borders using computed S and Z
        filtered_borders = [border for sublist in feature_borders if sublist is not None for border in sublist]
        #print('filtered_borders')
        processed_borders = quantized_to_int(np.array(filtered_borders), S_threshold, Z_threshold, N)
        pb_index = 0
        updated_borders = []
        for border in feature_borders:
            if border is not None:
                updated_borders.append(processed_borders[pb_index:pb_index+len(border)].tolist())
                pb_index += len(border)
            else:
                updated_borders.append(None)

        for i, feature in enumerate(float_features):
            feature['borders'] = updated_borders[i]
                   
    # Save the updated model
    if output_filepath is None:
        raise ValueError("An output_filepath must be specified.")
    with open(output_filepath, "w") as f2:
        json.dump(model, f2, indent=2)
        
        
        
        
def update_to_PTQ(input_filepath, output_filepath, Flag=[ True , True , True , True],N=16):
    
    
    with open(input_filepath, "r") as f1:
        model = json.load(f1)

    # Step 1: Collect all threshold values and splits from the model into one list.
    all_threshold_values = []
    all_leaf_values = []


    for index in set(range(len(model['oblivious_trees']))):
        tree = model['oblivious_trees'][index]
        if tree.get('splits') is not None:
            for split in tree['splits']:
                if 'float_feature_index' in split and split['border'] is not None:
                    # all_threshold_values.append(float(split['border']))
                    continue
        if 'leaf_values' in tree:
            all_leaf_values.extend(tree['leaf_values'])
            

    # Extract feature borders and add to all_threshold_values
    float_features = model['features_info']['float_features']
    feature_borders = [feature.get('borders') for feature in float_features]
    all_threshold_values.extend([border for sublist in feature_borders if sublist is not None for border in sublist])

    # print(all_threshold_values)
    # Step 2: Compute the global S and Z from the collected list.
    _, S_threshold, Z_threshold = linear_quantization(np.array(all_threshold_values), N, Flag[1])
    _, S_leaf, Z_leaf = linear_quantization(np.array(all_leaf_values), N, Flag[3])


    # Update tree splits and leaf values using global S and Z
    for index in set(range(len(model['oblivious_trees']))):
        tree = model['oblivious_trees'][index]
        if tree.get('splits') is not None:
            threshold_value = np.zeros_like(tree['splits'], dtype=np.float)
            for i, split in enumerate(tree['splits']):
                if 'float_feature_index' in split and split['border'] is not None:
                    threshold_value[i] = float(split['border'])

            threshold_value = quantize(threshold_value, S_threshold, Z_threshold, N).tolist()
            for i, split in enumerate(tree['splits']):
                if 'float_feature_index' in split and split['border'] is not None:
                    split['border'] = float(threshold_value[i])

        if 'leaf_values' in tree:
            leaf_values = tree['leaf_values']
            tree['leaf_values'] = quantize(leaf_values, S_leaf, Z_leaf, N).tolist()

    # Update feature borders using global S and Z
    filtered_borders = [border for sublist in feature_borders if sublist is not None for border in sublist]
    processed_borders = quantize(np.array(filtered_borders), S_threshold, Z_threshold, N)
    pb_index = 0
    updated_borders = []
    for border in feature_borders:
        if border is not None:
            updated_borders.append(processed_borders[pb_index:pb_index+len(border)].tolist())
            pb_index += len(border)
        else:
            updated_borders.append(None)

    for i, feature in enumerate(float_features):
        feature['borders'] = updated_borders[i]

    if output_filepath is None:
        raise ValueError("An output_filepath must be specified.")
    with open(output_filepath, "w") as f2:
        json.dump(model, f2, indent=2)   
    
    return  S_threshold, Z_threshold, S_leaf, Z_leaf