import math as m
import json
import os
import numpy as np
import pandas as pd
from functions.normalization import Normalization

def optimal_synapsis(N_x,N_y,samples):
    """
    N_x - input space dimensions.

    N_y - output space dimensions

    samples - count of samples in dataset

    returns (min,max) amount of synapsis this is needed to generalize data distribution

    Optimal value lies somewhere between.

    A formula is known in the theory of neural networks 
    that is a consequence of the 
    Arnold – Kolmogorov – Hecht-Nielsen theorem.
    Computes optimal amount of synapsis needed to build neural network
    that generalizes data distribution
    """
    minS = N_y*samples/(1+m.log2(samples))
    maxS = N_y*(samples/N_x+1)*(N_x+N_y+1)+N_y
    return minS,maxS

def four_layers_optimal_size(N_x,N_y,samples,alpha1,alpha2):
    """
    for network N_x * L * K1 * K2 * N_y

    N_x - input space dimensions.

    N_y - output space dimensions

    samples - count of samples in dataset

    alpha1 - relation of third layer to first K1=alpha1*L

    alpha2 - relation of third layer to first K2=alpha2*L

    returns (min,max) size of second layer (L) for neural network. Third and fourth layer size can be obtained by formula above

    Optimal value lies somewhere between.

    A formula is known in the theory of neural networks 
    that is a consequence of the 
    Arnold – Kolmogorov – Hecht-Nielsen theorem.
    Computes optimal amount of synapsis needed to build neural network
    that generalizes data distribution
    """
    maxS,minS = optimal_synapsis(N_x,N_y,samples)
    C=alpha1+alpha1*alpha2
    M=N_x+alpha2*N_y
    L_min = (np.sqrt(M*M+4*C*minS)-M)*0.5/C
    L_max = (np.sqrt(M*M+4*C*maxS)-M)*0.5/C
    return (L_min,L_max)

def three_layers_optimal_size(N_x,N_y,samples,alpha):
    """
    for network N_x * L * K * N_y

    N_x - input space dimensions.

    N_y - output space dimensions

    samples - count of samples in dataset

    alpha - relation of third layer to first K=alpha*L

    returns (min,max) size of second layer (L) for neural network. Third layer size can be obtained as K=alpha*L

    Optimal value lies somewhere between.

    A formula is known in the theory of neural networks 
    that is a consequence of the 
    Arnold – Kolmogorov – Hecht-Nielsen theorem.
    Computes optimal amount of synapsis needed to build neural network
    that generalizes data distribution
    """
    maxS,minS = optimal_synapsis(N_x,N_y,samples)
    N_sum = N_x+alpha*N_y
    N_sum_sq = N_sum**2

    L_min = (np.sqrt(N_sum_sq+4*alpha*minS)-N_sum)*0.5/alpha
    L_max = (np.sqrt(N_sum_sq+4*alpha*maxS)-N_sum)*0.5/alpha

    return L_min,L_max

def two_layers_optimal_size(N_x,N_y,samples):
    """
    for network N_x * L * N_y

    N_x - input space dimensions.

    N_y - output space dimensions

    samples - count of samples in dataset

    returns (min,max) size of second layer (L) for neural network

    Optimal value lies somewhere between.

    A formula is known in the theory of neural networks 
    that is a consequence of the 
    Arnold – Kolmogorov – Hecht-Nielsen theorem.
    Computes optimal amount of synapsis needed to build neural network
    that generalizes data distribution
    """
    maxS,minS = optimal_synapsis(N_x,N_y,samples)
    s_sum = N_x+N_y
    return minS/s_sum,maxS/s_sum

def load_dataset(working_dir,dataset):
    """Loads dataset, returns data, normalization"""
    # load model
    data_original = pd.read_csv(f"{working_dir}/{dataset}").dropna()
    data=data_original.to_numpy(dtype='float32')

    if os.path.isfile(f"{working_dir}/meta.json"):
        with open(f"{working_dir}/meta.json") as f:
            meta = json.loads(f.read())
        translation = np.array(meta["translation"],dtype="float32")
        scale = np.array(meta["scale"],dtype="float32")
        norm = Normalization(translation,scale)
        data = norm.transform(data)
    else:
        # make it zero centred by subtracting median
        norm = Normalization()
        data = norm.fit(data)

        with open(f"{working_dir}/meta.json","w") as f:
            f.write(json.dumps({
                "translation":[float(i) for i in norm.translation],
                "scale": [float(i) for i in norm.scale]
            }))
    return data,norm

def get_prediction_errors(data,model,data_input_split):
    """
    Returns error vector that is in same order as data 
    rows and indices row that contains permutation for 
    data to sort it from highest model error to lowest
    """
    prediction = model.predict(data[:,:data_input_split])

    # just use translation abs error
    error = np.average(np.abs(data[:,data_input_split:]-prediction),axis=1)
    return error

def average_prediction_error(data,models,data_input_split):
    """
    find average prediction error vector on some amount of models
    """
    avg_error = np.zeros(len(data),dtype="float32")
    for model in models:
        error = get_prediction_errors(data,model,data_input_split)
        avg_error+=error
    avg_error/=len(models)
    return avg_error
