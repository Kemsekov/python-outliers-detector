from keras import layers, regularizers, losses, optimizers, metrics, activations
import keras
from functions.utils import *

# this function is used to find outliers. It must return `Functional` type - tf model
def build_model(X,Y,xi):
    return build_model_2_layers(X,Y,xi)
    # return build_model_3_layers(X,Y,xi,0.5)
    # return build_model_4_layers(X,Y,xi,0.7,0.3)

# below is examples of build model functions that you can use
# -----------------------------------------------


regularization_coefficient = 0.01
regularization=regularizers.L2(regularization_coefficient)
learning_rate = 0.001
epochs=100
activation = activations.tanh

# build 2 layers NN
def build_model_2_layers(X,Y,xi):
    """
    Builds two layers model: N_x * L * N_y
    
    X - input rows

    Y - output rows

    xi - value in range [0;1] that indicates min and max 
    theoretical size of layer L needed to build model 
    that can learn and generalize data X:Y.
    Optimal value of xi is somewhere between 0 and 1 and it exists for any given dataset.   
    """
    N_x = X.shape[1]
    N_y = Y.shape[1]
    Q = X.shape[0]
    minN,maxN = two_layers_optimal_size(N_x,N_y,Q)
    n = minN+xi*(maxN-minN)
    
    model = keras.Sequential()

    model.add(layers.InputLayer(input_shape=(N_x,)))
    model.add(
        layers.Dense(
            n,
            activation=activation,
            kernel_regularizer=regularization))

    model.add(
        layers.Dense(
            N_y, 
            activation=activation,
            kernel_regularizer=regularization))
    # model.summary()
    model.compile(
        optimizer=optimizers.Adam(learning_rate), 
        loss=losses.mse,
        metrics=metrics.mean_absolute_error
    )
    
    model.fit(x=X,y=Y,epochs=epochs,verbose=0)
    return model

# build 3 layers NN
def build_model_3_layers(X,Y,xi,alpha):
    """
    Builds three layers model: N_x * L * K * N_y
    
    X - input rows

    Y - output rows

    xi - value in range [0;1] that indicates min and max 
    theoretical size of layer L needed to build model that can learn and generalize data X:Y.
    Optimal value of xi is somewhere between 0 and 1 and it exists for any given dataset.   

    alpha - relation coefficient that is used to find size of third layer K = alpha * L
    """
    N_x = X.shape[1]
    N_y = Y.shape[1]
    Q = X.shape[0]
    minN,maxN = three_layers_optimal_size(N_x,N_y,Q,alpha)
    L_size = minN+xi*(maxN-minN)
    K_size = alpha*L_size

    model = keras.Sequential()
    
    model.add(layers.InputLayer(input_shape=(N_x,)))
    model.add(
        layers.Dense(
            L_size,
            activation=activation,
            kernel_regularizer=regularization))

    model.add(
        layers.Dense(
            K_size,
            activation=activation,
            kernel_regularizer=regularization))

    model.add(
        layers.Dense(
            N_y, 
            activation=activation,
            kernel_regularizer=regularization))
    # model.summary()
    model.compile(
        optimizer=optimizers.Adam(learning_rate), 
        loss=losses.mse,
        metrics=metrics.mean_absolute_error
    )
    
    model.fit(x=X,y=Y,epochs=epochs,verbose=0)
    return model

# build 4 layers NN
def build_model_4_layers(X,Y,xi,alpha1,alpha2):
    """
    Builds four layers model: N_x * L * K1 * K2 * N_y
    
    X - input rows

    Y - output rows

    xi - value in range [0;1] that indicates min and max 
    theoretical size of layer L needed to build model that can learn and generalize data X:Y.
    Optimal value of xi is somewhere between 0 and 1 and it exists for any given dataset.   

    alpha1 - relation coefficient that is used to find size of third layer K1 = alpha1 * L

    alpha2 - relation coefficient that is used to find size of fourth layer K2 = alpha2 * L
    """
    N_x = X.shape[1]
    N_y = Y.shape[1]
    Q = X.shape[0]
    minN,maxN = four_layers_optimal_size(N_x,N_y,Q,alpha1,alpha2)
    L_size = minN+xi*(maxN-minN)
    K1_size = alpha1*L_size
    K2_size = alpha2*L_size

    model = keras.Sequential()
    
    model.add(layers.InputLayer(input_shape=(N_x,)))
    model.add(
        layers.Dense(
            L_size,
            activation=activation,
            kernel_regularizer=regularization))

    model.add(
        layers.Dense(
            K1_size,
            activation=activation,
            kernel_regularizer=regularization))

    model.add(
        layers.Dense(
            K2_size,
            activation=activation,
            kernel_regularizer=regularization))

    model.add(
        layers.Dense(
            N_y, 
            activation=activation,
            kernel_regularizer=regularization))
    # model.summary()
    model.compile(
        optimizer=optimizers.Adam(learning_rate), 
        loss=losses.mse,
        metrics=metrics.mean_absolute_error
    )
    
    model.fit(x=X,y=Y,epochs=epochs,verbose=0)
    return model

