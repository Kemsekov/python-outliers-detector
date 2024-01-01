from keras import layers, regularizers, losses, optimizers, metrics
import keras
from functions.utils import *

#build 2 layer NN
def build_model(X,Y,xi):
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
            activation='tanh',
            kernel_regularizer=regularizers.L2(0.01)))

    model.add(
        layers.Dense(
            N_y, 
            activation='tanh',
            kernel_regularizer=regularizers.L2(0.01)))
    # model.summary()
    model.compile(
        optimizer=optimizers.Adam(0.001), 
        loss=losses.mse,
        metrics=metrics.mean_absolute_error
    )
    
    model.fit(x=X,y=Y,epochs=100,verbose=0)
    return model