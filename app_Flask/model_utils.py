from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_nn_model(input_shape, layers, config=None):
    model = Sequential()
    for i, layer in enumerate(layers):
        if i == 0:
            model.add(Dense(layer, input_shape=input_shape, activation='relu'))
        else:
            model.add(Dense(layer, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model