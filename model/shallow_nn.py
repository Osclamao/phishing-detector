import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_size):
    model = Sequential([
        Dense(16, activation="relu", input_shape=(input_size,)),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
