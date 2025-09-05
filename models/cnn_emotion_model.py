from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer

def build_cnn_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(48, 48, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # Assuming 7 emotions
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
