from function_library import *
#pandas for reading csv
#numpy for mathematical calculations
import pandas as pd, numpy as np
#regular expression - manipulate sentences
import re


path_to_origin = r'C:/Users/maximilian.weber/yahoo_github'
path_to_train_file = '/train.csv'
path_to_test_file = '/test.csv'

#Hyperparameters
#MAX_NB_WORDS provides limit of words stored in vocabulary
MAX_NB_WORDS = 200000
#MAX_SEQUENCE_LENGTH sets limit of words in sequence
MAX_SEQUENCE_LENGTH = 150
#dimensions of embedding matrix
embedding_dim = 50

unprocessed_training_data, unprocessed_testing_data = process_data(path_to_origin, path_to_train_file, path_to_test_file)
processed_training_data, processed_testing_data = delete_data(unprocessed_training_data, unprocessed_testing_data)


#keras model
from keras.models import Sequential
from keras import layers
#embedding_dim = 50 --> unnecessary predefined above
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           weights=[embedding_matrix],
                           input_length=maxlen,
                           trainable=True))
model.add(layers.SpatialDropout1D(0.5))
model.add(layers.LSTM(196, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(4, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

#model evaluation
history = model.fit(padded_training_set, categorical_labels, batch_size=64, epochs=3, validation_split=0.1)

# Save the weights
model.save_weights('model_weights.h5')
# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
