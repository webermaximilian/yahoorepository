from function_library import *
#pandas for reading csv
#numpy for mathematical calculations
import pandas as pd, numpy as np
#regular expression - manipulate sentences
import re

#C:/Users/maximilian.weber/yahoo_github
path_to_origin = r'yahoorepository'
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
training_data_label_1, training_data_label_2, training_data_label_3, testing_data_label_1, testing_data_label_2,testing_data_label_3 = get_specific_content(processed_training_data, processed_testing_data)
training_data_label_1, training_data_label_2, training_data_label_3 = cut_training_data(training_data_label_1, training_data_label_2, training_data_label_3)
training_data_label_1_2_3, testing_data_label_1_2_3 = merge_data(training_data_label_1, training_data_label_2, training_data_label_3, testing_data_label_1, testing_data_label_2, testing_data_label_3)
shuffled_training_data, shuffled_testing_data = shuffle_data(training_data_label_1_2_3, testing_data_label_1_2_3)
x_train_unordered, y_train_unordered, x_test_unordered, y_test_unordered = get_features_and_labels(shuffled_training_data, shuffled_testing_data)
x_train, y_train, x_test, y_test = remove_na_cells(x_train_unordered, y_train_unordered, x_test_unordered, y_test_unordered)
#load_lemmatizer()
#split_training_sentences, split_testing_sentences = process_sentences_to_wordlist(x_train, x_test)
#tokenizer, tokenized_training_set, tokenized_testing_set, vocab_size = perform_tokenizer(split_training_sentences, split_testing_sentences, MAX_NB_WORDS)
#save_tokenizer(tokenizer)
#save_vocab_size(vocab_size)
#padded_training_set, padded_testing_set = pad_tokenized_sequences(tokenized_training_set, tokenized_testing_set, MAX_SEQUENCE_LENGTH)
#save_preprocessed_train_data(padded_training_set)
#save_preprocessed_test_data(padded_testing_set)
tokenizer = load_tokenizer()
vocab_size = load_vocab_size()
padded_training_set = load_preprocessed_train_data()
padded_testing_set = load_preprocessed_test_data()
embedding_matrix = build_embedding_matrix(path_to_origin, tokenizer)
categorical_labels = predefined_keras_transformation(y_train)


#keras model
from keras.models import Sequential
from keras import layers
#embedding_dim = 50 --> unnecessary predefined above
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           weights=[embedding_matrix],
                           input_length=MAX_SEQUENCE_LENGTH,
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
