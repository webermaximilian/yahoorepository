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
