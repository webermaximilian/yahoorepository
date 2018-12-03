from function_library import *

from keras.models import model_from_json

# Model reconstruction from JSON file
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('model_weights.h5')


# In[5]:


import pickle
# loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# In[7]:


import numpy as np

example = 'dont_stop'

print(100*'_')
print('Start categorizing or type "quit" to close the program!')
print('Text can be categorized into 3 categories!')
print('- Society & Culture')
print('- Science & Math')
print('- Health')
print(100*'_')

while example != 'quit':
    example = input('Text: ')

    if example == 'quit':
        print('Program successfully closed!')
        break

    processed_example = sentence_to_wordlist(example)
    processed_example = tokenizer.texts_to_sequences([processed_example])
    processed_example = pad_sequences(processed_example, maxlen=maxlen)

    category_probability = model.predict(processed_example)

    probability = max(category_probability[0])
    print('Probability: ' + str(probability))

    number = np.argmax(category_probability[0])
    if number == 1:
        print('Predicted category: Society & Culture')
    elif number == 2:
        print('Predicted category: Science & Math')
    elif number == 3:
        print('Predicted category: Health')
    print(100*'_')
