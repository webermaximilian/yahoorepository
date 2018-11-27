
# coding: utf-8

# In[1]:


#regular expression
import re
import nltk

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 150


# In[2]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords

#convert into a list of words
#remove unnecessary split into words, no hyphens
def sentence_to_wordlist(raw):
    """
    Receives a raw review and clean it using the following steps:
    1. Remove all non-words
    2. Transform the review in lower case
    3. Remove all stop words
    4. Perform stemming

    Args:
        review: the review that iwill be cleaned
    Returns:
        a clean review using the mentioned steps above.
    """
    
    clean = re.sub("[^A-Za-z0-9]", " ", str(raw))
    clean = clean.lower()
    review = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    clean = review.sub('', str(clean))
    clean = clean.split()
    
    lemmatizer = WordNetLemmatizer()
    clean = [lemmatizer.lemmatize(i, pos='v') for i in clean]
    return clean


# In[3]:


#pad every sentence to the same sentence length
from keras.preprocessing.sequence import pad_sequences 
maxlen = MAX_SEQUENCE_LENGTH


# In[4]:


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

