
# coding: utf-8

# In[74]:


import pandas as pd, numpy as np
#regular expression
import re

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 150


# In[75]:


#Load whole yahoo training dataset
unprocessed_training_data = pd.read_csv(r'C:\Users\maximilian.weber\OneDrive - Synpulse\UserRoaming\Desktop\TextClassificationDatasets-20181112T073934Z-001\TextClassificationDatasets\yahoo_answers_csv\train.csv', header=None)
unprocessed_training_data.columns = ['labels','questions_1','questions_2', 'features']


# In[76]:


#Load whole yahoo testing dataset
unprocessed_testing_data = pd.read_csv(r'C:\Users\maximilian.weber\OneDrive - Synpulse\UserRoaming\Desktop\TextClassificationDatasets-20181112T073934Z-001\TextClassificationDatasets\yahoo_answers_csv\test.csv', header=None)
unprocessed_testing_data.columns = ['labels','questions_1','questions_2', 'features']


# In[77]:


unprocessed_training_data.head()


# In[78]:


unprocessed_testing_data.head()


# In[79]:


print(len(unprocessed_training_data))


# In[80]:


print(len(unprocessed_testing_data))


# In[81]:


unprocessed_training_data.isnull().sum()


# In[82]:


unprocessed_testing_data.isnull().sum()


# In[83]:


del unprocessed_training_data['questions_1']
del unprocessed_training_data['questions_2']


# In[84]:


del unprocessed_testing_data['questions_1']
del unprocessed_testing_data['questions_2']


# In[85]:


unprocessed_training_data.head()


# In[86]:


unprocessed_testing_data.head()


# In[87]:


processed_training_data = unprocessed_training_data.dropna()
processed_training_data.isnull().sum()


# In[88]:


processed_testing_data = unprocessed_testing_data.dropna()
processed_testing_data.isnull().sum()


# In[89]:


print(len(processed_training_data))


# In[90]:


print(len(processed_testing_data))


# In[91]:


#Get data content of training data with label 1-3
training_data_label_1 = processed_training_data[processed_training_data["labels"] == 1]
training_data_label_2 = processed_training_data[processed_training_data["labels"] == 2]
training_data_label_3 = processed_training_data[processed_training_data["labels"] == 3]


# In[92]:


##Get data content of testing data with label 1-3
testing_data_label_1 = processed_testing_data[processed_testing_data['labels'] == 1]
testing_data_label_2 = processed_testing_data[processed_testing_data['labels'] == 2]
testing_data_label_3 = processed_testing_data[processed_testing_data['labels'] == 3]


# In[93]:


print(len(training_data_label_1))
print(len(training_data_label_2))
print(len(training_data_label_3))


# In[94]:


print(len(testing_data_label_1))
print(len(testing_data_label_2))
print(len(testing_data_label_3))


# In[95]:


#cut datasets with label 1-3 to size of 10'000
training_data_label_1 = training_data_label_1[:50000]
training_data_label_2 = training_data_label_2[:50000]
training_data_label_3 = training_data_label_3[:50000]


# In[96]:


print(len(training_data_label_1))
print(len(training_data_label_2))
print(len(training_data_label_3))


# In[97]:


training_data_label_1.head()


# In[98]:


#merge training training data label 1 with 2
training_data_label_1_2 = training_data_label_1.append(training_data_label_2)


# In[99]:


#merge training data label 1 & 2 with 3
training_data_label_1_2_3 = training_data_label_1_2.append(training_data_label_3)


# In[100]:


print(len(training_data_label_1_2_3))


# In[101]:


testing_data_label_1.head()


# In[102]:


#merge testinig label data
testing_data_label_1_2 = testing_data_label_1.append(testing_data_label_2)


# In[103]:


testing_data_label_1_2_3 = testing_data_label_1_2.append(testing_data_label_3)


# In[104]:


print(len(testing_data_label_1_2_3))


# In[105]:


#shuffle training data
from sklearn.utils import shuffle
shuffled_training_data = shuffle(training_data_label_1_2_3)


# In[106]:


#shuffle testing data
shuffled_testing_data = shuffle(testing_data_label_1_2_3)


# In[107]:


#fill x_train with training sentences
#fill y_train with corresponding labels
x_train_unordered = shuffled_training_data['features']
y_train_unordered = shuffled_training_data['labels']


# In[108]:


print(len(x_train_unordered))
print(len(y_train_unordered))


# In[109]:


#fill x_test with testing sentences
#fill y_test with corresponding labels (validation purpose)
x_test_unordered = shuffled_testing_data['features']
y_test_unordered = shuffled_testing_data['labels']


# In[110]:


print(len(x_test_unordered))
print(len(y_test_unordered))


# In[111]:


x_train = []
y_train = []
for i in y_train_unordered:
    y_train.append(i)
    
for i in x_train_unordered:
    x_train.append(str(i))

y_train = np.array(y_train).reshape(-1,1) #important for keras


# In[112]:


x_test = []
y_test = []
for i in y_test_unordered:
    y_test.append(i)
    
for i in x_test_unordered:
    x_test.append(i)


# In[113]:


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


# In[114]:


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


# In[43]:


from tqdm import tqdm

#split trainig sentences into words
split_training_sentences = []
for raw_sentence in tqdm(x_train):
    split_training_sentences.append(sentence_to_wordlist(raw_sentence))


# In[44]:


print(split_training_sentences[132])


# In[45]:


#split testing sententes into words
split_testing_sentences = []
for raw in tqdm(x_test):
    split_testing_sentences.append(sentence_to_wordlist(raw))


# In[46]:


print(split_testing_sentences[4])


# In[72]:


#text to numbers ordered by most used 1 to less used high number
from keras.preprocessing.text import Tokenizer

#allows to vectorize a text corpus, by turning each text into either a sequence of integers
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(split_training_sentences)

tokenized_training_set = tokenizer.texts_to_sequences(split_training_sentences)
tokenized_testing_set = tokenizer.texts_to_sequences(split_testing_sentences)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index


# In[73]:


import pickle

# saving tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[48]:


#pad every sentence to the same sentence length
from keras.preprocessing.sequence import pad_sequences 

maxlen = MAX_SEQUENCE_LENGTH

padded_training_set = pad_sequences(tokenized_training_set, maxlen=maxlen)
padded_testing_set = pad_sequences(tokenized_testing_set, maxlen=maxlen)


# In[49]:


print(padded_training_set[4])


# In[50]:


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


# In[51]:


embedding_dim = 50
embedding_matrix = create_embedding_matrix(
    r'C:\Users\maximilian.weber\Downloads\glove.6B\glove.6B.50d.txt',
    tokenizer.word_index, embedding_dim)


# In[52]:


#necessary if not binary
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(y_train)


# In[53]:


from keras.models import Sequential
from keras import layers

embedding_dim = 50

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


# In[54]:


history = model.fit(padded_training_set, categorical_labels, batch_size=64, epochs=3, validation_split=0.1)


# In[59]:


results = model.predict(padded_testing_set)


# In[69]:


# Save the weights
model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())


# In[67]:


number = 501

probability = max(results[number])
predicted_category = np.argmax(results[number])
true_category = y_test[number]
text_sequence = split_testing_sentences[number]

print('I\'m ' + str(round(probability * 100)) + '% sure about my prediction!')
print(100*'_')
if predicted_category == 1:
    print('predicted: Society & Culture')
elif predicted_category == 2:
    print('predicted: Science & Math')
elif predicted_category == 3:
    print('predicted: Health')
print(100*'_')
if true_category == 1:
    print('True Category: Society & Culture')
elif true_category == 2:
    print('True Category: Science & Math')
elif true_category == 3:
    print('True Category: Health')
print(100*'_')
print(text_sequence)
print(100*'_')
print(x_test[number])


# In[57]:


#gets all maximum numbers of 3 predictions per array
new_list = []
for i in results:
    new_list.append(max(i))


# In[58]:


#gets all categories corresponding to each array
categories_list = []
for i in results:
    categories_list.append(np.argmax(i))


# In[115]:


#amount of overconfident predictions
overconfident_amount = 0
for i, j, k in zip(new_list, y_test, categories_list):
    if i >= 0.9 and j != k:
        overconfident_amount = overconfident_amount + 1
    else:
        pass


# In[116]:


#go through all overconfident predicted ones and give back index in list
lookagain = []
for i, j, k in zip(new_list, y_test, categories_list):
    if i >= 0.9 and j != k:
        lookagain.append(new_list.index(i))
    else:
        pass


# In[117]:


print(lookagain[449])


# In[109]:


##NOTWORKING FOR NOW!!!!
listofallwrong = []
number = 0
for i in split_testing_sentences:
    if split_testing_sentences.index(i) == lookagain[number]:
        listofallwrong.append(i)
        number = number + 1
    else:
        pass


# In[118]:


percent = round(overconfident_amount/len(results) * 100,2)
print(str(percent)+'%')
print(overconfident_amount)
print(len(results))

#vorher 3% jetzt 1.56% mit glove trainable = false


# In[375]:


good = 0
bad = 0
for x, y in zip(results, y_test):
    true_category = y
    probability = max(x)
    predicted_category = np.argmax(x)

    if predicted_category == true_category:
        good = good + 1
    else:
        bad = bad + 1


# In[376]:


print(good)
print(bad)
ow = bad + good
print(str(round(good/ow*100))+'%')


# In[61]:


example = input('Test: ')
example = sentence_to_wordlist(example)
example = tokenizer.texts_to_sequences([example])
example = pad_sequences(example, maxlen=maxlen)

category_probability = model.predict(example)

print(max(category_probability[0]))

number = np.argmax(category_probability)
if number == 1:
    print('Society & Culture')
elif number == 2:
    print('Science & Math')
elif number == 3:
    print('Health')

