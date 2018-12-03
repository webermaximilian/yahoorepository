#pandas for reading csv
#numpy for mathematical calculations
import pandas as pd, numpy as np
#regular expression - manipulate sentences
import re
#Hyperparameters
#MAX_NB_WORDS provides limit of words stored in vocabulary
MAX_NB_WORDS = 200000
#MAX_SEQUENCE_LENGTH sets limit of words in sequence
MAX_SEQUENCE_LENGTH = 150
#dimensions of embedding matrix
embedding_dim = 50


#MAYBE INDEX = NONE BECAUSE OF UNNECESSARY 4th COLUMN!! + PATH RELATIVE!!
def process_data(path_to_origin, path_to_train_file, path_to_test_file):
    #Load whole yahoo TRAINING dataset from filepath
    unprocessed_training_data = pd.read_csv(path_to_origin + path_to_train_file, header=None, index_col=None)
    #own header titles
    unprocessed_training_data.columns = ['labels','questions_1','questions_2', 'features']
    #Load whole yahoo TESTING dataset from filepath
    unprocessed_testing_data = pd.read_csv(path_to_origin + path_to_test_file, header=None, index_col=None)
    #Own header titles
    unprocessed_testing_data.columns = ['labels','questions_1','questions_2', 'features']
    return unprocessed_training_data, unprocessed_testing_data


def delete_data(unprocessed_training_data, unprocessed_testing_data):
    #delete unnecessary columns of TRAINING dataset
    del unprocessed_training_data['questions_1']
    del unprocessed_training_data['questions_2']
    #delete unnecessary columns of TESTING dataset
    del unprocessed_testing_data['questions_1']
    del unprocessed_testing_data['questions_2']
    #delete all rows without content of TRAINING dataset
    processed_training_data = unprocessed_training_data.dropna()
    #delete all rows without content of TESTING dataset
    processed_testing_data = unprocessed_testing_data.dropna()
    return processed_training_data, processed_testing_data


def drop_na_data():
    #delete all rows without content of TRAINING dataset
    processed_training_data = unprocessed_training_data.dropna()
    #delete all rows without content of TESTING dataset
    processed_testing_data = unprocessed_testing_data.dropna()


def get_specific_content():
    #Get data content of TRAINING data with label 1-3
    training_data_label_1 = processed_training_data[processed_training_data["labels"] == 1]
    training_data_label_2 = processed_training_data[processed_training_data["labels"] == 2]
    training_data_label_3 = processed_training_data[processed_training_data["labels"] == 3]
    ##Get data content of TESTING data with label 1-3
    testing_data_label_1 = processed_testing_data[processed_testing_data['labels'] == 1]
    testing_data_label_2 = processed_testing_data[processed_testing_data['labels'] == 2]
    testing_data_label_3 = processed_testing_data[processed_testing_data['labels'] == 3]


def cut_training_data():
    #cut ONLY TRAINING datasets with label 1-3 to size of 50'000
    training_data_label_1 = training_data_label_1[:50000]
    training_data_label_2 = training_data_label_2[:50000]
    training_data_label_3 = training_data_label_3[:50000]


def merge_data():
    #merge TRAINING data label 1 with 2
    training_data_label_1_2 = training_data_label_1.append(training_data_label_2)
    #merge TRAINING data label 1 & 2 with 3
    training_data_label_1_2_3 = training_data_label_1_2.append(training_data_label_3)
    #merge TESTING data label 1 with 2
    testing_data_label_1_2 = testing_data_label_1.append(testing_data_label_2)
    #merge TESTING data label 1 & 2 with 3
    testing_data_label_1_2_3 = testing_data_label_1_2.append(testing_data_label_3)


def shuffle_data():
    #shuffle TRAINING data
    from sklearn.utils import shuffle
    shuffled_training_data = shuffle(training_data_label_1_2_3)
    #shuffle TESTING data
    shuffled_testing_data = shuffle(testing_data_label_1_2_3)


def get_features_and_labels():
    #unordered because data was removed & appended --> some n/a arrays
    #fill x_train_unordered with TRAINING sentences
    #fill y_train_unordered with corresponding labels
    x_train_unordered = shuffled_training_data['features']
    y_train_unordered = shuffled_training_data['labels']
    #unordered because data was removed & appended --> some n/a arrays
    #fill x_test_unordered with testing sentences
    #fill y_test_unordered with corresponding labels (validation purpose ONLY)
    x_test_unordered = shuffled_testing_data['features']
    y_test_unordered = shuffled_testing_data['labels']


def remove_na_cells():
    #removes n/a arrays in x_train_unordered and y_train_unordered
    x_train = []
    y_train = []
    for i in y_train_unordered:
        y_train.append(i)

    for i in x_train_unordered:
        x_train.append(str(i))

    #important for keras model (matrix format)
    y_train = np.array(y_train).reshape(-1,1)
    #removes n/a arrays in x_test_unordered and y_test_unordered
    x_test = []
    y_test = []
    for i in y_test_unordered:
        y_test.append(i)

    for i in x_test_unordered:
        x_test.append(i)
    #no need for reshaping because nothing gets trained in keras model


def load_lemmatizer():
    #WordNetLemmatizer formats words back to its basis - removes noise in dataset
    #example: go, went, gone, goes, going --> go
    import nltk
    from nltk.stem import WordNetLemmatizers
    nltk.download('wordnet')


#@Pre raw sentences
#@Post cleaned wordlist
#convert into a list of words
#remove unnecessary split into words, no hyphens
def sentence_to_wordlist(raw):
    import nltk
    from nltk.stem import WordNetLemmatizer
    #stopwords removes any unvaluable words like this, for, in, ...
    from nltk.corpus import stopwords
    """
    Receives a raw sentence and cleans it using the following steps:
    1. Includes only words and numbers
    2. Transforms the review in lower case
    3. Removes all stop words
    4. Performs lemmatize
    """

    clean = re.sub("[^A-Za-z0-9]", " ", str(raw))
    clean = clean.lower()
    review = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    clean = review.sub('', str(clean))
    clean = clean.split()

    lemmatizer = WordNetLemmatizer()
    clean = [lemmatizer.lemmatize(i, pos='v') for i in clean]
    return clean


def process_sentences_to_wordlist():
    #displays loading bar
    from tqdm import tqdm
    #split TRAINING sentences into wordlist
    split_training_sentences = []
    for raw_sentence in tqdm(x_train):
        split_training_sentences.append(sentence_to_wordlist(raw_sentence))

    #split TESTING sententes into wordlist
    split_testing_sentences = []
    for raw in tqdm(x_test):
        split_testing_sentences.append(sentence_to_wordlist(raw))


def perform_tokenizer():
    #text to numbers ordered by most used 1 to less used high number
    from keras.preprocessing.text import Tokenizer

    #allows to vectorize a text corpus, by turning each text into a sequence of integers
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(split_training_sentences)

    tokenized_training_set = tokenizer.texts_to_sequences(split_training_sentences)
    tokenized_testing_set = tokenizer.texts_to_sequences(split_testing_sentences)
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

def save_tokenizer():
    import pickle
    #saves tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#had to take it out of the function because model_predictor needed it!!!!
from keras.preprocessing.sequence import pad_sequences
maxlen = MAX_SEQUENCE_LENGTH
def pad_tokenized_sequences():
    #pad every sentence to the same sentence length (necessary for matrix calculations)
    padded_training_set = pad_sequences(tokenized_training_set, maxlen=maxlen)
    padded_testing_set = pad_sequences(tokenized_testing_set, maxlen=maxlen)


#@Pre filepath of pretrained model, word_index of pretrained tokenizer, embedding_dim as Hyperparameter
#@Post returns embedding_matrix which is used in keras model
#creates embedding matrix for later use with pretrained glove model
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


def build_embedding_matrix():
    #builds embedding_matrix with pretrained glove model
    embedding_matrix = create_embedding_matrix(
        r'C:\Users\maximilian.weber\Downloads\glove.6B\glove.6B.50d.txt',
        tokenizer.word_index, embedding_dim)


def predefined_keras_transformation():
    #necessary if model classifies non-binary
    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(y_train)
