import os.path
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
import nltk
from nltk.util import ngrams
import gensim

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc, [self.labels_list[idx]])


class Reader:

    dir = os.getcwd() # Gets the current working directory

    encoded_tweets = [] # Different encoding of tweets (One Hot Encoding, TF-IDF, One hot encoding of ngrams)
    word_dict = []  # Dictionary of all the words that exist
    words_of_tweets = [] # Saves all the tweet cleared from stop-words, stemmed and tokenized
    bigrams = [] # Bi-grams of all tweets

    train_A = None
    train_A_emoji = None
    train_A_emoji_hash = None
    train_B = None
    train_B_emoji = None
    train_B_emoji_hash = None

    input_A = None
    input_A_emoji = None
    input_B = None
    input_B_emoji = None




    ##############################################################################################################################################################

    # Pre-processing and convert the input using one hot encoding and TF-IDF

    ##############################################################################################################################################################


    # Pre-processing of the tweets
    def pre_processing(self):
        for i in range(0, len(self.train_A)):
            # Estimate the size of the vocabulary
            #words = set(text_to_word_sequence(self.train_A.iloc[i][2], filters='!"$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'))

    ##        print("printing words: ", words)

            # Tokenize tweets
            from nltk.tokenize import word_tokenize
            words = word_tokenize(self.train_A.iloc[i][2])

            # remove punctuation from each word
            import string
            table = str.maketrans('', '', string.punctuation)
            words = [w.translate(table) for w in words]

            # remove all tokens that are not alphabetic
            words = [word for word in words if word.isalpha()]

            # Delete Stop-Words
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]

            # stemming of words
            from nltk.stem.porter import PorterStemmer
            porter = PorterStemmer()
            words = [porter.stem(word) for word in words]

            # Keep the tokenized tweets
            self.words_of_tweets.append(words)



    # Create a dictionary for one hot encoding and encode with one hot encoding
    def one_hot_enc(self):
        # Create a dictionary from all the words that exist in tweets
        for i in range(0, len(self.train_A)):
            for j in range(0, len(self.words_of_tweets[i])):
                if self.words_of_tweets[i][j] not in self.word_dict:
                    self.word_dict.append(self.words_of_tweets[i][j])

        tweets_size = len(self.train_A)  # Rows of the array encoded_tweets
        vocab_size = len(self.word_dict)  # Columns of the array encoded_tweets

        # Creates a list with rows representing tweets and sized as 'tweets_size' and columns representing unique vocabulary of tweets and sized as 'vocab_size', all set to 0
        self.encoded_tweets = [[0 for x in range(vocab_size)] for y in range(tweets_size)]

        # Set 1 to the encoded_tweets list on the indexes that the words of a specific tweet matches the words of the dictionary
        for y in range(0, tweets_size):
            for x in range(0, vocab_size):
                for a in range(0, len(self.words_of_tweets[y])):
                    # print("len of words: ", tweets_size)
                    if self.words_of_tweets[y][a] == self.word_dict[x]:
                        ##print("got another: ", y, "from: ", tweets_size)
                        self.encoded_tweets[y][x] = 1



    # TF-IDF and statistics
    def tf_idf(self):
        # Create the tokenizer used for TF-IDF
        t = Tokenizer()

        # fit the tokenizer on the documents
        t.fit_on_texts(self.words_of_tweets)

        # summarize what was learned
        print(t.word_counts) # A dictionary of words and their counts
        # print(t.word_docs) # Similar to word_counts - output structure is changed. An integer count of the total number of documents that were used to fit the Tokenizer
        # print(t.document_count) # A dictionary of words and how many documents each appeared in
        # print(t.word_index) # A dictionary of words and their uniquely assigned integers

        # integer encode documents using TF-IDF
        self.encoded_tweets = t.texts_to_matrix(self.words_of_tweets, mode='tfidf')
        print(self.encoded_tweets)



    def bigrams_enc(self):
        # Use the pre-processing done above
        for y in range(0, len(self.words_of_tweets)):
            self.bigrams.append(list(ngrams(self.words_of_tweets[y], 2)))
        #        for y in range(0, len(self.bigrams)):
        #            print(self.bigrams[y])

        diction = []  # The dictionary of bigrams

        tweets_size = len(self.train_A)  # Rows of the array

        # Create the dictionary of bigrams
        for i in range(0, tweets_size):
            for j in range(0, len(self.bigrams[i])):
                if self.bigrams[i][j] not in diction:
                    diction.append(self.bigrams[i][j])
        print(diction)

        vocab_size = len(diction)  # Columns of the array

        # Use One Hot Encoding on the created bigrams
        # Creates a list with rows representing tweets and sized as 'tweets_size' and columns representing unique vocabulary of tweets and sized as 'vocab_size', all set to 0
        self.encoded_tweets = [[0 for x in range(vocab_size)] for y in range(tweets_size)]

        # Set 1 to the ngram list on the indexes that the words of a specific tweet matches the words of the dictionary
        for y in range(0, tweets_size):
            for x in range(0, vocab_size):
                for a in range(0, len(self.bigrams[y])):
                    # print("len of words: ", tweets_size)
                    if self.bigrams[y][a] == diction[x]:
                        ##print("got another: ", y, "from: ", tweets_size)
                        self.encoded_tweets[y][x] = 1

        # for y in range(0, tweets_size):
        # print(self.encoded_tweets[y])

        # CHECK THAT THE ENCODING IS INDEED CORRECT
        for i in range(0, len(self.encoded_tweets[12])):
            if self.encoded_tweets[12][i] == 1:
                print("Diction 12: ", diction[i])
        print("Bigram encod 12: ", self.encoded_tweets[12])


###############################################################################################################################################
###############################################################################################################################################


    def Word2Vec_enc(self):
        from gensim.models import Word2Vec
        from multiprocessing import Pool

        # sg: CBOW if 0, skip-gram if 1
        # window: number of words accounted for each context( if the window size is 3, 3 word in the left neighorhood and 3 word in the right neighborhood are considered)
        model = Word2Vec(sentences=self.words_of_tweets, size=7992, sg=1, window=3, min_count=1, iter=10, workers=Pool()._processes)

        # iterator returned over all documents
#        it = LabeledLineSentence(self.words_of_tweets, self.train_A['label'])
#        model = Word2Vec(size=300, sg=1, window=3, min_count=1, iter=10, workers=Pool()._processes)
#        self.encoded_tweets = model.build_vocab(it)

        model.init_sims(replace=True)  # To make the model memory efficient
        print(model.most_similar('word'))
        print(model.wv.syn0.shape)

        #        print(self.encoded_tweets)

        # Save model locally to reduce time of training the model again
        #model.save('word2vec_model')
        #model = Word2Vec.load('word2vec_model')

        # Training Doc2Vec
        #for epoch in range(10):
            #model.train(self.words_of_tweets.sentences_perm())

        num_features = 7992  # Word vector dimensionality

        #self.encoded_tweets = model.wv
       # print(self.encoded_tweets)
        from w2v_cal import get_doc_matrix
        self.encoded_tweets = get_doc_matrix(model, self.words_of_tweets)
        print(self.encoded_tweets)

        #self.encoded_tweets = self.getAvgFeatureVecs(self.words_of_tweets, model, num_features)




    # Function for calculating the average feature vector
    def getAvgFeatureVecs(self, reviews, model, num_features):
        counter = 0
        reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
        for review in reviews:
            # Printing a status message every 1000th review
            if counter % 1000 == 0:
                print("Review %d of %d" % (counter, len(reviews)))

            reviewFeatureVecs[counter] = self.featureVecMethod(review, model, num_features)
            counter = counter + 1

        return reviewFeatureVecs

    # Function to average all word vectors in a paragraph
    def featureVecMethod(self, words, model, num_features):
        # Pre-initialising empty numpy array for speed
        featureVec = np.zeros(num_features, dtype="float32")
        nwords = 0

        # Converting Index2Word which is a list to a set for better speed in the execution.
        index2word_set = set(model.wv.index2word)

        for word in words:
            if word in index2word_set:
                nwords = nwords + 1
                featureVec = np.add(featureVec, model[word])

        # Dividing the result by number of words to get average
        featureVec = np.divide(featureVec, nwords)
        return featureVec


###############################################################################################################################################
###############################################################################################################################################


    def Doc2Vec_enc(self):
        from gensim.models import Doc2Vec
        from multiprocessing import Pool

        # dm: DBOW if 0, distributed-memory if 1
        # window: number of words accounted for each context( if the window size is 3, 3 word in the left neighorhood and 3 word in the right neighborhood are considered)
        model = Doc2Vec(documents=self.words_of_tweets, dm=1, size=100, window=3, min_count=1, iter=10, workers=Pool()._processes)
        model.init_sims(replace=True)
        model.build_vocab(self.words_of_tweets)

        # Training Doc2Vec
        #for epoch in range(10):
            #model.train(self.words_of_tweets.sentences_perm())




###############################################################################################################################################
###############################################################################################################################################


    def GloVe_enc(self):
        from glove import Corpus, Glove
        from multiprocessing import Pool

        corpus = Corpus()
        corpus.fit(self.words_of_tweets, window=3)  # window parameter denotes the distance of context
        glove = Glove(no_components=100, learning_rate=0.05)

        # matrix: co - occurence matrix of the corpus
        # no_threads: number of training threads
        glove.fit(matrix=corpus.matrix, epochs=30, no_threads=Pool()._processes, verbose=True)
        glove.add_dictionary(corpus.dictionary)  # supply a word-id dictionary to allow similarity queries



    ##############################################################################################################################################################

    # Read the training files for task A and B (with emojis / without emojis / with irony hashtags emojis)

    # train_A
    # train_A_emoji
    # train_A_emoji_hash
    # train_B
    # train_B_emoji
    # train_B_emoji_hash

    ##############################################################################################################################################################

    def readTrain(self):

        # Read the training file for task A without emojis

        train_file_A = self.dir + '\\dataset\\train\\SemEval2018-T3-train-taskA.txt'

        data_fields = ['id', 'label', 'tweet'] # Define the names of the columns
        self.train_A = pd.read_csv(train_file_A, sep='\t', header=None, names=data_fields, quoting=3) # quoting=3 tells Python to ignore doubled quotes, header=None defines that the  first line of the file is not the names of the columnsv


        # Clearing training dataset and Integer Encoding

        self.train_A['tweet'] = self.train_A['tweet'].str.replace('http\S+|www.\S+', '', case=False) # Delete URLs
        self.train_A['tweet'] = self.train_A['tweet'].str.replace(r'@\S+', '', case=False) # Delete Usernames
        self.train_A['tweet'] = self.train_A['tweet'].str.replace(r'#', ' ', case=False)  # Replace hashtags with space to deal with the case where the tweet appears to be one word but is consisted by more seperated from hashtags

##        print('Average number of words per sentence: ', np.mean([len(s.split(" ")) for s in self.train_A.tweet]))

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Pre-processing
        self.pre_processing()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TF-IDF
#        self.tf_idf()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# One hot encoding
#        self.one_hot_enc()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Bi-grams
#        self.bigrams_enc()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Word2Vec
        self.Word2Vec_enc()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Doc2Vec
#        self.Doc2Vec_enc()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GloVe
#        self.GloVe_enc()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        '''
        # Create a dictionary from all the words that exist in tweets
        dict = []  # Dictionary of all the words that exist
        for i in range(0, len(self.train_A)):
            for j in range(0, len(self.train_A.iloc[i][2])):
                if self.train_A.iloc[i][2][j] not in dict:
                    dict.append(self.train_A.iloc[i][2][j])
        print(dict)

        '''




    '''
        
        # Read the training file for task A with emojis

        train_file_A_emoji = self.dir + '\\dataset\\train\\SemEval2018-T3-train-taskA_emoji.txt'

        with open(train_file_A_emoji, encoding = "utf8") as f: # 1.When using UTF-8 encoding it needs to be specified   2.No need to specify 'r': this is the default.
            self.train_A_emoji = f.readlines() # read one by one file lines
            self.train_A_emoji = [x.strip('\ufeff') for x in self.train_A_emoji] # remove the special character \ufeff, that is used in the start of a document
            self.train_A_emoji = [x.strip() for x in self.train_A_emoji] # remove whitespace characters like `\n` at the end of each line
            self.train_A_emoji = [x.split('\t') for x in self.train_A_emoji]  # create a nx3 array, n is the number of tweets

        #[print(x) for x in self.train_A_emoji] # print each line separately

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Read the training file for task A with irony hashtags emojis

        train_file_A_emoji_hash = self.dir + '\\dataset\\train\\SemEval2018-T3-train-taskA_emoji_ironyHashtags.txt'

        with open(train_file_A_emoji_hash, encoding = "utf8") as f: # 1.When using UTF-8 encoding it needs to be specified   2.No need to specify 'r': this is the default.
            self.train_A_emoji_hash = f.readlines() # read one by one file lines
            self.train_A_emoji_hash = [x.strip('\ufeff') for x in self.train_A_emoji_hash] # remove the special character \ufeff, that is used in the start of a document
            self.train_A_emoji_hash = [x.strip() for x in self.train_A_emoji_hash] # remove whitespace characters like `\n` at the end of each line
            self.train_A_emoji_hash = [x.split('\t') for x in self.train_A_emoji_hash]  # create a nx3 array, n is the number of tweets

        #[print(x) for x in self.train_A_emoji_hash] # print each line separately

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Read the training file for task B without emojis

        train_file_B = self.dir + '\\dataset\\train\\SemEval2018-T3-train-taskB.txt'

        with open(train_file_B, encoding = "utf8") as f: # 1.When using UTF-8 encoding it needs to be specified   2.No need to specify 'r': this is the default.
            self.train_B = f.readlines() # read one by one file lines
            self.train_B = [x.strip('\ufeff') for x in self.train_B] # remove the special character \ufeff, that is used in the start of a document
            self.train_B = [x.strip() for x in self.train_B] # remove whitespace characters like `\n` at the end of each line
            self.train_B = [x.split('\t') for x in self.train_B]  # create a nx3 array, n is the number of tweets

        #[print(x) for x in self.train_B] # print each line separately

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Read the training file for task B with emojis

        train_file_B_emoji = self.dir + '\\dataset\\train\\SemEval2018-T3-train-taskB_emoji.txt'

        with open(train_file_B_emoji, encoding = "utf8") as f: # 1.When using UTF-8 encoding it needs to be specified   2.No need to specify 'r': this is the default.
            self.train_B_emoji = f.readlines() # read one by one file lines
            self.train_B_emoji = [x.strip('\ufeff') for x in self.train_B_emoji]  # remove the special character \ufeff, that is used in the start of a document
            self.train_B_emoji = [x.strip() for x in self.train_B_emoji] # remove whitespace characters like `\n` at the end of each line
            self.train_B_emoji = [x.split('\t') for x in self.train_B_emoji]  # create a nx3 array, n is the number of tweets

        #[print(x) for x in self.train_B_emoji] # print each line separately

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Read the training file for task B with irony hashtags emojis

        train_file_B_emoji_hash = self.dir + '\\dataset\\train\\SemEval2018-T3-train-taskB_emoji_ironyHashtags.txt'

        with open(train_file_B_emoji_hash, encoding = "utf8") as f: # 1.When using UTF-8 encoding it needs to be specified   2.No need to specify 'r': this is the default.
            self.train_B_emoji_hash = f.readlines() # read one by one file lines
            self.train_B_emoji_hash = [x.strip('\ufeff') for x in self.train_B_emoji_hash]  # remove the special character \ufeff, that is used in the start of a document
            self.train_B_emoji_hash = [x.strip() for x in self.train_B_emoji_hash] # remove whitespace characters like `\n` at the end of each line
            self.train_B_emoji_hash = [x.split('\t') for x in self.train_B_emoji_hash]  # create a nx3 array, n is the number of tweets

        #[print(x) for x in self.train_B_emoji_hash] # print each line separately


    '''


    ##############################################################################################################################################################

    # Check if the dataset is imbalanced

    ##############################################################################################################################################################


    def  checkImbalance(self):



        # Checking if file A without emojis is imbalanced


        counter0 = 0
        counter1 = 0
        counter_all = 0
        for i in range(0, len(self.train_A)):
            counter_all += 1
            if(self.train_A.iloc[i][1] == 1):
                counter0 += 1
            else:
                counter1 += 1
        print('File A without emojis -> Percentage of tweets classified as 0: ' + str((counter0 / counter_all) * 100))
        print('File A without emojis -> Percentage of tweets classified as 1: ' + str((counter1 / counter_all) * 100) + '\n ----------------------------------------')

        '''


        # Checking if file A with emojis is imbalanced

        counter0 = 0;
        counter1 = 0;
        counter_all = 0;
        for i in range(0, len(self.train_A_emoji)):
            counter_all += 1
            if (self.train_A_emoji[i][1] == '1'):
                counter0 += 1
            else:
                counter1 += 1
        print('File A with emojis -> Percentage of tweets classified as 0: ' + str(counter0 / counter_all))
        print('File A with emojis -> Percentage of tweets classified as 1: ' + str(counter1 / counter_all) + '\n ----------------------------------------')



        # Checking if the file A with irony hashtags emojis is imbalanced

        counter0 = 0;
        counter1 = 0;
        counter_all = 0;
        for i in range(0, len(self.train_A_emoji_hash)):
            counter_all += 1
            if (self.train_A_emoji_hash[i][1] == '1'):
                counter0 += 1
            else:
                counter1 += 1
        print('File A with irony hashtags emojis -> Percentage of tweets classified as 0: ' + str(counter0 / counter_all))
        print('File A with irony hashtags emojis -> Percentage of tweets classified as 1: ' + str(counter1 / counter_all) + '\n ----------------------------------------')




        # Checking if file B without emojis is imbalanced

        counter0 = 0;
        counter1 = 0;
        counter_all = 0;
        for i in range(0, len(self.train_B)):
            counter_all += 1
            if (self.train_B[i][1] == '1'):
                counter0 += 1
            else:
                counter1 += 1
        print('File B without emojis -> Percentage of tweets classified as 0: ' + str(counter0 / counter_all))
        print('File B without emojis -> Percentage of tweets classified as 1: ' + str(counter1 / counter_all) + '\n ----------------------------------------')



        # Checking if file B with emojis is imbalanced

        counter0 = 0;
        counter1 = 0;
        counter_all = 0;
        for i in range(0, len(self.train_B_emoji)):
            counter_all += 1
            if (self.train_B_emoji[i][1] == '1'):
                counter0 += 1
            else:
                counter1 += 1
        print('File B with emojis -> Percentage of tweets classified as 0: ' + str(counter0 / counter_all))
        print('File B with emojis -> Percentage of tweets classified as 1: ' + str(counter1 / counter_all) + '\n ----------------------------------------')



        # Checking if the file B with irony hashtags emojis is imbalanced

        counter0 = 0;
        counter1 = 0;
        counter_all = 0;
        for i in range(0, len(self.train_B_emoji_hash)):
            counter_all += 1
            if (self.train_B_emoji_hash[i][1] == '1'):
                counter0 += 1
            else:
                counter1 += 1
        print('File B with irony hashtags emojis -> Percentage of tweets classified as 0: ' + str(counter0 / counter_all))
        print('File B with irony hashtags emojis -> Percentage of tweets classified as 1: ' + str(counter1 / counter_all) + '\n ----------------------------------------')


        '''


    ##############################################################################################################################################################

    # Read the input files for task A and B (with and without emojis)

    # input_A
    # input_A_emoji
    # input_B
    # input_B_emoji

    ##############################################################################################################################################################

    def readInput(self):

        # Read the input file for task A without emojis

        input_file_A = self.dir + '\\dataset\\input\\SemEval2018-T3_input_test_taskA.txt'

        with open(input_file_A, encoding = "utf8") as f: # 1.When using UTF-8 encoding it needs to be specified   2.No need to specify 'r': this is the default.
            self.input_A = f.readlines() # read one by one file lines
            self.input_A = [x.strip() for x in self.input_A] # remove whitespace characters like `\n` at the end of each line
        # self.input_A = [print(x) for x in self.input_A] # print each line separately



        # Read the input file for task A with emojis

        input_file_A_emoji = self.dir + '\\dataset\\input\\SemEval2018-T3_input_test_taskA_emoji.txt'

        with open(input_file_A_emoji, encoding = "utf8") as f: # 1.When using UTF-8 encoding it needs to be specified   2.No need to specify 'r': this is the default.
            self.input_A_emoji = f.readlines() # read one by one file lines
            self.input_A_emoji = [x.strip() for x in self.input_A_emoji] # remove whitespace characters like `\n` at the end of each line
        # self.input_A_emoji = [print(x) for x in self.input_A_emoji] # print each line separately



        # Read the input file for task B without emojis

        input_file_B = self.dir + '\\dataset\\input\\SemEval2018-T3_input_test_taskB.txt'

        with open(input_file_B, encoding = "utf8") as f: # 1.When using UTF-8 encoding it needs to be specified   2.No need to specify 'r': this is the default.
            self.input_B = f.readlines() # read one by one file lines
            self.input_B = [x.strip() for x in self.input_B] # remove whitespace characters like `\n` at the end of each line
        # self.input_B = [print(x) for x in self.input_B] # print each line separately



        # Read the input file for task B with emojis

        input_file_B_emoji = self.dir + '\\dataset\\input\\SemEval2018-T3_input_test_taskB_emoji.txt'

        with open(input_file_B_emoji, encoding = "utf8") as f: # 1.When using UTF-8 encoding it needs to be specified   2.No need to specify 'r': this is the default.
            self.input_B_emoji = f.readlines() # read one by one file lines
            self.input_B_emoji = [x.strip() for x in self.input_B_emoji] # remove whitespace characters like `\n` at the end of each line
        # self.input_B_emoji = [print(x) for x in self.input_B_emoji] # print each line separately