import os.path
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from nltk.util import ngrams
import gensim



class Reader:

    dir = os.getcwd() # Gets the current working directory

    words_of_tweets = [] # Saves all the tweet cleared from stop-words, stemmed and tokenized

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


###############################################################################################################################################
###############################################################################################################################################

    # Select the proper encoding
    def get_enc(self, x_enc):
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TF-IDF
        encoded_tweets = self.tf_idf(x_enc)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# One hot encoding
#        encoded_tweets = self.one_hot_enc(x_enc)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Bi-grams
#        encoded_tweets = self.bigrams_enc(x_enc)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Word2Vec
#        encoded_tweets = self.Word2Vec_enc(x_enc)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Doc2Vec
#        encoded_tweets = self.Doc2Vec_enc(x_enc)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# GloVe
#        encoded_tweets = self.GloVe_enc(x_enc)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        print(encoded_tweets)
        return encoded_tweets

###############################################################################################################################################
###############################################################################################################################################


    # Create a dictionary for one hot encoding and encode with one hot encoding
    def one_hot_enc(self, x_enc):
        word_dict = []  # Dictionary of all the words that exist

        # Create a dictionary from all the words that exist in tweets
        print(len(x_enc))
        for i in range(0, len(x_enc)):
            for j in range(0, len(x_enc[i])):
                if x_enc[i][j] not in word_dict:
                    word_dict.append(x_enc[i][j])

        tweets_size = len(x_enc)  # Rows of the array encoded_tweets
        vocab_size = len(word_dict)  # Columns of the array encoded_tweets

        # Creates a list with rows representing tweets and sized as 'tweets_size' and columns representing unique vocabulary of tweets and sized as 'vocab_size', all set to 0
        encoded_tweets = [[0 for x in range(vocab_size)] for y in range(tweets_size)]

        # Set 1 to the encoded_tweets list on the indexes that the words of a specific tweet matches the words of the dictionary
        for y in range(0, tweets_size):
            for x in range(0, vocab_size):
                for a in range(0, len(x_enc[y])):
                    if x_enc[y][a] == word_dict[x]:
                        encoded_tweets[y][x] = 1

        return encoded_tweets


###############################################################################################################################################
###############################################################################################################################################


    # TF-IDF and statistics
    def tf_idf(self, x_enc):
        # Create the tokenizer used for TF-IDF
        t = Tokenizer()

        # fit the tokenizer on the documents
        t.fit_on_texts(x_enc)

        # summarize what was learned
#        print(t.word_counts) # A dictionary of words and their counts
        # print(t.word_docs) # Similar to word_counts - output structure is changed. An integer count of the total number of documents that were used to fit the Tokenizer
        # print(t.document_count) # A dictionary of words and how many documents each appeared in
        # print(t.word_index) # A dictionary of words and their uniquely assigned integers

        # integer encode documents using TF-IDF
        encoded_tweets = t.texts_to_matrix(x_enc, mode='tfidf')
#        print(encoded_tweets)
        print(encoded_tweets.shape)
        return encoded_tweets


###############################################################################################################################################
###############################################################################################################################################


    def bigrams_enc(self, x_enc):
        bigrams = []  # Bi-grams of all tweets

        # Use the pre-processing done above
        for y in range(0, len(x_enc)):
            bigrams.append(list(ngrams(x_enc[y], 2)))
        #for y in range(0, len(self.bigrams)):
        #    print(self.bigrams[y])

        diction = []  # The dictionary of bigrams

        tweets_size = len(x_enc)  # Rows of the array

        # Create the dictionary of bigrams
        for i in range(0, tweets_size):
            for j in range(0, len(bigrams[i])):
                if bigrams[i][j] not in diction:
                    diction.append(bigrams[i][j])
#        print(diction)

        vocab_size = len(diction)  # Columns of the array

        # Use One Hot Encoding on the created bigrams
        # Creates a list with rows representing tweets and sized as 'tweets_size' and columns representing unique vocabulary of tweets and sized as 'vocab_size', all set to 0
        encoded_tweets = [[0 for x in range(vocab_size)] for y in range(tweets_size)]

        # Set 1 to the ngram list on the indexes that the words of a specific tweet matches the words of the dictionary
        for y in range(0, tweets_size):
            for x in range(0, vocab_size):
                for a in range(0, len(bigrams[y])):
                    # print("len of words: ", tweets_size)
                    if bigrams[y][a] == diction[x]:
                        ##print("got another: ", y, "from: ", tweets_size)
                        encoded_tweets[y][x] = 1

        # for y in range(0, tweets_size):
        # print(encoded_tweets[y])

        # CHECK THAT THE ENCODING IS INDEED CORRECT
#        for i in range(0, len(encoded_tweets[12])):
#            if encoded_tweets[12][i] == 1:
#                print("Diction 12: ", diction[i])
#        print("Bigram encod 12: ", encoded_tweets[12])

        return encoded_tweets


###############################################################################################################################################
###############################################################################################################################################


    def Word2Vec_enc(self, x_enc):
        from gensim.models import Word2Vec

        encoded_tweets = self.labelizeTweets(x_enc, 'TRAIN')
        print(encoded_tweets[0])

        # sg: CBOW if 0, skip-gram if 1
        # ‘min_count’ is for neglecting infrequent words.
        # negative (int) – If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
        # window: number of words accounted for each context( if the window size is 3, 3 word in the left neighorhood and 3 word in the right neighborhood are considered)
        model = Word2Vec(size=7992, min_count=0)
        model.build_vocab([x.words for x in encoded_tweets])
#        model.init_sims(replace=True)  # To make the model memory efficient
        # total_examples (int) – Count of sentences.
        print("This is the length of encoded_tweets: ", len(encoded_tweets))
        model.train([x.words for x in encoded_tweets], total_examples=len(encoded_tweets), epochs=10)


#        print(model.most_similar('woman'))
        print(model.wv.syn0.shape)


        # Save model locally to reduce time of training the model again
        #model.save('word2vec_model')
        #model = Word2Vec.load('word2vec_model')

        from sklearn.feature_extraction.text import TfidfVectorizer

        print('building tf-idf matrix ...')
        # min_df : float in range [0.0, 1.0] or int, default=1
        # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
        # This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents,
        # integer absolute counts. This parameter is ignored if vocabulary is not None.
        vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=7)
        matrix = vectorizer.fit_transform([x.words for x in encoded_tweets])
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        print('vocab size :', len(tfidf))

        from sklearn.preprocessing import scale
        train_vecs_w2v = np.concatenate([self.buildWordVector(model, z, 7992, tfidf) for z in map(lambda x: x.words, encoded_tweets)])
        encoded_tweets = scale(train_vecs_w2v)

        return encoded_tweets



    def buildWordVector(self, model, tokens, size, tfidf):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in tokens:
            try:
                vec += model[word].reshape((1, size)) * tfidf[word]
                count += 1.
            except KeyError:  # handling the case where the token is not
                # in the corpus. useful for testing.
                continue
        if count != 0:
            vec /= count
        return vec



    def labelizeTweets(self, tweets, label_type):
        LabeledSentence = gensim.models.doc2vec.LabeledSentence

        labelized = []
        for i, v in enumerate(tweets):
            label = '%s_%s' % (label_type, i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

###############################################################################################################################################
###############################################################################################################################################


    def Doc2Vec_enc(self, x_enc):
        from gensim.models import Doc2Vec

        encoded_tweets = self.labelizeTweets(x_enc, 'TRAIN')

        # dm: DBOW if 0, distributed-memory if 1
        # window: number of words accounted for each context( if the window size is 3, 3 word in the left neighorhood and 3 word in the right neighborhood are considered)
#        model = Doc2Vec(documents=x_enc, dm=1, size=100, window=3, min_count=1, iter=10, workers=Pool()._processes)
        model = Doc2Vec(vector_size=7992, min_count=0)
#        model.init_sims(replace=True)

        from sklearn import utils

        model.build_vocab([x for x in encoded_tweets])
        model.train(utils.shuffle([x for x in encoded_tweets]), total_examples=len(encoded_tweets), epochs=10)

        # printing the vector of document at index 1 in docLabels
        print(model.docvecs[len(x_enc)-1])
        print(model['TRAIN_0'])

        #for docvec in model.docvecs:
            #encoded_tweets = docvec
            #print(encoded_tweets)

        print(len(x_enc))
        for i in range(0, len(x_enc)):
            prefix_train_pos = 'TRAIN_' + str(i)
            encoded_tweets[i] = model.docvecs[prefix_train_pos]

        print(encoded_tweets)
        return encoded_tweets



###############################################################################################################################################
###############################################################################################################################################


    def GloVe_enc(self, x_enc):
        from glove import Corpus, Glove
        from multiprocessing import Pool

        encoded_tweets = []  # Different encoding of tweets (One Hot Encoding, TF-IDF, One hot encoding of ngrams)

        corpus = Corpus()
        corpus.fit(x_enc, window=3)  # window parameter denotes the distance of context
        glove = Glove(no_components=100, learning_rate=0.05)

        # matrix: co - occurence matrix of the corpus
        # no_threads: number of training threads
        glove.fit(matrix=corpus.matrix, epochs=30, no_threads=Pool()._processes, verbose=True)
        glove.add_dictionary(corpus.dictionary)  # supply a word-id dictionary to allow similarity queries

        return encoded_tweets



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

#        print('Average number of words per sentence: ', np.mean([len(s.split(" ")) for s in self.train_A.tweet]))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Pre-processing
        self.pre_processing()

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