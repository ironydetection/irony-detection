import Final_ClassRead # Reads the input and the training sets
import Final_Conv1D # Implements Conv1D classification
import Final_NeuralNets # Implements a Neural Net
import Final_NaiveBayes # Implements Naive Bayes Classification
import Final_SVM # Implements SVM classification
import Final_Bernoulli # Implements Bernoulli classification
import Final_LogisticRegression # Implements Logistic Regression classification
import Final_KNeighbors # Implements K-Neighbors classification
import Final_MultinomialNB # Implements Multinomial NB classification
import Final_VotingEnsembles # Implements Voting Ensembles classification
import Final_LSTM # Implements LSTM classification



##############################################################################################################################################################
##############################################################################################################################################################

                                                                    # Main

##############################################################################################################################################################
##############################################################################################################################################################


reading = Final_ClassRead.Reader() # Import the ClassRead.py file, that reads the input and the training sets


##############################################################################################################################################################

# Read input and training file, check if the dataset is imbalanced

##############################################################################################################################################################


reading.readTrain()
#reading.checkImbalance()


##############################################################################################################################################################

# Call a Neural Net to predict irony and evaluate the outcome

##############################################################################################################################################################


#Final_NeuralNets.neural(reading.train_A, reading.train_words_of_tweets, reading.train_extra_features, reading.test_words_of_tweets, reading.test_A, reading.test_extra_features)


##############################################################################################################################################################

# Call Naive Bayes classification for Irony Detection

##############################################################################################################################################################


#Final_NaiveBayes.Bayes(reading.train_A, reading.train_words_of_tweets, reading.train_extra_features, reading.test_words_of_tweets, reading.test_A, reading.test_extra_features)


##############################################################################################################################################################

# Call SVM classification for Irony Detection

##############################################################################################################################################################


#Final_SVM.final_svm(reading.train_A, reading.train_words_of_tweets, reading.train_extra_features, reading.test_words_of_tweets, reading.test_A, reading.test_extra_features)


##############################################################################################################################################################

# Call Bernoulli to predict irony and evaluate the outcome

##############################################################################################################################################################


#Final_Bernoulli.BernoulliClass(reading.train_A, reading.train_words_of_tweets, reading.train_extra_features, reading.test_words_of_tweets, reading.test_A, reading.test_extra_features)


##############################################################################################################################################################

# Call Logistic Regression to predict irony and evaluate the outcome

##############################################################################################################################################################


#Final_LogisticRegression.Logistic_Regression(reading.train_A, reading.train_words_of_tweets, reading.train_extra_features, reading.test_words_of_tweets, reading.test_A, reading.test_extra_features)


##############################################################################################################################################################

# Call K-Neighbors to predict irony and evaluate the outcome

##############################################################################################################################################################


#Final_KNeighbors.K_Neighbors(reading.train_A, reading.train_words_of_tweets, reading.train_extra_features, reading.test_words_of_tweets, reading.test_A, reading.test_extra_features)


##############################################################################################################################################################

# Call Multinomial Naive Bayes to predict irony and evaluate the outcome

##############################################################################################################################################################


#Final_MultinomialNB.Multinomial_NB(reading.train_A, reading.train_words_of_tweets, reading.train_extra_features, reading.test_words_of_tweets, reading.test_A, reading.test_extra_features)


##############################################################################################################################################################

# Call Voting Ensembles, using various algorithms, to predict irony and evaluate the outcome

##############################################################################################################################################################


#Final_VotingEnsembles.Voting_Ensembles(reading.train_A, reading.train_words_of_tweets, reading.train_extra_features, reading.test_words_of_tweets, reading.test_A, reading.test_extra_features)


##############################################################################################################################################################

# Call LSTM, using various algorithms, to predict irony and evaluate the outcome

##############################################################################################################################################################


#Final_LSTM.lstm(reading.train_A, reading.train_words_of_tweets, reading.train_extra_features, reading.test_words_of_tweets, reading.test_A, reading.test_extra_features)


##############################################################################################################################################################

# Call Conv1D, using various algorithms, to predict irony and evaluate the outcome

##############################################################################################################################################################


#Final_Conv1D.conv1d(reading.train_A, reading.train_words_of_tweets, reading.train_extra_features, reading.test_words_of_tweets, reading.test_A, reading.test_extra_features)


##############################################################################################################################################################
