import ClassRead # Reads the input and the training sets
import NeuralNets # Implements a Neural Net
import NaiveBayes # Implements Naive Bayes Classification
import SVM # Implements SVM classification
import Bernoulli # Implements Bernoulli classification
import LogisticRegression # Implements LogisticRegression classification
import KNeighbors # Implements KNeighbors classification
import MultinomialNB # Implements MultinomialNB classification
import VotingEnsembles # Implements VotingEnsembles classification
import LSTM # Implements  LSTM classification
import Conv1D # Implements Conv1D classification
import os.path


##############################################################################################################################################################
##############################################################################################################################################################

                                                                    # Main

##############################################################################################################################################################
##############################################################################################################################################################


reading = ClassRead.Reader() # Import the ClassRead.py file, that reads the input and the training sets
dir = os.getcwd() # Gets the current working directory


##############################################################################################################################################################

# Read input and training file, check if the dataset is imbalanced

##############################################################################################################################################################


reading.readTrain()
#reading.checkImbalance()


##############################################################################################################################################################

# Call all algorithms with different combinations of feature selection and encoding

##############################################################################################################################################################







##############################################################################################################################################################

# Call SVM classification for Irony Detection

##############################################################################################################################################################

'''

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\SVM\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\SVM\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\SVM\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\SVM\\SVD + TF-IDF.txt')
print('DONE FILE 4')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\SVM\\SVD + One-Hot.txt')
print('DONE FILE 5')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\SVM\\SVD + Bigrams.txt')
print('DONE FILE 6')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\SVM\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\SVM\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\SVM\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\SVM\\PCA + TF-IDF.txt')
print('DONE FILE 10')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\SVM\\PCA + One-Hot.txt')
print('DONE FILE 11')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\SVM\\PCA + Bigrams.txt')
print('DONE FILE 12')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\SVM\\word2vec.txt')
print('DONE FILE 13')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\SVM\\GloVe.txt')
print('DONE FILE 14')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\SVM\\doc2vec.txt')
print('DONE FILE 15')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\SVM\\TF-IDF.txt')
print('DONE FILE 16')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\SVM\\One-Hot.txt')
print('DONE FILE 17')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\SVM\\Bigrams.txt')
print('DONE FILE 18')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\SVM\\RFE + TF-IDF.txt')
print('DONE FILE 19')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\SVM\\RFE + One-Hot.txt')
print('DONE FILE 20')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\SVM\\RFE + Bigrams.txt')
print('DONE FILE 21')

'''

##############################################################################################################################################################

# Call Naive Bayes classification for Irony Detection

##############################################################################################################################################################

'''
NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\Bayes\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\Bayes\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\Bayes\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\Bayes\\SVD + TF-IDF.txt')
print('DONE FILE 4')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\Bayes\\SVD + One-Hot.txt')
print('DONE FILE 5')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\Bayes\\SVD + Bigrams.txt')
print('DONE FILE 6')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\Bayes\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\Bayes\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\Bayes\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\Bayes\\PCA + TF-IDF.txt')
print('DONE FILE 10')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\Bayes\\PCA + One-Hot.txt')
print('DONE FILE 11')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\Bayes\\PCA + Bigrams.txt')
print('DONE FILE 12')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\Bayes\\word2vec.txt')
print('DONE FILE 13')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\Bayes\\GloVe.txt')
print('DONE FILE 14')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\Bayes\\doc2vec.txt')
print('DONE FILE 15')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\Bayes\\TF-IDF.txt')
print('DONE FILE 16')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\Bayes\\One-Hot.txt')
print('DONE FILE 17')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\Bayes\\Bigrams.txt')
print('DONE FILE 18')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\Bayes\\RFE + TF-IDF.txt')
print('DONE FILE 19')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\Bayes\\RFE + One-Hot.txt')
print('DONE FILE 20')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\Bayes\\RFE + Bigrams.txt')
print('DONE FILE 21')

'''



##############################################################################################################################################################

# Call LSTM classification for Irony Detection

##############################################################################################################################################################

'''
LSTM.lstm(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\LSTM\\word2vec.txt')
LSTM.lstm(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\LSTM\\doc2vec.txt')
LSTM.lstm(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\LSTM\\GloVe.txt')
'''



##############################################################################################################################################################

# Call Conv1D classification for Irony Detection

##############################################################################################################################################################

'''
Conv1D.conv1d_class(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\Conv1D\\word2vec.txt')
Conv1D.conv1d_class(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\Conv1D\\doc2vec.txt')
Conv1D.conv1d_class(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\Conv1D\\GloVe.txt')
'''



##############################################################################################################################################################

# Call a Neural Net to predict irony and evaluate the outcome

##############################################################################################################################################################

'''
NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\NeuralNets\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\NeuralNets\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\NeuralNets\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\NeuralNets\\SVD + TF-IDF.txt')
print('DONE FILE 4')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\NeuralNets\\SVD + One-Hot.txt')
print('DONE FILE 5')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\NeuralNets\\SVD + Bigrams.txt')
print('DONE FILE 6')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\NeuralNets\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\NeuralNets\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\NeuralNets\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\NeuralNets\\PCA + TF-IDF.txt')
print('DONE FILE 10')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\NeuralNets\\PCA + One-Hot.txt')
print('DONE FILE 11')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\NeuralNets\\PCA + Bigrams.txt')
print('DONE FILE 12')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\NeuralNets\\word2vec.txt')
print('DONE FILE 13')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\NeuralNets\\GloVe.txt')
print('DONE FILE 14')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\NeuralNets\\doc2vec.txt')
print('DONE FILE 15')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\NeuralNets\\TF-IDF.txt')
print('DONE FILE 16')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\NeuralNets\\One-Hot.txt')
print('DONE FILE 17')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\NeuralNets\\Bigrams.txt')
print('DONE FILE 18')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\NeuralNets\\RFE + TF-IDF.txt')
print('DONE FILE 19')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\NeuralNets\\RFE + One-Hot.txt')
print('DONE FILE 20')

NeuralNets.neural(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\NeuralNets\\RFE + Bigrams.txt')
print('DONE FILE 21')

'''


##############################################################################################################################################################

# Call Bernoulli to predict irony and evaluate the outcome

##############################################################################################################################################################

'''
Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\Bernoulli\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\Bernoulli\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\Bernoulli\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\Bernoulli\\SVD + TF-IDF.txt')
print('DONE FILE 4')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\Bernoulli\\SVD + One-Hot.txt')
print('DONE FILE 5')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\Bernoulli\\SVD + Bigrams.txt')
print('DONE FILE 6')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\Bernoulli\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\Bernoulli\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\Bernoulli\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\Bernoulli\\PCA + TF-IDF.txt')
print('DONE FILE 10')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\Bernoulli\\PCA + One-Hot.txt')
print('DONE FILE 11')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\Bernoulli\\PCA + Bigrams.txt')
print('DONE FILE 12')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\Bernoulli\\word2vec.txt')
print('DONE FILE 13')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\Bernoulli\\GloVe.txt')
print('DONE FILE 14')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\Bernoulli\\doc2vec.txt')
print('DONE FILE 15')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\Bernoulli\\TF-IDF.txt')
print('DONE FILE 16')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\Bernoulli\\One-Hot.txt')
print('DONE FILE 17')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\Bernoulli\\Bigrams.txt')
print('DONE FILE 18')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\Bernoulli\\RFE + TF-IDF.txt')
print('DONE FILE 19')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\Bernoulli\\RFE + One-Hot.txt')
print('DONE FILE 20')

Bernoulli.BernoulliClass(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\Bernoulli\\RFE + Bigrams.txt')
print('DONE FILE 21')

'''



##############################################################################################################################################################

# Call Logistic Regression to predict irony and evaluate the outcome

##############################################################################################################################################################


'''
LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\LogisticRegression\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\LogisticRegression\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\LogisticRegression\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\LogisticRegression\\SVD + TF-IDF.txt')
print('DONE FILE 4')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\LogisticRegression\\SVD + One-Hot.txt')
print('DONE FILE 5')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\LogisticRegression\\SVD + Bigrams.txt')
print('DONE FILE 6')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\LogisticRegression\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\LogisticRegression\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\LogisticRegression\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\LogisticRegression\\PCA + TF-IDF.txt')
print('DONE FILE 10')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\LogisticRegression\\PCA + One-Hot.txt')
print('DONE FILE 11')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\LogisticRegression\\PCA + Bigrams.txt')
print('DONE FILE 12')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\LogisticRegression\\word2vec.txt')
print('DONE FILE 13')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\LogisticRegression\\GloVe.txt')
print('DONE FILE 14')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\LogisticRegression\\doc2vec.txt')
print('DONE FILE 15')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\LogisticRegression\\TF-IDF.txt')
print('DONE FILE 16')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\LogisticRegression\\One-Hot.txt')
print('DONE FILE 17')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\LogisticRegression\\Bigrams.txt')
print('DONE FILE 18')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\LogisticRegression\\RFE + TF-IDF.txt')
print('DONE FILE 19')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\LogisticRegression\\RFE + One-Hot.txt')
print('DONE FILE 20')

LogisticRegression.Logistic_Regression(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\LogisticRegression\\RFE + Bigrams.txt')
print('DONE FILE 21')

'''



##############################################################################################################################################################

# Call K-Neighbors to predict irony and evaluate the outcome

##############################################################################################################################################################


'''
KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\KNeighbors\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\KNeighbors\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\KNeighbors\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\KNeighbors\\SVD + TF-IDF.txt')
print('DONE FILE 4')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\KNeighbors\\SVD + One-Hot.txt')
print('DONE FILE 5')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\KNeighbors\\SVD + Bigrams.txt')
print('DONE FILE 6')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\KNeighbors\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\KNeighbors\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\KNeighbors\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\KNeighbors\\PCA + TF-IDF.txt')
print('DONE FILE 10')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\KNeighbors\\PCA + One-Hot.txt')
print('DONE FILE 11')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\KNeighbors\\PCA + Bigrams.txt')
print('DONE FILE 12')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\KNeighbors\\word2vec.txt')
print('DONE FILE 13')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\KNeighbors\\GloVe.txt')
print('DONE FILE 14')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\KNeighbors\\doc2vec.txt')
print('DONE FILE 15')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\KNeighbors\\TF-IDF.txt')
print('DONE FILE 16')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\KNeighbors\\One-Hot.txt')
print('DONE FILE 17')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\KNeighbors\\Bigrams.txt')
print('DONE FILE 18')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\KNeighbors\\RFE + TF-IDF.txt')
print('DONE FILE 19')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\KNeighbors\\RFE + One-Hot.txt')
print('DONE FILE 20')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\KNeighbors\\RFE + Bigrams.txt')
print('DONE FILE 21')

'''



##############################################################################################################################################################

# Call Multinomial Naive Bayes to predict irony and evaluate the outcome

##############################################################################################################################################################


'''
MultinomialNB.Multinomial_NB(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\MultinomialNB\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

MultinomialNB.Multinomial_NB(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\MultinomialNB\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

MultinomialNB.Multinomial_NB(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\MultinomialNB\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

MultinomialNB.Multinomial_NB(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\MultinomialNB\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 4')

MultinomialNB.Multinomial_NB(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\MultinomialNB\\Feature Improtance + One-Hot.txt')
print('DONE FILE 5')

MultinomialNB.Multinomial_NB(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\MultinomialNB\\Feature Improtance + Bigrams.txt')
print('DONE FILE 6')

MultinomialNB.Multinomial_NB(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\MultinomialNB\\TF-IDF.txt')
print('DONE FILE 7')

MultinomialNB.Multinomial_NB(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\MultinomialNB\\One-Hot.txt')
print('DONE FILE 8')

MultinomialNB.Multinomial_NB(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\MultinomialNB\\Bigrams.txt')
print('DONE FILE 9')
'''



##############################################################################################################################################################

# Call Voting Ensembles, using various algorithms, to predict irony and evaluate the outcome

##############################################################################################################################################################

'''
VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\VotingEnsembles\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\VotingEnsembles\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\VotingEnsembles\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\VotingEnsembles\\SVD + TF-IDF.txt')
print('DONE FILE 4')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\VotingEnsembles\\SVD + One-Hot.txt')
print('DONE FILE 5')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\VotingEnsembles\\SVD + Bigrams.txt')
print('DONE FILE 6')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\VotingEnsembles\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\VotingEnsembles\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\VotingEnsembles\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\VotingEnsembles\\PCA + TF-IDF.txt')
print('DONE FILE 10')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\VotingEnsembles\\PCA + One-Hot.txt')
print('DONE FILE 11')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\VotingEnsembles\\PCA + Bigrams.txt')
print('DONE FILE 12')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\VotingEnsembles\\word2vec.txt')
print('DONE FILE 13')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\VotingEnsembles\\GloVe.txt')
print('DONE FILE 14')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\VotingEnsembles\\doc2vec.txt')
print('DONE FILE 15')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\VotingEnsembles\\TF-IDF.txt')
print('DONE FILE 16')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\VotingEnsembles\\One-Hot.txt')
print('DONE FILE 17')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\VotingEnsembles\\Bigrams.txt')
print('DONE FILE 18')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\VotingEnsembles\\RFE + TF-IDF.txt')
print('DONE FILE 19')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\VotingEnsembles\\RFE + One-Hot.txt')
print('DONE FILE 20')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\VotingEnsembles\\RFE + Bigrams.txt')
print('DONE FILE 21')

'''