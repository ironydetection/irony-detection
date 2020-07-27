import Final_ClassRead # Reads the input and the training sets
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def Multinomial_NB(train_A, train_words_of_tweets, train_extra_features, test_words_of_tweets, test_A, test_extra_features):
    reading = Final_ClassRead.Reader()  # Import the Final_ClassRead.py file, to get the encoding

    x_train = np.array(train_words_of_tweets)
    y_train = train_A['label']

    x_test = np.array(test_words_of_tweets)
    y_test = test_A['label']

    # This indexs your train and test data for your cross validation and sorts them in random order, since we used shuffle equals True
    x_train = reading.get_enc(x_train, 1, y_train, train_extra_features)
    x_test = reading.get_enc(x_test, 0, y_test, test_extra_features)


#######################################################################################################################

    model = MultinomialNB()

    # Fit Multinomial Naive Bayes according to x, y
    # Make a prediction using the Multinomial Naive Bayes Model
    model.fit(x_train, y_train) # x : array-like, shape (n_samples, n_features)   Training vectors, where n_samples is the number of samples and n_features is the number of features.
                                # y : array-like, shape (n_samples,)   Target values.

    y_pred = model.predict(x_test)

#######################################################################################################################

    # Your model is fit. Time to predict our output and test our training data
    print("Evaluating model...")
    roc = roc_auc_score(y_test, y_pred)

#######################################################################################################################

    y_pred = (y_pred > 0.5)

    # Creating the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    # Print metrics
    # Print ROC-AUC score
    print('ROC: ', roc)
    print("Precision: ", precision)
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("F1-score: ", f1score)