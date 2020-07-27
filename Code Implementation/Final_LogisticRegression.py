import Final_ClassRead # Reads the input and the training sets
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def Logistic_Regression(train_A, train_words_of_tweets, train_extra_features, test_words_of_tweets, test_A, test_extra_features):
    reading = Final_ClassRead.Reader()  # Import the Final_ClassRead.py file, to get the encoding

    x_train = np.array(train_words_of_tweets)
    y_train = train_A['label']

    x_test = np.array(test_words_of_tweets)
    y_test = test_A['label']

    # This indexs your train and test data for your cross validation and sorts them in random order, since we used shuffle equals True
    x_train, x_test = reading.get_enc(x_train, 1, y_train, train_extra_features), reading.get_enc(x_test, 0, y_test, test_extra_features)



#######################################################################################################################

    classifier = LogisticRegression(solver='liblinear', C=0.1)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

#######################################################################################################################

    # Your model is fit. Time to predict our output and test our training data
    print("Evaluating model...")
    roc = roc_auc_score(y_test, y_pred)

    # Print your ROC-AUC score for your kfold, and the running score average
    print('ROC: ', roc)

#######################################################################################################################

    y_pred = (y_pred > 0.5)

    # Creating the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    # Print average of metrics
    print("Precision: ", precision)
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("F1-score: ", f1score)