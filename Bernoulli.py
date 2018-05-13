import ClassRead # Reads the input and the training sets
import numpy as np

def BernoulliClass(train_A, words_of_tweets):
    reading = ClassRead.Reader()  # Import the ClassRead.py file, to get the encoding

    x = np.array(words_of_tweets)
    y = train_A['label']

    from sklearn.model_selection import KFold

    # Initialize the roc-auc score running average list
    # Initialize a count to print the number of folds
    # Initialize metrics to print their average
    av_roc = 0.
    count = 0
    precision = 0
    accuracy = 0
    recall = 0
    f1score = 0

    # Initialize your 10 - cross vailidation
    # Set shuffle equals True to randomize your splits on your training data
    kf = KFold(n_splits=10, random_state=41, shuffle=True)

    # Set up for loop to run for the number of cross vals you defined in your parameter
    for train_index, test_index in kf.split(x):
        count += 1
        print('Fold #: ', count)

        # This indexs your train and test data for your cross validation and sorts them in random order, since we used shuffle equals True
        x_train, x_test = reading.get_enc(x[train_index]), reading.get_enc(x[test_index])
        y_train, y_test = y[train_index], y[test_index]

#######################################################################################################################

        from sklearn.naive_bayes import BernoulliNB

        classifier = BernoulliNB()
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

#######################################################################################################################

        # Your model is fit. Time to predict our output and test our training data
        print("Evaluating model...")

        from sklearn.metrics import roc_auc_score

        roc = roc_auc_score(y_test, y_pred)

        # Print your ROC-AUC score for your kfold, and the running score average
        print('ROC: ', roc)
        av_roc += roc
        print('Continued Avg: ', av_roc / count)

#######################################################################################################################

        y_pred = (y_pred > 0.5)

        # Creating the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        precision += cm[0][0] / (cm[0][0] + cm[0][1])
        accuracy += (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        recall += cm[0][0] / (cm[0][0] + cm[1][0])
        temRecall = cm[0][0] / (cm[0][0] + cm[1][0])  # Calculate recall for the current iteration to calculate f1score
        tempPrecision = cm[0][0] / (cm[0][0] + cm[0][1])  # Calculate precision for the current iteration to calculate f1score
        f1score += 2 / ((1 / temRecall) + (1 / tempPrecision))

    # Print average of metrics
    print("Average Precision: ", precision / 10)
    print("Average Accuracy: ", accuracy / 10)
    print("Average Recall: ", recall / 10)
    print("Average F1-score: ", f1score / 10)

    # Print your final average ROC-AUC score and organize your models predictions in a dataframe
    print('Average ROC:', av_roc / 10)