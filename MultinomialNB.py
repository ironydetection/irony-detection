import numpy as np

def Multinomial_NB(train_A, encoded_tweets):
    x = np.array(encoded_tweets)
    y = train_A['label']

    from sklearn.model_selection import cross_val_score, KFold, train_test_split

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
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

#######################################################################################################################

        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
#### NOT NECESSARY
        print(x_train)
####
        # Fit Multinomial Naive Bayes according to x, y
        # Make a prediction using the Multinomial Naive Bayes Model
        model.fit(x_train, y_train) # x : array-like, shape (n_samples, n_features)   Training vectors, where n_samples is the number of samples and n_features is the number of features.
                                    # y : array-like, shape (n_samples,)   Target values.

        y_pred = model.predict(x_test)
#######################################################################################################################

        # Your model is fit. Time to predict our output and test our training data
        print("Evaluating model...")
#        test_preds = model.predict_proba(x_test)

        from sklearn.metrics import roc_auc_score

#        print("test pred: ", test_preds)

        roc = roc_auc_score(y_test, y_pred)
#        for j in range(0, len(test_preds)):
#            print("This is y_test: ", y_test.iloc[j])
#            print("This is test_preds: ", test_preds[j][y_test.iloc[j]])
#            roc = roc_auc_score(y_test.iloc[j], test_preds[j][y_test.iloc[j]]) # Take the value of y_test - 0 or 1 - and in the 2 dimensional array test_preds choose the probability that corresponds to the specific value of y_test (the first column of test_preds is the probability of the prediction being 0 and the second for 1)

#        scores = model.evaluate(x_test, y_test)
 #       print(scores)

        # Print your model summary
#        print(model.summary())

        # Print your ROC-AUC score for your kfold, and the running score average
        print('ROC: ', roc)
        av_roc += roc
        print('Continued Avg: ', av_roc / count)

#######################################################################################################################

        y_pred = (y_pred > 0.5)

        # Creating the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
#        print(cm)
        precision += cm[0][0] / (cm[0][0] + cm[0][1])
        accuracy += (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        recall += cm[0][0] / (cm[0][0] + cm[1][0])
        temRecall = cm[0][0] / (cm[0][0] + cm[1][0]) # Calculate recall for the current iteration to calculate f1score
        tempPrecision = cm[0][0] / (cm[0][0] + cm[0][1]) # Calculate precision for the current iteration to calculate f1score
        f1score += 2 / ((1 / temRecall) + (1 / tempPrecision))

    # Print average of metrics
    print("Average Precision: ", precision / 10)
    print("Average Accuracy: ", accuracy / 10)
    print("Average Recall: ", recall / 10)
    print("Average F1-score: ", f1score / 10)

    # Print your final average ROC-AUC score and organize your models predictions in a dataframe
    print('Average ROC:', av_roc / 10)




    '''
    #    print("Number of mislabeled points out of a total %d points : %d" % (len(train_A), (y != y_pred).sum()))
        sum = 0
        for i in range(0, len(train_A)):
            if(train_A.iloc[i][1] != y_pred[i]):
                sum += 1
        print("Number of mislabeled points out of a total %d points : %d" % (len(train_A), sum))
    '''