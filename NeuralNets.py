import numpy as np

def neural(train_A, encoded_tweets):
    x = np.array(encoded_tweets)
    y = train_A['label']

    #x = [None] * 140 # Initialize the array with the maximum number of words that can exist in a tweet
    #a = numpy.empty(n, dtype=object) # This creates an array of length n that can store objects.It cant be resized or appended to. In particular, it doesnt waste space by padding its length.

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
##    sc = StandardScaler()
##    x = sc.fit_transform(x)
##    x = sc.fit_transform(encoded_tweets)
    #x_train = sc.fit_transform(x_train)
    #x_test = sc.transform(x_test)

##    from keras.utils import np_utils
##    y = np_utils.to_categorical(y)
    #y_train = np_utils.to_categorical(y_train)
    #y_test = np_utils.to_categorical(y_test)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Importing the Keras libraries and packages
        from keras.models import Sequential
        from keras.layers import Dense

        # Initializing Neural Network
        classifier = Sequential()

        # Adding the input layer and the first hidden layer (12 neurons)
        classifier.add(Dense(80, kernel_initializer='uniform', activation='relu', input_dim=7992))  # changed input_dim from 11 to 2
        # Adding the second hidden layer (8 neurons)
        classifier.add(Dense(80, kernel_initializer='uniform', activation='relu'))
        # Adding the third hidden layer (8 neurons)
        classifier.add(Dense(80, kernel_initializer='uniform', activation='relu'))

        # from keras.layers import Flatten
        # classifier.add(Flatten())

        # Adding the output layer with 1 output
        classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

        # Compiling Neural Network
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fitting our model
        classifier.fit(x_train, y_train, batch_size=10, epochs=20)
        #fit(X_nn_train, y_nn_train, epochs=15, batch_size=30, verbose=0)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Your model is fit. Time to predict our output and test our training data
        print("Evaluating model...")
        test_preds = classifier.predict_proba(x_test, verbose=0)

        from sklearn.metrics import roc_auc_score

        roc = roc_auc_score(y_test, test_preds)
        scores = classifier.evaluate(x_test, y_test)
        print(scores)

        # Print your model summary
        print(classifier.summary())

        # Print your ROC-AUC score for your kfold, and the running score average
        print('ROC: ', roc)
        av_roc += roc
        print('Continued Avg: ', av_roc / count)


        # Predicting the Test set results
        y_pred = classifier.predict(x_test)
        y_pred = (y_pred > 0.5)

        # Creating the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
#        print(cm)
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
    #predict_proba_all = pd.DataFrame(classifier.predict_proba(encoded_tweets, verbose=0))
    #return pd.DataFrame(predict_proba_all)