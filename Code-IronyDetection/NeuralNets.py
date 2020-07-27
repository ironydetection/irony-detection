import ClassRead # Reads the input and the training sets
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.optimizers import RMSprop
from keras import callbacks
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV



def create_model():
    # Initializing Neural Network
    classifier = Sequential()

    # Adding the input layer and the first hidden layer (20 neurons)
    # input_dim changes depending on the used algorithms of data encoding and feature selection
    classifier.add(Dense(20, kernel_initializer='glorot_uniform', activation='softsign', input_dim=109, kernel_constraint=maxnorm(2)))
    classifier.add(Dropout(0.2))

    # Adding the second hidden layer (10 neurons)
    classifier.add(Dense(10, kernel_initializer='glorot_uniform', activation='softsign', kernel_constraint=maxnorm(2)))
    classifier.add(Dropout(0.2))

    # Adding the output layer with 1 output
    classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))

    optimizer = RMSprop(lr=0.001)

    # Compiling Neural Network
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier




def compute_ROC_Curve(tprs, mean_fpr, aucs):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()





def neural(train_A, words_of_tweets, extra_features, feature_selection, encoding, print_file):
    reading = ClassRead.Reader()  # Import the ClassRead.py file, to get the encoding

    x = np.array(words_of_tweets)
    y = train_A['label']

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Initialize the roc-auc score running average list
    # Initialize a count to print the number of folds
    # Initialize metrics to print their average
    av_roc = 0.
    count = 0
    precision = 0
    accuracy = 0
    recall = 0
    f1score = 0
    # Above 3 variables are used for ROC-AUC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Initialize your 10 - cross vailidation
    # Set shuffle equals True to randomize your splits on your training data
    kf = KFold(n_splits=10, random_state=41, shuffle=True)

    # Set up for loop to run for the number of cross vals you defined in your parameter
    for train_index, test_index in kf.split(x):
        count += 1
        print('Fold #: ', count)

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write('Fold #: ' + str(count) + '\n')

        # This indexs your train and test data for your cross validation and sorts them in random order, since we used shuffle equals True
        x_train, x_test = reading.get_enc(x[train_index], 1, y[train_index], train_index, extra_features, feature_selection, encoding, print_file), reading.get_enc(x[test_index], 0, y[test_index], test_index, extra_features, feature_selection, encoding, print_file)
        y_train, y_test = y[train_index], y[test_index]

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Initializing Neural Network
        classifier = Sequential()

        feature_dimensions = x_train.shape[1]
        print("second dimension (feature dimension): ", x_train.shape[1])

        # Adding the input layer and the first hidden layer (20 neurons)
        classifier.add(Dense(20, kernel_initializer='glorot_uniform', activation='softsign', input_dim=feature_dimensions, kernel_constraint=maxnorm(2)))
        classifier.add(Dropout(0.2))

        # Adding the second hidden layer (10 neurons)
        classifier.add(Dense(10, kernel_initializer='glorot_uniform', activation='softsign', kernel_constraint=maxnorm(2)))
        classifier.add(Dropout(0.2))

        # Adding the output layer with 1 output
        classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))

        optimizer = RMSprop(lr=0.001)

        # Compiling Neural Network
        classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        '''

# How to Tune Batch Size and Number of Epochs
        # create model
        model = KerasClassifier(build_fn=create_model, verbose=0)
        # define the grid search parameters
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 20, 40]
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        '''







        '''
		
		# create model
        model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=20, verbose=0)
		
# How to Tune the Training Optimization Algorithm
        # define the grid search parameters
        optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
        param_grid = dict(optimizer=optimizer)
        
        
# How to Tune Learning Rate and Momentum
        # define the grid search parameters
        learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
       # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
        param_grid = dict(learn_rate=learn_rate)
        

# How to Tune Network Weight Initialization
        # define the grid search parameters
        init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                     'he_uniform']
        param_grid = dict(init_mode=init_mode)
       

# How to Tune the Neuron Activation Function
        # define the grid search parameters
        activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
        param_grid = dict(activation=activation)
        
# How to Tune Dropout Regularization
        # define the grid search parameters
        weight_constraint = [1, 2, 3, 4, 5]
        dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
       




# How to Tune the Number of Neurons in the Hidden Layer
        # define the grid search parameters
        neurons = [1, 5, 10, 15, 20, 25, 30, 35, 40]
        param_grid = dict(neurons=neurons)

        '''

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        '''
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
        # Use only the training data set (cannot use whole data set cause it is not encoded)
        grid_result = grid.fit(x_train, y_train)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        '''


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    #    classifier = model

    #    classifier = create_model()

        callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

        # Fitting our model
        classifier.fit(x_train, y_train, batch_size=20, epochs=50)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Your model is fit. Time to predict our output and test our training data
        print("Evaluating model...")

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write("Evaluating model..." + '\n')

        test_preds = classifier.predict_proba(x_test, verbose=0)

        roc = roc_auc_score(y_test, test_preds)
        scores = classifier.evaluate(x_test, y_test)
        print(scores)

        # Print your model summary
        print(classifier.summary())

        # Print your ROC-AUC score for your kfold, and the running score average
        print('ROC: ', roc)
        av_roc += roc
        print('Continued Avg: ', av_roc / count)

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write('Scores: ' + str(scores) + '\n' + 'Classifier summary: ' + str(classifier.summary()) + '\n' + 'ROC: ' + str(roc) + '\n' + 'Continued Avg: ' + str(av_roc / count) + '\n')

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        '''
        
        # Compute ROC curve and area the curve

        fpr, tpr, thresholds = roc_curve(y_test, test_preds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (count-1, roc_auc))
        
        '''
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Predicting the Test set results
        y_pred = classifier.predict(x_test)
        y_pred = (y_pred > 0.5)

        # Creating the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write(str(cm) + '\n')

        temp_accuracy = accuracy_score(y_test, y_pred)
        temp_precision, temp_recall, temp_f1_score, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                        average='binary')

        accuracy += temp_accuracy
        precision += temp_precision
        recall += temp_recall
        f1score += temp_f1_score

        print("Accuracy: ", temp_accuracy)
        print("Precision: ", temp_precision)
        print("Recall: ", temp_recall)
        print("F1 score: ", temp_f1_score)

    # Create ROC-AUC curve
#    compute_ROC_Curve(tprs, mean_fpr, aucs)

    # Print average of metrics
    print("Average Precision: ", precision / 10)
    print("Average Accuracy: ", accuracy / 10)
    print("Average Recall: ", recall / 10)
    print("Average F1-score: ", f1score / 10)

    # Print your final average ROC-AUC score and organize your models predictions in a dataframe
    print('Average ROC:', av_roc / 10)

    with open(print_file, "a") as myfile:  # Write above print into output file
        myfile.write("Average Precision: " + str(precision / 10) + '\n' + "Average Accuracy: " + str(accuracy / 10) + '\n' + "Average Recall: " + str(recall / 10) + '\n' + "Average F1-score: " + str(f1score / 10) + '\n' + 'Average ROC:' + str(av_roc / 10) + '\n')