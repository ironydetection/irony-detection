# Extracting customer feedback from social networks - Irony detection in tweets

## Thesis Description
Extensive study of machine learning algorithms and data mining methods in view of detecting ironic text excerpts. The code is in Python and it features Data Cleaning, Natural language processing, Feature Engineering (extraction-selection), Machine Learning, Data Visualization, Deep learning (Tensorflow), and Dimensionality reduction.

----------------------------------------------------------------------------------------------------------------------------------------

## Backend
Tensorflow (optionally - with GPU support)

----------------------------------------------------------------------------------------------------------------------------------------

## Necessary libraries to run the code implementation
1. matplotlib 
1. scipy 
1. wordcloud
1. emoji
1. re
1. numpy
1. sklearn
1. keras
1. ntlk
1. string
1. gensim
1. pandas
1. os
1. pathlib

----------------------------------------------------------------------------------------------------------------------------------------

  ## How to run the program
  There are two versions of each file, except from files ‘EmoticonDetector.py’ and ‘Feature_Extraction.py’. The first version implements the development stage models, in which 10-fold-cross-validation is applied, while the second version implements the final predicting models that are trained in the whole dataset in order to predict the label on the gold set. The second version files have the prefix ‘FINAL’ followed by the name of the corresponding file of the first version. An example of a first version file is ‘VotingEnsembles’, while the corresponding second version file is named as ‘Final_VotingEnsembles’.
  
----------------------------------------------------------------------------------------------------------------------------------------

  Για να τρέξουμε τα μοντέλα που υλοποιούν την πρώτη εκδοχή τρέχουμε το αρχείο ‘main.py’. Για τα αρχεία που υλοποιούν την **πρώτη εκδοχή**, όλη η διαχείριση γίνεται στο αρχείο ‘main.py’.  Πιο συγκεκριμένα, στο αρχείο υπάρχουν σε σχόλια όλα τα μοντέλα σε συνδυασμό με όλους τους αλγορίθμους κωδικοποίησης δεδομένων και επιλογής χαρακτηριστικών. Έτσι για την επιλογή ενός μοντέλου με συγκεκριμένο συνδυασμό αλγορίθμων κωδικοποίησης δεδομένων και επιλογής χαρακτηριστικών βρίσκουμε πρώτα την κατηγορία του αλγορίθμου μηχανικής μάθησης που θέλουμε να χρησιμοποιήσουμε και έπειτα κοιτάμε το τέταρτο και το πέμπτο όρισμα, των οποίων οι τιμές φαίνονται παρακάτω στους δύο πίνακες. Το τέταρτο όρισμα ορίζει ποιος αλγόριθμος επιλογής χαρακτηριστικών θα χρησιμοποιηθεί και παίρνει τιμές ‘7’ για την χρήση του ‘Univariate Selection’, ‘8’ για την χρήση του ‘Recursive Feature Elimination’, ‘9’ για τη χρήση του ‘PCA’, ‘10’ για τη χρήση του ‘Truncated SVD’ και ‘11’ για τη χρήση του ‘Feature Improtance’. Το πέμπτο όρισμα ορίζει τον αλγόριθμο κωδικοποίησης δεδομένων που θα χρησιμοποιηθεί και παίρνει τιμές ‘1’ για τη χρήση του ‘TF-IDF’, ‘2’ για τη χρήση του ‘One-Hot-Encoding’, ‘3’ για τη χρήση των ‘Bigrams με One-Hot-Encoding’, ‘4’ για τη χρήση του ‘word2vec’, ‘5’ για τη χρήση του ‘’doc2vec και ‘6’ για τη χρήση του ‘GloVe’. **Ωστόσο, επειδή δεν είναι πολύ αποδοτικός αυτός ο τρόπος επιλογής αλγορίθμων, για την άμεση εύρεση του κατάλληλου συνδυασμού αλγορίθμου κωδικοποίησης δεδομένων και επιλογής χαρακτηριστικών μπορούμε να δούμε το έκτο όρισμα που αναφέρει τα ονόματα των παραπάνω και είναι το όνομα του αρχείου στο οποίο αποθηκεύονται διάφορα δεδομένα που παράγονται κατά την λειτουργία της διαδικασίας.**
  

Text Encoding Algorithms | Values of fifth hyperparameter  
------------------------ | ----------------------- 
TF-IDF  |  1 
One-Hot-Encoding  |  2 
Bigrams  |  3 
word2vec  |  4 
doc2vec  |  5
GloVe  |  6


Feature Selection Algorithms | Values of fourth hyperparameter 
----------------------------------- | ------------------------
Univariate Selection| 7
Recursive Feature Elimination | 8  
PCA | 9  
Truncated SVD | 10
Feature Importance | 11  

  **To run the models of the development stage (first file version), it is mandatory to include 10 folders in the project path that will be utilized to save the results produced by each model. The file names are listed below and each folder corresponds to each machine learning algorithm used. Folder names can be changed in the "main.py" file (sixth hyperparameter). Folders can be empty, as the necessary files are created automatically.**
 
**Necessary folders to run the program**
1. Bayes
1. Bernoulli 
1. Conv1D
1. KNeighbors
1. LogisticRegression
1. LSTM
1. MultinomialNB
1. NeuralNets
1. SVM
1. VotingEnsembles

----------------------------------------------------------------------------------------------------------------------------------------

  Για τα αρχεία που υλοποιούν την **δεύτερη εκδοχή εκδοχή**, η διαχείριση γίνεται από δύο αρχεία, το ‘Final_Main.py’ και το ‘Final_ClassRead’. Στο αρχείο ‘Final_Main.py’ υπάρχουν σε σχόλια όλοι οι αλγόριθμοι μηχανικής μάθησης που υλοποιήθηκαν, οπότε για να επιλέξουμε κάποιον αλγόριθμο μηχανικής μάθησης τον αφαιρούμε από τα σχόλια. Στο αρχείο ‘Final_ClassRead’ γίνεται η επιλογή του αλγορίθμου κωδικοποίησης δεδομένων και επιλογής χαρακτηριστικών. Αυτό γίνεται στην συνάρτηση get_enc, στην οποία στο πρώτο μέρος υπάρχουν σε σχόλια όλοι οι αλγόριθμοι κωδικοποίησης δεδομένων που υλοποιήθηκαν, ενώ στο δεύτερο μέρος υπάρχουν οι αλγόριθμοι επιλογής χαρακτηριστικών, πάλι σε σχόλια. Έτσι, από εδώ επιλέγουμε όποιο συνδυασμό θέλουμε αφαιρώντας από τα σχόλια τους αλγορίθμους που θέλουμε να χρησιμοποιήσουμε. Τέλος, για να τρέξουμε τα μοντέλα που υλοποιούν την δεύτερη εκδοχή τρέχουμε το αρχείο ‘Final_Main.py’.
