import sys
import csv
import gensim
import re
from gensim import corpora, models
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib





def buildAndCleanCSV(fileName,sendersArray):
    dictionariesOfSenders={}
    allWordsOfSenders={}
    row_count=0
    for sender in sendersArray:
        dictionariesOfSenders[sender]=[]

    with open(fileName+'.csv', 'rb') as csv_file:
        csv.field_size_limit(sys.maxint)
        reader = csv.reader(csv_file)
        row_count = sum(1 for row in reader)
    with open(fileName+'.csv', 'rb') as csv_file:
        csv.field_size_limit(sys.maxint)
        reader = csv.reader(csv_file)
        current_row=1
        line = next(reader, None)
        while (line):
            if current_row % 1000==0:
                print('row is: '+str(current_row)+'/'+str(row_count))
            # Remove HTML
            text = BeautifulSoup(line[1], "html.parser").get_text()
            #  Remove non-letters and words longer than 3 letters
            letters_only = re.sub("[^a-zA-Z]", " ", text)
            letters_only = ' '.join(word for word in letters_only.split() if len(word) > 2)
            #lower letters
            letters_only = letters_only.lower()
            # Convert to lower case, split into individual words
            words = letters_only.split()
            # a list, so convert the stop words to a set (faster)
            stops = set(stopwords.words("english"))
            emailStopWords = {'Additional', 'option' ,'for', 'viewing',' and', 'saving', 'the', 'attached', 'documents','http','mass','bgu','ac' ,'il', 'nm', 'php' ,'mm' ,'b' ,'cfe' ,'a','Website', 'www', 'bgu','attachments', 'view', 'npdf', 'pdf', 'alternative', 'view','save', 'additional'}
            # Remove stop words
            meaningful_words = [w for w in words if not w in stops]
            meaningful_words = [w for w in meaningful_words if not w in emailStopWords]
            # 6. Join the words back into one string separated by space and return the result.
            words_combined=" ".join(meaningful_words)
            #stemming
            #porter = nltk.PorterStemmer()
            #[porter.stem(w) for w in words_combined]
            #taking only emails with more than 50 characters to avoid non-informative emails
            if(len(words_combined)>30):
                dictionariesOfSenders[line[0]].append(words_combined.split())
            line = next(reader, None)
            current_row+=1
    sendersBOWlength=[]
    for sender in sendersArray:
        combined=[]
        for tmp in dictionariesOfSenders[sender]:
            combined=combined+tmp
        allWordsOfSenders[sender]=set(combined)
        #BOW size will be the number of different words and if its above 5000, max size will be set to 5000
        sendersBOWlength.append(len(allWordsOfSenders[sender]) if len(allWordsOfSenders[sender])<=5000 else 5000)
    #creating the bag of words
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    sendersVectorizers_featureNames={}
    sendersVectorizers_arrays={}
    sendersVectorizers_models={}
    #combine all clean emails from all
    combined_y_label=[]
    text_representation = []
    for sender in sendersArray:
        for emailWordsList in dictionariesOfSenders[sender]:
            combined_y_label.append(sender)
            text_representation.append(" ".join(emailWordsList))
    #vectorizing, we chose max_features to be the max size of the BOW texts length or if higher than 5000 the size will be set to 5000
    num_of_features=max(sendersBOWlength)
    print('num of features is '+str(num_of_features))
    sendersVectorizers= CountVectorizer(analyzer="word", \
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words=None, \
                             max_features=num_of_features)
    #reading the new prediction test part to have the same number of features
    (pred_data_test_x,pred_data_test_y)=readPredData(senderlist)
    #merging test and train for the vectorization
    merged_list=text_representation+pred_data_test_x
    #sendersVectorizers_model = sendersVectorizers.fit_transform(text_representation)
    sendersVectorizers_model = sendersVectorizers.fit_transform(merged_list)
    sendersVectorizers_arrays=sendersVectorizers_model.toarray()
    # seperate train and test
    train_index=len(text_representation)
    #test needs to be 30% of the size of the train
    test_index=int(train_index*0.3)
    splitted_train_data=sendersVectorizers_arrays[:train_index]
    splitted_test_data=sendersVectorizers_arrays[train_index:]
    #creating the random forest classifier
    train_data_for_all_x=splitted_train_data
    train_data_for_all_y=combined_y_label
    #shuffle the test arrays and choose only 30%
    c2 = list(zip(splitted_test_data, pred_data_test_y))
    random.shuffle(c2)
    splitted_test_data, pred_data_test_y = zip(*c2)
    splitted_test_data=splitted_test_data[:test_index]
    pred_data_test_y=pred_data_test_y[:test_index]
    #shuffle the arrays
    c = list(zip(train_data_for_all_x, train_data_for_all_y))
    random.shuffle(c)
    train_data_for_all_x, train_data_for_all_y = zip(*c)
    # split to train & test
    x_train, x_test, y_train, y_test = train_test_split(train_data_for_all_x, train_data_for_all_y, test_size=0.2)

    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators = 100)
    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    #
    # This may take a few minutes to run
    forest = forest.fit( x_train, y_train )
    #joblib.dump(forest, 'model.pkl', compress=9)

    y_test_pred=forest.predict(splitted_test_data)
    # Evaluate accuracy best on the test set
    print('score is: ' +str(forest.score(splitted_test_data,pred_data_test_y)))
    #the confusion matrix of the senders of the test data (for random forest):
    print(str(confusion_matrix(pred_data_test_y, y_test_pred)))
    #comparing more models:
    import matplotlib.pyplot as plt
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    # prepare configuration for cross validation test harness
    seed = 7
    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    print('done')
def readPredData(senderlist):
    dictionariesOfSenders = {}
    text_representation_senders = []
    y_labels_representation_senders=[]
    row_count = 0
    for sender in senderlist:
        dictionariesOfSenders[sender] = []
        with open(sender + '_pred.csv', 'rb') as csv_file:
            csv.field_size_limit(sys.maxint)
            reader = csv.reader(csv_file)
            line = next(reader, None)
            while (line):
                pred_email = line[0]
                # Remove HTML
                text = BeautifulSoup(pred_email, "html.parser").get_text()
                #  Remove non-letters and words longer than 3 letters
                letters_only = re.sub("[^a-zA-Z]", " ", text)
                letters_only = ' '.join(word for word in letters_only.split() if len(word) > 2)
                # lower letters
                letters_only = letters_only.lower()
                # Convert to lower case, split into individual words
                words = letters_only.split()
                # a list, so convert the stop words to a set (faster)
                stops = set(stopwords.words("english"))
                emailStopWords = {'Additional', 'option', 'for', 'viewing', ' and', 'saving', 'the', 'attached',
                                  'documents', 'http', 'mass', 'bgu', 'ac', 'il', 'nm', 'php', 'mm', 'b', 'cfe', 'a',
                                  'Website', 'www', 'bgu', 'attachments', 'view', 'npdf', 'pdf', 'alternative', 'view',
                                  'save', 'additional'}
                # Remove stop words
                meaningful_words = [w for w in words if not w in stops]
                meaningful_words = [w for w in meaningful_words if not w in emailStopWords]
                # 6. Join the words back into one string separated by space and return the result.
                words_combined = " ".join(meaningful_words)
                dictionariesOfSenders[sender].append(words_combined.split())
                line = next(reader, None)

            combined_y_label = []
            text_representation = []
            for emailWordsList in dictionariesOfSenders[sender]:
                combined_y_label.append(sender)
                text_representation.append(" ".join(emailWordsList))
            text_representation_senders=text_representation_senders+text_representation
            y_labels_representation_senders=y_labels_representation_senders+combined_y_label
    return (text_representation_senders,y_labels_representation_senders)
#main
print('part 2')

#nltk.download()
senderlist = ['dean@bgu.ac.il', 'peler@exchange.bgu.ac.il', 'bitahon@bgu.ac.il', 'career@bgu.ac.il',
              'shanigu@bgu.ac.il']
#we decided to use the code from part two with the trained data
#because we needed the features number to be the same as in the training data
#so we used the new predicted data as the test data.
#we also used this part methods because we wanted the data to be precessed with the same preprocessing
buildAndCleanCSV('filteredBySendersTranslated',senderlist)