import csv
import sys

import itertools

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import numpy as np
from keras.layers.recurrent import SimpleRNN
from keras.models import model_from_json
from collections import Counter
import re
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random


def createModels(fileName,senderlist):
    #read and seperate each sender
    with open(fileName+'.csv', 'rb') as csv_file:
        csv.field_size_limit(sys.maxint)
        reader = csv.reader(csv_file)
        sendersWords = {}
        for k in range(0, len(senderlist)):
            sendersWords[senderlist[k]] = []
        line = next(reader, None)
        while (line):
            sender = line[0]
            splitEmail = line[1];
            sendersWords[sender].append(splitEmail)
            line = next(reader, None)
        #starting working with keras to create the model for each sender
        for k in range(0, len(senderlist)):
            senderEmailsArr=sendersWords[senderlist[k]];
            model = Sequential()
            n_hidden = 256
            n_fac = 42
            input_list=[]
            output_list=[]
            #list for each sentence with first word and number of words
            first_word_list=[]
            allEmails=""
            for email in senderEmailsArr:
                allEmails=allEmails+email+"--"
                emailSplit=email.split(" ")
                firstWord=""
                for i in range(0,len(emailSplit)):
                    if (emailSplit[i]!=''):
                        firstWord=emailSplit[i]
                        break
                first_word_list.append((firstWord, len(emailSplit)))
            allEmails=allEmails[:-2]
            vocabulary_size = 800
            unknown_token = "UNKNOWNTOKEN"
            sentence_start_token = "SENTENCESTART"
            sentence_end_token = "SENTENCEEND"
            line_break = "NEWLINE"
            separator = "SEPARATOR"
            emailText = allEmails.replace('\n', ' ' + line_break + ' ')
            emailText = emailText.replace('--', ' ' + separator + ' ')
            emailText = emailText.replace('.', ' ' + sentence_end_token + ' ' + sentence_start_token + ' ')
            emailText = re.sub(r'\d+', '', emailText)
            emailTextSeq = text_to_word_sequence(emailText, lower=True, split=" ")  # using only 10000 first words

            token = Tokenizer(nb_words=vocabulary_size, char_level=False)
            token.fit_on_texts(emailTextSeq)
            text_mtx = token.texts_to_matrix(emailTextSeq, mode='binary')
            input_ = text_mtx[:-1]
            output_ = text_mtx[1:]
            print('input shape is: '+str(input_.shape)+' output shape is: '+ str(output_.shape))
            #training the model
            model.add(Embedding(input_dim=input_.shape[1], output_dim=42, input_length=input_.shape[1]))
            model.add(Flatten())
            model.add(Dense(output_.shape[1], activation='sigmoid'))
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
            model.fit(input_, y=output_, batch_size=500, nb_epoch=50, verbose=1, validation_split=0.2)
            saveModelToFile(model, senderlist[k])
            with open(senderlist[k] + '_pred.csv', 'wb') as result_csv_file:
                writer = csv.writer(result_csv_file)
                for pred_email in first_word_list:
                    pred_text=""
                    nextWord=separator
                    for i in range(0,pred_email[1]):
                        if(i==vocabulary_size-1):
                            break
                        try:
                            pred_word=get_next(nextWord, token, model, text_mtx, emailTextSeq)
                            pred_text=pred_text+" "+pred_word
                            nextWord=pred_word
                        except ValueError:
                            print nextWord
                        if (nextWord == separator):
                            break
                    writer.writerow([pred_text])
            # evaluate the model
            #scores = model.evaluate(input_, output_, verbose=0)
            #print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            saveModelToFile(model,senderlist[k])


def loadModelFromFile(name):
    # load json and create model
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])
    # load weights into new model
    loaded_model.load_weights(name+".h5")
    print("Loaded model from disk")
    return loaded_model

def saveModelToFile(model,name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+".h5")
    print("Saved model to disk")
#a function to predict the next word
def get_next(text,token,model,fullmtx,fullText):
    tmp = text_to_word_sequence(text, lower=True, split=" ")
    tmp = token.texts_to_matrix(tmp, mode='binary')
    p = model.predict(tmp)
    bestMatch = np.min(np.argmax(p))
    options=np.where(fullmtx[:,bestMatch]>0)[0]
    smart_rand_pos=random.randint(0,len(options)-1)
    next_idx = smart_rand_pos
    return fullText[next_idx]

def readClassifier(file_name):
    return joblib.load(file_name)


def read_random_parts(file_name, percentage):
    lines = sum(1 for line in open(file_name)) #number of records in file (excludes header)
    print lines
    C = int(lines * (percentage/100.0)) #sample size (percentage%)
    buffer = []
    f = open(file_name, 'r')
    for line_num, line in enumerate(f):
        n = line_num + 1.0
        r = random.random()
        if n <= C:
            buffer.append(line.strip())
        elif r < C/n:
            loc = random.randint(0, C-1)
            buffer[loc] = line.strip()
    return buffer

def evaluateTheModel(fileName,senderlist):
    # read and seperate each sender
    with open(fileName + '.csv', 'rb') as csv_file:
        csv.field_size_limit(sys.maxint)
        reader = csv.reader(csv_file)
        sendersWords = {}
        for k in range(0, len(senderlist)):
            sendersWords[senderlist[k]] = []
        line = next(reader, None)
        while (line):
            sender = line[0]
            splitEmail = line[1];
            sendersWords[sender].append(splitEmail)
            line = next(reader, None)
        # starting working with keras to create the model for each sender
        for k in range(0, len(senderlist)):
            senderEmailsArr = sendersWords[senderlist[k]];
            model = Sequential()
            n_hidden = 256
            n_fac = 42
            input_list = []
            output_list = []
            # list for each sentence with first word and number of words
            first_word_list = []
            allEmails = ""
            for email in senderEmailsArr:
                allEmails = allEmails + email + "--"
                emailSplit = email.split(" ")
                firstWord = ""
                for i in range(0, len(emailSplit)):
                    if (emailSplit[i] != ''):
                        firstWord = emailSplit[i]
                        break
                first_word_list.append((firstWord, len(emailSplit)))
            allEmails = allEmails[:-2]
            vocabulary_size = 800
            unknown_token = "UNKNOWNTOKEN"
            sentence_start_token = "SENTENCESTART"
            sentence_end_token = "SENTENCEEND"
            line_break = "NEWLINE"
            separator = "SEPARATOR"
            emailText = allEmails.replace('\n', ' ' + line_break + ' ')
            emailText = emailText.replace('--', ' ' + separator + ' ')
            emailText = emailText.replace('.', ' ' + sentence_end_token + ' ' + sentence_start_token + ' ')
            emailText = re.sub(r'\d+', '', emailText)
            emailTextSeq = text_to_word_sequence(emailText, lower=True, split=" ")  # using only 10000 first words

            token = Tokenizer(nb_words=vocabulary_size, char_level=False)
            token.fit_on_texts(emailTextSeq)
            text_mtx = token.texts_to_matrix(emailTextSeq, mode='binary')
            input_ = text_mtx[:-1]
            output_ = text_mtx[1:]
            print('input shape is: ' + str(input_.shape) + ' output shape is: ' + str(output_.shape))
            # training the model
            model=loadModelFromFile(senderlist[k])
            #evaluate the model
            scores = model.evaluate(input_, output_, verbose=0)
            print("sender "+senderlist[k]+ "accuracy is: ")
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

#main
senderlist=['dean@bgu.ac.il','peler@exchange.bgu.ac.il','bitahon@bgu.ac.il','career@bgu.ac.il','shanigu@bgu.ac.il']
#createModels('filteredBySendersTranslated',senderlist)
#the accuracy of the model checked with cross validation of 0.2 and saved on the model, after trying to change the features values, the values that
#are we chose to use, are fitted to out data and our resources (cpu, memory, etc...)
evaluateTheModel('filteredBySendersTranslated',senderlist)
print('done.')