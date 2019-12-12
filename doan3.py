from pyvi import ViTokenizer, ViPosTagger #Thư viện NLP tiếng Việt
from tqdm import tqdm
from io import open
from sklearn import naive_bayes
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

import gensim #Thư viện NLP
import pickle #Save file .pkl
import numpy as np
import matplotlib.pyplot as plt

#Load path
import os
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'DA3')

#Fucntion load data
def get_data(folder_path):
    Data = [] #Chứa data
    Label = [] #Chứa label
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines) #Xóa các kí tự đặc biệt
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines) #Xóa khoảng trắng

                Data.append(lines)
                Label.append(path)
    return Data, Label

'''
#Save data vào file pkl
train_path = os.path.join(dir_path, 'Data/Train_1')
Data_train, Label_train = get_data(train_path)

test_path = os.path.join(dir_path, 'Data/Test_1')
Data_test, Label_test = get_data(test_path)
'''

#Load data train and test từ file pkl
Data_train = pickle.load(open('Data/X_data.pkl', 'rb'))
Label_train = pickle.load(open('Data/y_data.pkl', 'rb'))

Data_test = pickle.load(open('Data/X_test.pkl', 'rb'))
Label_test = pickle.load(open('Data/y_test.pkl', 'rb'))

#Tf-Idf Vectors as Features
# word level - we choose max number of words equal to 30000 except all words (100k+ words)
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
tfidf_vect.fit(Data_train) #Training
Data_train_tfidf =  tfidf_vect.transform(Data_train)
# assume that we don't have test set before
Data_test_tfidf =  tfidf_vect.transform(Data_test)

#Chuyển label về dạng số
le = preprocessing.LabelEncoder()
Label_train_n = le.fit_transform(Label_train)
Label_test_n = le.fit_transform(Label_test)
le.classes_

#Build model
def train_model(classifier, Data_train, Label_train, Data_test, Label_test, is_neuralnet=False, n_epochs=3):       
    X_train, X_val, Y_train, Y_val = train_test_split(Data_train, Label_train, test_size=0.1, random_state=42)
    
    if is_neuralnet:
        classifier.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=n_epochs, batch_size=512)
        
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict()
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, Y_train)
    
        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(Data_test)

    #Đánh giá giải thuật
    print("Train accuracy: ", metrics.accuracy_score(train_predictions, Y_train))
    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, Y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, Label_test))

#Chạy model
train_model(naive_bayes.MultinomialNB(), Data_train_tfidf, Label_train, Data_test_tfidf, Label_test, is_neuralnet=False)