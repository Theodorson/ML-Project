import numpy as np
import sklearn
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
import csv

from Bag_of_words import *


# functie de normalizare a datelor
def normalize_data(train_data, test_data):
    scaler = None
    scaler = preprocessing.Normalizer(norm='l2')
    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return (scaled_train_data, scaled_test_data)
    else:
        print("Scalarea nu a fost facuta, datele initiale sunt intoarse.")
        return (train_data, test_data)




# incarcare date
train_data = np.genfromtxt("train_samples.txt",encoding="utf-8",dtype=None, delimiter='\t', names = ('c1','c2'))
train_samples = train_data['c2']
train_samples = np.array(train_samples)


test_data =  np.genfromtxt("test_samples.txt",encoding="utf-8",dtype=None, delimiter='\t', names = ('c1','c2'))
test_samples = test_data['c2']
test_samples = np.array(test_samples)


# stocarea etichetelor
etichete = test_data['c1']
etichete = np.array(etichete)


# incarcarea labelurilor
train_labels_data = np.genfromtxt("train_labels.txt",encoding="utf-8",dtype=None, delimiter='\t', names = ('c1','c2'))
train_labels = train_labels_data['c2']
train_labels = np.array(train_labels)




print("===================================================================================================")
print("Train data")
# bag_of_words
Model = Bag_of_words()
print ("Build Vocabulary")
Model.build_vocabulary(train_samples[:5000])

print("Start Algorithm")
train_features = Model.get_features(train_samples[:2623])
test_features = Model.get_features(test_samples)
print(train_features.shape)
print(test_features.shape)

# normalizare
scaled_train_data, scaled_test_data = normalize_data(train_features, test_features)

#SVM classifier
SVM = svm.SVC(C=100, kernel='linear')
SVM.fit(scaled_train_data, train_labels[:2623])
predicted_labels = SVM.predict(scaled_test_data)

#scrierea predictiilor in fisierul csv
with open('file.csv', mode='w', newline="") as file:
        writer=csv.writer(file)
        writer.writerow(['id','label'])
        contor = 0
        for j in range(len(predicted_labels)):
                writer.writerow([etichete[contor], predicted_labels[j]])
                contor +=1
        print(predicted_labels)

print("===================================================================================================")










