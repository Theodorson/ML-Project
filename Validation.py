import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from Bag_of_words import *

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


def plot_confusion_matrix(confusion_matrix, title, dialect_list, cmap=plt.cm.Blues):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(dialect_list))
    plt.xticks(tick_marks, dialect_list, rotation=45)
    plt.yticks(tick_marks, dialect_list)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def validation_doc ():
    print("===================================================================================================")
    print("Validation data")
    #load data
    validation_data = np.genfromtxt("validation_samples.txt", encoding="utf-8", dtype=None, delimiter='\t', names=('c1', 'c2'))
    validation_samples = validation_data['c2']
    validation_samples = np.array(validation_samples)


    validation_data_labels = np.genfromtxt("validation_labels.txt", encoding="utf-8", dtype=None, delimiter='\t', names=('c1', 'c2'))
    validation_labels = validation_data_labels['c2']
    validation_labels = np.array(validation_labels)


    #split data
    X_train, X_test , y_train, y_test = train_test_split (validation_samples, validation_labels, test_size=0.2, random_state=1)


    Model = Bag_of_words()
    print("Build Vocabulary")
    Model.build_vocabulary(X_train)

    print("Start Algorithm")
    train_features = Model.get_features(X_train)
    test_features = Model.get_features(X_test)
    print(train_features.shape)
    print(test_features.shape)

    # normalizare
    scaled_train_data, scaled_test_data = normalize_data(train_features, test_features)

    # SVM classifier
    SVM = svm.SVC(C=100, kernel='linear')
    SVM.fit(scaled_train_data, y_train)
    predicted_labels = SVM.predict(scaled_test_data)
    #Predictii
    print(predicted_labels)
    #F1 score
    print('F1 score: ', f1_score(np.asarray(y_test), predicted_labels))
    #Confusion matrix
    cm_validation_set = confusion_matrix(y_test, predicted_labels)
    dialect_list = ["RO", "MD"]
    print("Confusion Matrix: ")
    print(cm_validation_set)
    plot_confusion_matrix( cm_validation_set, "Confusion matrix for validation set", dialect_list)
    print("===================================================================================================")

validation_doc()