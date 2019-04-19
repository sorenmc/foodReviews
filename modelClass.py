import nltk
import math
import numpy as np
import pandas as pd
import os,sys
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation, Dropout, BatchNormalization

DATA_DIR = "data"

class modelClass:
    def __init__(self):
        self.xTrain = None
        self.yTrain = None
        self.xValidation = None
        self.yValidation = None
        self.xTest = None
        self.yTest = None
        self.nClasses = None
        self.models = None
        self.model = None

    def loadData(self):        
        dataFrame = pd.read_csv(os.path.join(DATA_DIR, "dataset_examples.tsv"),sep="\t")

        vectorizer = CountVectorizer()
        oneHotText = vectorizer.fit_transform(list(dataFrame["text"]))
        


        #make labels into 0,1
        encoder = LabelBinarizer()
        labels = encoder.fit_transform(list(dataFrame["sentiment"]))

        self.stratifyData(oneHotText,labels)
        
        
    
    def stratifyData(self,oneHotText,labels):
        """
        Given data is split stratified wise into 70% training, 15% validation and 15% test sets.
        """
        xTrain,xValidation,yTrain,yValidation =  train_test_split(oneHotText,labels,test_size = 0.3,random_state=42,stratify=labels)
        xValidation,xTest,yValidation,yTest =  train_test_split(xValidation,yValidation,test_size = 0.5,random_state=42,stratify=yValidation)

        
        self.xTrain = xTrain
        self.xValidation = xValidation
        self.xTest = xTest
        self.yTrainDecoded = yTrain
        self.yTrain = yTrain
        self.yValidation = yValidation
        self.yTest = yTest
    
    def trainModelLR(self,C):
        lrModel = LogisticRegression(C=C)
        self.model = lrModel
        score = self.crossEval(10)
        return score
    
    def crossEval(self,folds):

        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        cross = skf.split(self.xTrain,self.yTrainDecoded)
        scores = []
        for train,test in cross:
            model = clone(self.model)
            model.fit(self.xTrain[train],self.yTrain[train])
            tempScore = model.score(self.xTrain[test],self.yTrain[test])
            scores.append(tempScore)

            print("scored", tempScore )
        
        return np.mean(scores)


def main():
    dataModel = modelClass()
    dataModel.loadData()

    
    optimizer = BayesianOptimization(
        f = self.trainModelLR,
        pbounds = {'C':(0.1,1000)})
    optimizer.maximize(init_points=3, n_iter= 20)

if __name__ == "__main__":
    main()
