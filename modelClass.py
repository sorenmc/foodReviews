import nltk
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import os,sys
import GPyOpt

from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense, Embedding, Activation, Dropout,BatchNormalization,Conv2D,Conv1D,Flatten,LSTM,MaxPool1D,TimeDistributed

DATA_DIR = "data"

class modelClass:
    def __init__(self):

        #Model parameters
        self.nCNN = None
        self.nDense = None
        self.nEmbedding = None
        self.nCNNFilters = None
        self.nNNFilters = None
        self.nKernel = None
        self.nStrides = None
        self.poolSize = None


        self.vocab_size = None
        self.maxLen = None

        self.nClasses = None

        #Data
        self.xTrain = None
        self.yTrain = None
        self.xValidation = None
        self.yValidation = None
        self.xTest = None
        self.yTest = None
        
        self.model = None

    def loadDataOneHot(self):        
        dataFrame = pd.read_csv(os.path.join(DATA_DIR, "dataset_examples.tsv"),sep="\t")

        vectorizer = CountVectorizer()
        texts = vectorizer.fit_transform(list(dataFrame["text"]))
        


        #make labels into 0,1
        encoder = LabelBinarizer()
        labels = encoder.fit_transform(list(dataFrame["sentiment"]))

        self.stratifyData(texts,labels)
    
    def loadDataSequence(self):
        #dataFrame = pd.read_csv("/content/drive/My Drive/Colab Notebooks/data/dataset_examples.tsv",sep="\t")
        dataFrame = pd.read_csv(os.path.join(DATA_DIR, "dataset_examples.tsv"),sep="\t")
        

        texts = list(dataFrame["text"])
        tokenizer = Tokenizer()
        
        tokenizer.fit_on_texts(texts)
        self.vocab_size = len(tokenizer.word_index) + 1
        sequenceText = tokenizer.texts_to_sequences(texts)

        self.maxLen = max([len(text) for text in sequenceText ])
        padSequenceText = pad_sequences(sequenceText,padding = "post",maxlen = self.maxLen )


        #make labels into 0,1
        encoder = LabelBinarizer()
        #encoder = LabelEncoder()
        #labels = to_categorical(dataFrame["sentiment"])
        labels = encoder.fit_transform(dataFrame["sentiment"])
        #labels = labels.flatten()
        self.stratifyData(padSequenceText,labels)
    
    def stratifyData(self,texts,labels):
        """
        Given data is split stratified wise into 70% training, 15% validation and 15% test sets.
        """
        xTrain,xValidation,yTrain,yValidation =  train_test_split(texts,labels,test_size = 0.3,random_state=42,stratify=labels)
        xValidation,xTest,yValidation,yTest =  train_test_split(xValidation,yValidation,test_size = 0.5,random_state=42,stratify=yValidation)

        
        self.xTrain = xTrain
        self.xValidation = xValidation
        self.xTest = xTest
        self.yTrainDecoded = yTrain
        self.yTrain = yTrain
        self.yValidation = yValidation
        self.yTest = yTest
        self.nClasses = len(set(yTest.flatten()))
    
    def optimizeLR(self,C):
        print("C is right now",C[0][0])
        self.model = LogisticRegression(C=C[0][0])
        score = self.crossEval(10)
        return score

    def optimizeCNN(self,variables):# nDense, nEmbedding, nCNNFilters, nNNFilters, nKernel, nStrides,poolSize):
        self.nCNN = int(variables[0][0])
        self.nDense = int(variables[0][1])
        self.nEmbedding = int(variables[0][2])
        self.nCNNFilters = int(variables[0][3])
        self.nNNFilters = int(variables[0][4])
        self.nKernel = int(variables[0][5])
        self.nStrides = int(variables[0][6])
        self.poolSize = int(variables[0][7])

        self.buildCNN()

        score = self.crossEval(10)
        return score
    
    def addConvLayer(self,model):
        model.add(Conv1D(kernel_size = self.nKernel, filters = self.nCNNFilters,strides = self.nStrides, padding="valid" ))
        model.add(Activation("elu"))
        model.add(BatchNormalization())
        model.add(MaxPool1D(pool_size = self.poolSize,padding="valid"))
        return model

    def buildCNN(self):
        
        model = Sequential()
        model.add(Embedding(input_dim = self.vocab_size, output_dim = self.nEmbedding, input_length = self.maxLen ))

        #add nCNN conv layers 
        for _ in range(0,self.nCNN):
            model = self.addConvLayer(model)
        
        model.add(Flatten())
        #add nDense
        
        for _ in range(0,self.nDense):
            model.add(Dense(self.nNNFilters))
        
        model.add(Dense(1, activation = "softmax" ))
        model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        self.model = model

    
        
    

    def buildLR(self,C):
        self.model = LogisticRegression(C=C)
    
    def crossEval(self,folds):

        skf = StratifiedKFold(n_splits=folds, shuffle=True,random_state=42)
        type1 ="<class 'sklearn.linear_model.logistic.LogisticRegression'>"
        count = 1
        scores = []
        for train,test in skf.split(self.xTrain,self.yTrain):
            
            if str(type(self.model)) == type1:
                self.model.fit(self.xTrain[train],self.yTrain[train])
                score = self.model.score(self.xTrain[test],self.yTrain[test])
            else:
                self.buildCNN()
                self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                self.model.fit(self.xTrain[train],self.yTrain[train],epochs=100,batch_size=10,verbose=0)
                score = self.model.evaluate(self.xTrain[test],self.yTrain[test])
            
            print("The score of iteration ", count, "was", score)
            scores.append(score)
            count = count+1
        
        meanScore = np.mean(scores)*100
        print("Final score was ", meanScore)

        return 1-meanScore
    
    def trainModel(self):
        self.model.fit(self.xTrain,self.yTrain)
    
    def validateModel(self):
        print("Validation score is", self.model.score(self.xValidation,self.yValidation))

    def testModel(self):
        print("test score is", self.model.score(self.xTest,self.yTest))


def main():
    dataModel = modelClass()
    dataModel.loadDataSequence()
    
    domain = [{'name': 'nCNN','type':'discrete','domain':tuple(range(1,6))},
              {'name': 'nDense','type':'discrete','domain':tuple(range(0,3))},
              {'name': 'nEmbedding','type':'discrete','domain':tuple(range(5,200))},
              {'name': 'nCNNFilters','type':'discrete','domain':tuple(range(2,1000))},
              {'name': 'nNNFilters','type':'discrete','domain':tuple(range(3,1000))},
              {'name': 'nKernel','type':'discrete','domain':tuple(range(1,4))},
              {'name': 'nStrides','type':'discrete','domain':tuple(range(1,2))},
              {'name': 'poolSize','type':'discrete','domain':tuple(range(1,2))}]
    
   # bounds = [{"name":'C','type':'continuous','domain':(0.1,100)}]
    #tf.enable_eager_execution()
    optimizer = GPyOpt.methods.BayesianOptimization(
        f = dataModel.optimizeCNN,
        domain = domain,
        acquisition_type ='LCB',       # LCB acquisition
        acquisition_weight = 0.1
        )
    max_iter = 20
    optimizer.run_optimization(max_iter)
    print(optimizer.x_opt)


if __name__ == "__main__":
    main()
