import numpy as np
import pandas as pd
import random

from DT import DecisionTree
from costFunction import Entropy, costFunction,Gini

random.seed(0)

def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size*len(df))#Calculating the test size 

    indices = df.index.tolist()
    test_indices = random.sample(population = indices, k = test_size)#Random sampling of data

    test_df = df.loc[test_indices]#Getting the test data set
    train_df = df.drop(test_indices)#Getting the train data set
    return train_df, test_df

df = pd.read_csv("diabetes.csv")
print(df.shape)
# print(df[1].shape)

# cF = costFunction()


train_val_df, test_df = train_test_split(df, test_size = 0.25)
train_val_df = train_val_df.to_numpy()
test_df = test_df.to_numpy()
print(Entropy().getValue(train_val_df))
# DT = DecisionTree(train_val_df,Gini(),100)
# DT.build(parent = DT.root)

# DT.printTree(DT.root,"")

def getAcc(DT,data):
    correct_pred = 0
    for d in data:
        if DT.predict(d) == d[-1]:
            correct_pred+=1

    print("Accuracy: ",(correct_pred/len(test_df))*100)

# getAcc(DT,test_df)


DT1 = DecisionTree(train_val_df,Entropy(),100)
DT1.build(parent = DT1.root)

# DT1.printTree(DT1.root,"")

getAcc(DT1,test_df)
