import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from DT import DecisionTree
from costFunction import Entropy, costFunction,Gini
from utility import Gain


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size*len(df))#Calculating the test size 

    indices = df.index.tolist()
    test_indices = random.sample(population = indices, k = test_size)#Random sampling of data

    test_df = df.loc[test_indices]#Getting the test data set
    train_df = df.drop(test_indices)#Getting the train data set
    return train_df, test_df

df = pd.read_csv("diabetes.csv")



def bestTree(costFunction):
    k = 0
    max_acc = 0
    avg_acc = 0
    for i in range(11,32,2):
        k+=1
        random.seed(i)
        train_val_df, test_df = train_test_split(df, test_size = 0.2)
        train_df,val_df = train_test_split(train_val_df,test_size = 0.2)
        train_val_df = train_val_df.to_numpy()
        train_df = train_df.to_numpy()
        val_df = val_df.to_numpy()
        test_df = test_df.to_numpy()
        DT = DecisionTree(train_df,costFunction,100,test = test_df,validation=val_df)
        DT.build(parent = DT.root)
        acc = DT.getAcc(test_df)
        avg_acc += acc
        if acc > max_acc:
            max_acc = acc
            best_tree = DecisionTree(train_df,costFunction,100,test = test_df,validation=val_df)
    avg_acc /= k
    print("Average Acccuracy = ",avg_acc)
    print("Best Accuracy = ",max_acc)
    return best_tree

best_tree_gini = bestTree(Gini())
best_tree_gini.build(best_tree_gini.root)
best_tree_ent = bestTree(Entropy())
best_tree_ent.build(best_tree_ent.root)



def depthAnalysis(train_val_df,test_df,costFunction=Gini()):
    grid_search = {"max_depth": [],  "acc_train": [], "acc_test": []}
    node_acc = {"node": [], "acc": []}

    for max_depth in range (1,30):#Recording train and test error over different max depths
        tree = DecisionTree(train_val_df,costFunction, max_depth)
        tree.build(parent = tree.root)
        acc_train = tree.getAcc(train_val_df)
        acc_test = tree.getAcc(test_df)
        num_node = tree.get_num_nodes(node = tree.root)
        
        grid_search["max_depth"].append(max_depth)
        grid_search["acc_train"].append(acc_train)
        grid_search["acc_test"].append(acc_test)

        node_acc["node"].append(num_node)
        node_acc["acc"].append(acc_test)
        
        # print("Progress: Iteration {}".format(max_depth))
        
    grid_search = pd.DataFrame(grid_search)
    grid_search.sort_values("acc_test", ascending = False)

    plt.plot(grid_search["max_depth"], grid_search["acc_train"], label = 'Training Accuracy')
    plt.plot(grid_search["max_depth"], grid_search["acc_test"], label = 'Test_accuracy')
    plt.xlabel("Height")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    max_accuracy = grid_search["acc_test"][0]
    best_depth = 1
    for i in range(0, len(grid_search["acc_test"])):
        if grid_search["acc_test"][i] >= max_accuracy:
            max_accuracy = grid_search["acc_test"][i]
            best_depth = i+1

    print("Best Depth of the tree for which test accuracy is maximum is : {}".format(best_depth))
    

    best_tree = DecisionTree(train_val_df, costFunction, best_depth)
    best_tree.build(best_tree.root)
    print("Best accuracy = ",best_tree.getAcc(test_df))

    max_accuracy = node_acc["acc"][0]
    best_num_nodes = 1
    for i in range(0, len(node_acc["acc"])):
        if node_acc["acc"][i] >= max_accuracy:
            max_accuracy = node_acc["acc"][i]
            best_num_nodes = node_acc["node"][i]
    
    print("Best accuracy is at {} num of nodes".format(best_num_nodes))

    node_acc = pd.DataFrame(node_acc)
    node_acc.sort_values("acc", ascending = False)

    plt.plot(node_acc["node"], node_acc["acc"], label = 'Test_accuracy')
    plt.xlabel("Number of Nodes")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

train_val_df, test_df = train_test_split(df, test_size = 0.2)
train_df,val_df = train_test_split(train_val_df,test_size = 0.2)
train_val_df = train_val_df.to_numpy()
train_df = train_df.to_numpy()
val_df = val_df.to_numpy()
test_df = test_df.to_numpy()


# DT = DecisionTree(train_df,Gini(),100)
# DT.build(DT.root)
# DT.printTree()
# print(DT.getAcc(test_df))
depthAnalysis(train_df,test_df,Gini())
depthAnalysis(train_df,test_df,Entropy())

print("Test accuracy before pruning = ",best_tree_gini.getAcc(best_tree_gini.test))
print("Val accuracy before pruning = ",best_tree_gini.getAcc(best_tree_gini.validation))
best_tree_gini.post_pruning(best_tree_gini.validation)
print("Test accuracy after pruning = ",best_tree_gini.getAcc(best_tree_gini.test))
print("Val accuracy after pruning = ",best_tree_gini.getAcc(best_tree_gini.validation))

print("Test accuracy before pruning = ",best_tree_ent.getAcc(best_tree_ent.test))
print("Val accuracy before pruning = ",best_tree_ent.getAcc(best_tree_ent.validation))
best_tree_ent.post_pruning(best_tree_ent.validation)
print("Test accuracy after pruning = ",best_tree_ent.getAcc(best_tree_ent.test))
print("Val accuracy after pruning = ",best_tree_ent.getAcc(best_tree_ent.validation))

