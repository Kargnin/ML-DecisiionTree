from costFunction import costFunction
import numpy as np
import pandas as pd

from utility import determine_best_split,check_purity, split_data

class Node:
    def __init__(self,attribute,data=[],limit = [0,np.Inf],dtype = "",val=None):
        self.attribute = attribute
        self.data = data
        self.limit = limit
        self.dtype = dtype
        self.leftChild = None
        self.rightChild = None
        if self.dtype == "leaf":
            self.val = val

    def getdata(self):
        return self.data
    
    def getlimit(self):
        return self.limit

    def getdtype(self):
        return self.dtype
    
    def setleftChild(self,node):
        self.leftChild = node
    
    def setrightChild(self,node):
        self.rightChild = node
    
class DecisionTree:
    def __init__(self,data,costFunction,maxDepth = 5):
        self.data = data
        self.costFunction = costFunction
        self.maxDepth = maxDepth
        self.root = Node(attribute = None,data=data,limit=[0,np.Inf],dtype ="root")
        

    def build(self,parent,height = 1):
        if check_purity(parent.data):
            parent.setleftChild(Node(attribute=-1,dtype="leaf",val = parent.data[:,-1][0]))
        else:
            sc,sv = determine_best_split(parent.data)
            d_b,d_a = split_data(parent.data,sc,sv)
            parent.setleftChild(Node(attribute = sc,data = d_b,limit=[0,sv]))
            parent.setrightChild(Node(sc,d_a,limit=[sv,np.Inf]))
            self.build(parent.leftChild,height+1)
            self.build(parent.rightChild,height+1)

    def predict(self,data):
        node = self.root
        while node.leftChild.dtype != "leaf":
            if node.leftChild.limit[0] <= data[node.leftChild.attribute] and  node.leftChild.limit[1] >= data[node.leftChild.attribute]:
                node = node.leftChild
            else:
                node = node.rightChild
        
        return node.leftChild.val
            
    def printTree(self,node,space):
        if node == None:
            return
        elif node.dtype == "leaf":
            if(node.val == 1):
                print(space+"Yes")
            else:
                print(space+"No")
        else:
            print(space+"Attribute column: ",node.attribute)
            print(space+"Limit: ",node.limit)
            print(space+"Cost: ",self.costFunction.getValue(node.data))
            space += "  "
            self.printTree(node.leftChild,space)
            self.printTree(node.rightChild,space)
            space = space[:-2]
            


