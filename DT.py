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
    def __init__(self,data,costFunction = None,maxDepth = 5,test=[],validation=[]):
        self.data = data
        self.test = test
        self.validation = validation
        self.costFunction = costFunction
        self.maxDepth = maxDepth
        self.root = Node(attribute = None,data=data,limit=[0,np.Inf],dtype ="root")
        

    def build(self,parent,height = 1):
        if check_purity(parent.data):
            parent.setleftChild(Node(attribute=-1,dtype="leaf",val = parent.data[:,-1][0]))
        else:
            sc,sv = determine_best_split(parent.data,self.costFunction)
            if height>=self.maxDepth or (sc == -1 and sv == -1):
                p_ = 0
                n_ = 0
                for d in parent.data:
                    if d[-1] == 1:
                        p_ += 1
                    else:
                        n_ += 1
                if p_ > n_:
                    parent.setleftChild(Node(attribute=-1,dtype="leaf",val = 1))
                else:
                    parent.setleftChild(Node(attribute=-1,dtype="leaf",val = 0))
                return

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
    
    def get_num_nodes(self,node):
        if node == None:
            return 0
        n = 1
        n = n + self.get_num_nodes(node.leftChild)
        n = n + self.get_num_nodes(node.rightChild)
        return n
    
    def getAcc(self,data):
        correct_pred = 0
        for d in data:
            if self.predict(d) == d[-1]:
                correct_pred+=1
        return (correct_pred/len(data))*100

    def printTree(self):
        self.printTree_(self.root,"")
            
    def printTree_(self,node,space):
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
            self.printTree_(node.leftChild,space)
            self.printTree_(node.rightChild,space)
            space = space[:-2]
    
    def build_val(node,node_val):
        if node.leftChild.dtype != "leaf":
            node_val.setleftChild(Node(attribute = node.leftChild.attribute,data = [],limit=[0,node.leftChild.limit[1]]))
            node_val.setrightChild(Node(attribute = node.rightChild.attribute,data = [],limit=[node.rightChild.limit[0],np.Inf]))
        if node.leftChild.dtype == "leaf":
            node_val.setleftChild(Node(attribute = -1,dtype = "leaf",val = node.leftChild.val))
            return
        for d in node_val.data:
            if node.leftChild.limit[0] <= d[node.leftChild.attribute] and  node.leftChild.limit[1] >= d[node.leftChild.attribute]:
                node_val.leftChild.data.append(d)
            else:
                node_val.rightChild.data.append(d)
        DecisionTree.build_val(node.leftChild,node_val.leftChild)
        DecisionTree.build_val(node.rightChild,node_val.rightChild)
            
        
    def post_pruning(self,val_data):
        DT_val = DecisionTree(data = val_data)
        DecisionTree.build_val(self.root,DT_val.root)

        self.prune(self.root,DT_val.root)

    def prune(self,node,node_val):
        if len(node_val.data) == 0:
            return
        if node == None or node_val == None or node.leftChild.dtype == "leaf" :
            return
        else:
            p = 0
            
            for d in node_val.data:
                if self.predict(d)==d[-1]:
                    p+=1
            acc_before = p/len(node_val.data)
            p_ = 0
            n_ = 0
            for d in node.data:
                if d[-1] == 1:
                    p_ += 1
                else:
                    n_ += 1
            acc_after = 0
            if p_ > n_:
                for d in node_val.data:
                    if d[-1]==1:
                        acc_after += 1
            else:
                for d in node_val.data:
                    if d[-1]==0:
                        acc_after += 1
    
            acc_after /= len(node_val.data)

            if acc_after >= acc_before:
                node.rightChild = None
                if p_ > n_:
                    node.setleftChild(Node(attribute=-1,dtype="leaf",val = 1))
                else:
                    node.setleftChild(Node(attribute=-1,dtype="leaf",val = 0))
            else:
                self.prune(node.leftChild,node_val.leftChild)
                self.prune(node.rightChild,node_val.rightChild)


            
            
                


        
        
            


