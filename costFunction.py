import numpy as np

class Node:
    pass

class costFunction:
    def __init__(self,name) -> None:
        self.name = name
    
    def getName(self):
        return self.name
    
    def getvalue(data):
        pass

class Gini(costFunction):
    def __init__(self) -> None:
        super().__init__("Gini")

    def getValue(self,data):
        pos = 0
        neg = 0
        for d in data:
            if d[-1] == 1:
                pos+=1
            else:
                neg+=1
        l = pos+neg
        pos /= l
        neg /= l
        # return 1-pos**2-neg**2
        return 2*pos*neg
    
class Entropy(costFunction):
    def __init__(self) -> None:
        super().__init__("Entropy")

    def getValue(self,data):
        pos = 0
        neg = 0
        for d in data:
            if d[-1] == 1:
                pos+=1
            else:
                neg+=1
        l = pos + neg
        pos /= l
        neg /= l
        if pos == 0:
            p = 0
        else:
            p = -1*pos*(np.log2(pos))
        if neg == 0:
            n = 0
        else:
            n = -1*neg*(np.log2(neg))
        # print("pos = ",pos)
        # print("neg = ",neg)
        # print("gain = ",p+n)
        return p+n


