import numpy as np
from costFunction import Entropy, Gini

def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    return len(unique_classes) == 1

def get_potential_splits(data):#Getting all the potential points with which we can split data
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        for index in range(len(unique_values)):
            if index!=0:
                current_value = unique_values[index]
                previous_value = unique_values[index-1]
                potential_split = (current_value+previous_value)/2
                
                potential_splits[column_index].append(potential_split)
    
    return potential_splits

def split_data(data, split_column, split_value):#Splitting data with the target attribute specified and target attribute vale
    
    split_column_values = data[:, split_column]
    
    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >= split_value]
    
    return data_below, data_above

def Gain(data_below,data_above,cF):
    # print(len(data_below[:,0]))
    # print(len(data_above[:,0]))
    return  cF.getValue(np.vstack((data_below,data_above)))-(len(data_below[:,0])/(len(data_below[:,0])+len(data_above[:,0])))*cF.getValue(data_below)-(len(data_above[:,0])/(len(data_below[:,0])+len(data_above[:,0])))*cF.getValue(data_above) 


def determine_best_split(data,cF):
    potential_splits = get_potential_splits(data)

    max_gain = -1000
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column = column_index, split_value = value)
            overallgain = Gain(data_below,data_above,cF)
            # if overallgain < 0:
            #     print("Gain < 0")
            # print("Column = ",column_index)
            # print("Value = ",value)
            # print("gain = ",overallgain)
            if overallgain > max_gain:
                max_gain = overallgain
                best_split_column = column_index
                best_split_value = value

    if max_gain <= 0:
        return -1,-1
    return best_split_column, best_split_value


