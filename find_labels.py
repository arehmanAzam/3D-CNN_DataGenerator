import numpy as np

def findLabels(partition, labels):
    arr=[]
    for i in range(0,len(partition)):
        a = labels[i]
        obj={"path":partition[i],"class":np.where(a == a.max())[0][0]}
        arr.append(obj)
    return arr;
