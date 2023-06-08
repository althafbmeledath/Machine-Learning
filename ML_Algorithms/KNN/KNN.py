from collections import Counter
import operator


class KNN:
    
    #initialisation
    def __init__(self,k):
        self.k=k
        
    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        print("Training completed.....Ready to predict")
        
    def predict(self,X_test):
        distances={}
        counter=1
        #calculate the distance of the input point with every point in the training data
        for i in self.X_train:
            distances[counter]=((X_test[0][0]-i[0])**2+(X_test[0][1]-i[1])**2)**1/2
            counter+=1
            
        #sort the distances
        distances=sorted(distances.items(),key=operator.itemgetter(1))
        distances=distances[:self.k]
        
        labels=[]
        for i in distances:
            labels.append(self.y_train[i[0]])
        
        #return the majority count neighbour
        return (Counter(labels).most_common()[0][0])
        
    