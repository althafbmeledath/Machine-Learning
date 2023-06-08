import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KNN import KNN

#read the dataframe
df=pd.read_csv('Social_Network_Ads.csv')

#Extract the values of X and y
X=df.iloc[:,2:4].values
y=df.iloc[:,-1].values

#train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

#scale down the values
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#object of the KNN class
knn=KNN(k=5)
#fit the data
knn.fit(X_train,y_train)

#smart function to predict
def pred_out():
    age=int(input("Enter the age "))
    salary=int(input("Enter the salary "))
    #create an array from the inputs
    X_new=np.array([[age],[salary]]).reshape(1,2)
    #scale down the values
    X_new=scaler.transform(X_new)
    #store the prediction
    out=knn.predict(X_new)
    print(out)
    if out==1:
        print('Will purchase')
    else:
        print('Will not purchase')

pred_out()    