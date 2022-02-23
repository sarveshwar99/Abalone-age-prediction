import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


data=pd.read_csv("/mnt/F8F8B8AFF8B86E0E/abalone /abalone.csv")  #load dataset

X=data.iloc[:,:-1]



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score


data['age']=data['Rings']+1.5 # a process to compute abalone age
data.drop('Rings', axis=1, inplace=True)

y=data.iloc[:,-1]
from sklearn.preprocessing import LabelEncoder
labl=LabelEncoder()
X['Sex']=labl.fit_transform(X['Sex'])


from sklearn.preprocessing import StandardScaler
standardScale = StandardScaler()
X=standardScale.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=None)

print(data.info())

print(data.describe())

data.hist(figsize=(20,10),grid=False,layout=(2,4), bins=40)


#-converting all the feature to object

numerical=data.select_dtypes(include=np.number).columns
categorical=data.select_dtypes(include=np.object).columns


regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=3)
regr.fit(X_train,y_train)

y_pred=regr.predict(X_test)

y_demo=list(y_pred)

plt.plot(sorted(y_pred))
plt.plot(sorted(y_test))


from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))





