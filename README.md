# Codsoft
Task :Sales prediction 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
adv=pd.read_csv(r"c:\Users\anshika\Downloads\advertising.csv")
adv.head()
# EDA AND DATA CLEANING 
adv.info()
adv.describe()
adv.isnull().sum()
adv.shape
adv.duplicated().sum()
print(f"Average sales ={ adv['Sales'].head(10).mean()}")
# DATA VISUALIZATION 
sns.boxplot(adv['Sales'])
plt.ylabel('sales')
Plt.show()
sns.pairplot(adv,x_vars=['TV','Newspaper','Radio'],y_vars='Sales',height=4,aspect=1)
Plt.show()
sns.heatmap(adv.corr(),annot=True)
Plt.show()
# DATA MODELLING 
X=adv[['Radio']]
y=adv['Sales']
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test= train_test_split(X,y,train_size=0.5,test_size=0.2,random_state=100)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
import statsmodels.api as sm
X_train_sm= sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_sm).fit()
lr.params
print(lr.summary())
y_train_pred =lr.predict(X_train_sm)
res= (y_train-y_train_pred)
print("res",res[5])
y_train_pred.head()
fig=plt.figure()
sns.distplot(res,bins=15)
fig.suptitle('error terms',fontsize =15)
plt.xlabel('y')
plt.show()
[23/07, 2:11 pm] Anshika: X_test_sm= sm.add_constant(X_test)
y_pred=lr.predict(X_test_sm)
y_pred.head()
from sklearn.metrics import mean_squared_error ,r2_score
np.sqrt(mean_squared_error(y_test,y_pred))
r_squared= r2_score(y_test,y_pred)
r_squared
plt.scatter(X_train,y_train)
plt.plot(X_train,13.7249+0.0596*X_train,'r')
plt.show()


plt.scatter(X_test,y_test)
plt.plot(X_test,13.7249+0.0596*X_test,'r')
plt.show()




