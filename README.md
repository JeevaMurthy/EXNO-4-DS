# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```pyhton
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()

```  
![alt text](<Screenshot 2024-10-08 124256.png>)

```pyhton
df.dropna()

```
![alt text](<Screenshot 2024-10-08 124306.png>)

```pyhton
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals

```
![alt text](<Screenshot 2024-10-08 124310.png>)

```pyhton
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)

```
![alt text](<Screenshot 2024-10-08 124316.png>)

```pyhton
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df

```
![alt text](<Screenshot 2024-10-08 124323.png>)
```pyhton
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df

```
![alt text](<Screenshot 2024-10-08 124329.png>)

```pyhton
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()

```
![alt text](<Screenshot 2024-10-08 124343.png>)


```pyhton
import pandas as pd
import numpy as np
import seaborn as sns

```



```pyhton
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

```



```pyhton
df=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
df

```
![alt text](<Screenshot 2024-10-08 124403.png>)


```pyhton
df.info()

```

![alt text](<Screenshot 2024-10-08 124411.png>)

```pyhton
df.isnull().sum()

```
![alt text](<Screenshot 2024-10-08 124416.png>)

```pyhton
missing=df[df.isnull().any(axis=1)]
missing

```
![alt text](<Screenshot 2024-10-08 124426.png>)

```pyhton
df2=df.dropna(axis=0)
df2

```
![alt text](<Screenshot 2024-10-08 124434.png>)

```pyhton
sal=df['SalStat']

```


```pyhton
df2['SalStat']=df['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(df2['SalStat'])

```
![alt text](<Screenshot 2024-10-08 124445.png>)

```pyhton
df2

```
![alt text](<Screenshot 2024-10-08 124456.png>)

```pyhton
new_df=pd.get_dummies(df2, drop_first=True)
new_df

```
![alt text](<Screenshot 2024-10-08 124505.png>)

```pyhton
columns_list=list(new_df.columns)
print(columns_list)

```
![alt text](<Screenshot 2024-10-08 124515.png>)

```pyhton
features=list(set(columns_list)-set('SalStat'))
print(features)

```
![alt text](<Screenshot 2024-10-08 124523.png>)


```pyhton
y=new_df['SalStat'].values
print(y)

```
![alt text](<Screenshot 2024-10-08 124530.png>)

```pyhton
x=new_df[features].values
print(x)

```
![alt text](<Screenshot 2024-10-08 124534.png>)
```pyhton
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0)

```


```pyhton
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)

```


```pyhton
KNN_classifier.fit(train_x,train_y)

```
![alt text](<Screenshot 2024-10-08 124542.png>)

```pyhton
prediction = KNN_classifier.predict(test_x)

```



```pyhton
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)

```
![alt text](<Screenshot 2024-10-08 124548.png>)

```pyhton
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

```
![alt text](<Screenshot 2024-10-08 124554.png>)

```pyhton
print('Misclassified samples: %d' %(test_y !=prediction).sum())

```
![alt text](<Screenshot 2024-10-08 124557.png>)

```pyhton
df.shape

```
![alt text](<Screenshot 2024-10-08 124601.png>)
```pyhton
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

```
![alt text](<Screenshot 2024-10-08 124606.png>)

```pyhton
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)

```

![alt text](<Screenshot 2024-10-08 124612.png>)

```pyhton
chi2, p, _, _=chi2_contingency(contigency_table)
print(f"Chi-square statistics:{chi2}")
print(f"p-value:{p}")

```
![alt text](<Screenshot 2024-10-08 124616.png>)


```pyhton
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df

```
![alt text](<Screenshot 2024-10-08 124629.png>)




```pyhton
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

```

![alt text](<Screenshot 2024-10-08 124635.png>)





# RESULT:

Hence,Feature Scaling and Feature Selection process has been performed on the given data set.
