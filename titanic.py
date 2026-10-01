#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import joblib


# In[2]:


titanic = pd.read_csv("titanic.csv")


# In[3]:


titanic


# In[4]:


titanic.info()


# In[5]:


titanic.describe()


# In[6]:


titanic.isnull().sum()


# In[7]:


titanic.shape


# In[8]:


titanic["Age"].unique()


# In[9]:


titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())


# In[10]:


titanic.isnull().sum()


# In[11]:


titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis = 1)


# In[12]:


titanic


# In[13]:


titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])


# In[14]:


titanic.isnull().sum()


# In[15]:


## Survival Count
sns.countplot(data = titanic, x = 'Survived')

plt.title("Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Passenger Count")
plt.show()


# In[16]:


## Survival by Gender
sns.countplot(
data=titanic,
x='Sex',
hue='Survived'
)
plt.title('Survival by Gender')
plt.show()


# In[17]:


## Survival Rate by Gender
survival_gender = titanic.groupby('Sex')['Survived'].mean()


# In[18]:


survival_gender


# In[19]:


## Visualize
survival_gender.plot(kind='pie', autopct='%1.1f%%')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.show()


# In[20]:


## Survival by Passenger Class
sns.countplot(
data = titanic,
x='Pclass',
hue='Survived'
)
plt.title('Survival by Passenger Class')
plt.show()


# In[21]:


## Survival Rate by Class
class_survival = titanic.groupby('Pclass')['Survived'].mean()


# In[22]:


class_survival


# In[23]:


class_survival.plot(kind='bar')
plt.title('Survival Rate by Passenger Class')
plt.xlabel('class_survival')
plt.ylabel('Survival Rate')
plt.show()


# In[24]:


## Age Distribution
sns.histplot(
data = titanic,
x='Age',
hue='Survived',
kde=True
)
plt.title('Age Distribution by Survival')
plt.show()


# In[25]:


## Fare Distribution
sns.histplot(
data = titanic,
x='Fare',
hue='Survived',
kde=True
)
plt.title('Fare Distribution by Survival')
plt.show()


# In[26]:


## Feature Engineering
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1


# In[27]:


titanic.head()


# In[28]:


titanic['IsAlone'] = (titanic['FamilySize'] ==1).astype(int)


# In[29]:


titanic[['SibSp','Parch','FamilySize','IsAlone']].head()


# In[30]:


titanic.columns.tolist()


# In[31]:


titanic = pd.get_dummies(
titanic,
columns=['Sex', 'Embarked'],
drop_first=True
)


# In[32]:


titanic.head()


# In[33]:


## Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
titanic.corr(),
annot=True,
cmap='coolwarm'
)
plt.title('Feature Correlation Heatmap')
plt.show()


# In[34]:


x = titanic.drop('Survived', axis=1)
y = titanic['Survived']


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(
x,
y,
test_size=0.20,
random_state=42,
)


# In[36]:


print(x_train.shape)


# In[37]:


print(y_train.shape)


# In[38]:


print(x_test.shape)


# In[39]:


print(y_test.shape)


# In[40]:


## Feature Scaling
scalar = StandardScaler()


# In[41]:


x_train_scaled = scalar.fit_transform(x_train) 
x_test_scaled = scalar.transform(x_test) 


# In[42]:


## Logistic Regression
b = LogisticRegression()


# In[43]:


b.fit(x_train_scaled,y_train)


# In[44]:


b_pred = b.predict(x_test_scaled)


# In[45]:


## KNN 
knn = KNeighborsClassifier(n_neighbors=5)


# In[46]:


knn.fit(x_train_scaled,y_train)


# In[47]:


knn_pred = knn.predict(x_test_scaled)


# In[48]:


k_values = list(range(1 , 21))


# In[49]:


accuracies = []


# In[50]:


for k in k_values:
    kn = KNeighborsClassifier(n_neighbors = k)

    kn.fit(x_train_scaled,y_train)
    y_pred = kn.predict(x_test_scaled)
    acc = accuracy_score(y_test,y_pred)
    accuracies.append(acc)


# In[51]:


k_values


# In[52]:


accuracies


# In[53]:


print("Length of K values:", len(k_values))


# In[54]:


print("Length of Accuracies:", len(accuracies))


# In[55]:


plt.figure(figsize=(8,5))
plt.plot( 
k_values, 
accuracies, 
marker='o' 
) 
plt.xlabel('K') 
plt.ylabel('Accuracy') 
plt.title('KNN: Accuracy vs K') 
plt.grid()
plt.show() 


# In[56]:


best_k = k_values[np.argmax(accuracies)]


# In[57]:


best_k


# In[58]:


max(accuracies)


# In[59]:


## SVM
model = SVC(
    kernel='rbf', 
probability=True, 
random_state=42 
)


# In[60]:


model.fit(x_train_scaled,y_train)


# In[61]:


pred = model.predict(x_test_scaled)


# In[62]:


kernels = ['linear', 'rbf', 'poly']


# In[63]:


kernel_results = []


# In[64]:


for kernel in kernels:
    model =SVC(
        kernel=kernel,
        random_state=42)
    model.fit(x_train_scaled,y_train)
    pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test,pred)
    kernel_results.append([kernel,acc])

kernel_results


# In[65]:


df = pd.DataFrame(
    kernel_results,
    columns=['Kernel','Accuracy'])


# In[66]:


df


# In[67]:


df.plot(
    x = 'Kernel',
    y = 'Accuracy',
    kind = 'bar',
    legend = False
)
plt.title('SVM Kernel Comparison')
plt.ylabel('Accuracy')

plt.show()


# In[68]:


## Model Evaluation
def evaluate_model(name, y_test, y_pred): 

    print("\n", name) 

    print( 
        "Accuracy:", 
        accuracy_score(y_test, y_pred) 
    ) 

    print( 
        "Precision:", 
        precision_score(y_test, y_pred) 
    ) 

    print( 
        "Recall:", 
        recall_score(y_test, y_pred) 
    ) 

    print( 
        "F1 Score:", 
        f1_score(y_test, y_pred) 
    ) 


# In[69]:


evaluate_model( 
    "Logistic Regression", 
    y_test, 
    pred 
) 

evaluate_model( 
    "KNN", 
    y_test, 
    pred 
) 

evaluate_model( 
    "SVM", 
    y_test, 
    pred 
)


# In[70]:


## Model Comparison
results = pd.DataFrame({ 
'Model': [ 
'Logistic Regression', 
'KNN', 
'SVM' 
], 
'Accuracy': [ 
accuracy_score(y_test, pred), 
accuracy_score(y_test, pred), 
accuracy_score(y_test, pred) 
], 
'Precision': [ 
precision_score(y_test, pred), 
precision_score(y_test, pred), 
precision_score(y_test, pred) 
], 
'Recall': [ 
recall_score(y_test, pred), 
recall_score(y_test, pred), 
recall_score(y_test, pred) 
], 
'F1 Score': [ 
f1_score(y_test, pred), 
f1_score(y_test, pred), 
f1_score(y_test, pred) 
] 
}) 


# In[71]:


results


# In[72]:


results.set_index('Model').plot( 
kind='bar', 
figsize=(10, 6) 
) 
plt.title('ML Model Performance Comparison') 
plt.ylabel('Score') 
plt.ylim(0, 1) 
plt.xticks(rotation=0) 
plt.show() 


# In[73]:


## Confusion Matrix
cm = confusion_matrix(y_test, pred) 
disp = ConfusionMatrixDisplay( 
confusion_matrix=cm 
) 
disp.plot() 
plt.title('Logistic Regression Confusion Matrix') 
plt.show() 


# In[74]:


## KNN
cm = confusion_matrix(y_test, pred) 
ConfusionMatrixDisplay( 
confusion_matrix=cm 
).plot() 
plt.title('KNN Confusion Matrix') 
plt.show() 


# In[75]:


## SVM
cm = confusion_matrix(y_test,pred) 
ConfusionMatrixDisplay( 
confusion_matrix=cm 
).plot() 
plt.title('SVM Confusion Matrix') 
plt.show()


# In[76]:


## Prediction
new_passenger = pd.DataFrame({ 
'Pclass': [3], 
'Sex_male': [1], 
'Age': [25], 
'SibSp': [0], 
'Parch': [0], 
'Fare': [8.5], 
'Embarked_Q': [0], 
'Embarked_S': [1], 
'FamilySize': [1], 
'IsAlone': [1] 
})


# In[77]:


print("Scaler Columns:")
print(list(scalar.feature_names_in_))
print("\nNew passenger columns:")
print(list(new_passenger.columns.tolist()))


# In[78]:


new_passenger_fixed = new_passenger.reindex(columns = scalar.feature_names_in_)


# In[79]:


new_passenger_fixed = new_passenger_fixed.fillna(0)


# In[80]:


new_scaled = scalar.transform(new_passenger_fixed)


# In[81]:


prediction = kn.predict(new_scaled)


# In[82]:


prediction


# In[83]:


if prediction[0] == 1:
    print("Survived")

else :
    print("Not Survived")


# In[84]:


joblib.dump(b,"titanic_logisticregression.pkl")


# In[85]:


joblib.dump(model,"titanic_svm_model.pkl")


# In[86]:


joblib.dump(knn,"titani_knn_model.pkl")


# In[ ]:




