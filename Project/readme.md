# DANIYAL SHAFIQUE -61428 (Naive Bayes Model)
# ASADULLAH (SVM MODEL)

###### The Naïve Bayes model assign zero probability and  it won’t be able to make any predictions.
###### Because test data set has a explicit variable of a category because it wasn’t present in the training data set.
###### Applying Grid Search we have found that the best accuracy is (parameter = 10.0) and apply different parameters after several times we have got no change.

![projectkag](https://user-images.githubusercontent.com/43805740/170001794-36cc3856-b1a7-4924-8346-e47aac912282.PNG)


The problem I faced the data was very big and 
The model and code I tried to use took alot of time 
to run the code.And I used many times to run the model but it doesn't run accurately
And sometimes it was taking 5 to 6 hours to complete it.
So that's why I committed the code here on GitHub.











# PROJECT CODE

import pandas as pd
from sklearn.metrics import accuracy_score # For Accuracy Checking
from sklearn.model_selection import train_test_split # Splitting Data and train test
from sklearn.neighbors import KNeighborsClassifier # Apply ML Algo KNN
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB #  Multinomial Naive Bayes Model
from sklearn.model_selection import cross_val_score # For Cross Validation Checking
from sklearn.naive_bayes import BernoulliNB 
import warnings
warnings.filterwarnings('ignore')


trainDF = pd.read_csv(‘/content/sample_data/train.csv’)
test = pd.read_csv(‘/content/sample_data/test.csv’)


print(trainDF.head())
print(test.head())

print(trainDF.shape)
print(test.shape)


del trainDF['id']
del trainDF['f_27']

## del_test['id']
del test['f_27']

## Separate Target & other Columns
X = trainDF.drop(columns=['target'])
y = trainDF['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
bnbTesting = BernoulliNB() #Apply Classifier
bnbTesting.fit(X_train, y_train) #Fitting into model 
bnbTestingPred = bnbTesting.predict(X_test) #Prediction
bnbAcc = metrics.accuracy_score(y_test, bnbTestingPred) #Check Accuracy Score
print ("Naive Bayes Accuracy is: ", bnbAcc)


## NAIVE BAYES 
nav_clf = BernoulliNB()
nav_scores = cross_val_score(nav_clf, X_train, y_train, cv=6)
print('Naive Bayes Scores is: ',nav_scores)
nav_mean = nav_scores.mean()
print('Naive Bayes Mean Score is: ',nav_mean)


## NAIVE BAYES (LAPLACE SMOOTHING) 

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred  =  classifier.predict(X_test)

gnbAcc = metrics.accuracy_score(y_test, y_pred) # Check Accuracy Score
print ("Naive Bayes Accuracy is: ", gnbAcc)


## NAIVE BAYES  
nav_clf = GaussianNB()
nav_scores = cross_val_score(nav_clf, X_train, y_train, cv=6)
print('Naive Bayes Scores is: ',nav_scores)
nav_mean = nav_scores.mean()
print('Naive Bayes Mean Score is: ',nav_mean)
	

print(trainDF.shape)
print(test.shape)


daniyalCSV = test[['id']]

daniyalCSV


predT = test.drop(columns=['id'])

predT.head(2)

predictionOnTest = classifier.predict(predT)


print(predictionOnTest)
print(len(predictionOnTest))
daniyalCSV['target'] = predictionOnTest

daniyalCSV.head()

daniyalCSV.shape

daniyalCSV.to_csv('daniyalCSVTest.csv', index=False)



## PARAMETER TUNING

import numpy as np

param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}
## “https://medium.com/analytics-vidhya/how-to-improve-naive-bayes-9fa698e14cba”

from sklearn.model_selection import GridSearchCV

params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],
         }

bernoulli_nb_grid = GridSearchCV(BernoulliNB(), param_grid=params, n_jobs=-1, cv=10, verbose=10)
bernoulli_nb_grid.fit(X,y)

print('Train Accuracy : %.3f'%bernoulli_nb_grid.best_estimator_.score(X_train, y_train))
print('Test Accuracy : %.3f'%bernoulli_nb_grid.best_estimator_.score(X_test, y_test))
print('Best Accuracy Through Grid Search : %.3f'%bernoulli_nb_grid.best_score_)
print('Best Parameters : ',bernoulli_nb_grid.best_params_)


GaussianNB(priors=None, var_smoothing=1.0)


bnbTesting = BernoulliNB(alpha=10.0) #Classifier
bnbTesting.fit(X_train, y_train) #Training by fitting into model
bnbTestingPred = bnbTesting.predict(X_test) #Prediction
bnbAcc = metrics.accuracy_score(y_test, bnbTestingPred) #Checking Accuracy Score
print ("Naive Bayes Accuracy is: ", bnbAcc)


	

