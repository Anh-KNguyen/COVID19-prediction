from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
import matplotlib.pyplot as plt

startdate = datetime.datetime(2020, 1, 21)

def dateMapper(dateStr: str) -> int: 
    # we get a string of like '2020-04-17'
    # return the number of days since 2020-01-21
    date = datetime.datetime.strptime(dateStr, r'%Y-%m-%d')
    delta = date - startdate
    return delta.days


print("Loading training data")

data = pd.read_csv('./covid-19-data/us-states.csv')

stateMap = pd.Series(data['state'].unique()).tolist()

def stateToId(state: str) -> int:
    return stateMap.index(state)

def idToState(id: int) -> str:
    return stateMap[id]


# normalize and rank non integer features to integers
data['state'] = data['state'].map(stateToId)
data['date'] = data['date'].map(dateMapper)
data['logcases'] = np.log(data['cases'])
data['logcases'] = data['logcases'].map(lambda val: 0 if val == float('-inf') else val)

_, test = train_test_split(data, test_size=0.1) 
train = data

features = ['date', 'state']

trainingFeatures = train[features]
trainingLabels = train['logcases']

testingFeatures = test[features]
testingLabels = test['logcases']

print("Fitting Model...")

clf = svm.LinearSVR(C=0.1)
clf.fit(trainingFeatures, trainingLabels)

print("Estimating...")
predictions = clf.predict(testingFeatures)

data = pd.DataFrame(testingFeatures)
data['logcases'] = predictions
data['cases'] = np.exp(predictions)

caliP = data.loc[data['state'] == stateToId('California')]
caliP = caliP.sort_values(by='date')
plt.plot( caliP['date'], caliP['cases'],)
plt.ylabel('cases')
plt.xlabel('days since jan 21')
plt.show()



print("Result!")

err = mean_squared_error(testingLabels, predictions)

print("mse", err)