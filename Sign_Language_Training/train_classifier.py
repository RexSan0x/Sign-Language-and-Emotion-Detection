'''
The purpose of this python file is to train our classifier to classify the different hand signs
We us 'Random Forest Classifier' as our classifier.

'''

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data file
data_dict = pickle.load(open('./data2.pickle', 'rb'))

# Store the Data and labels as a varriable
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(data.shape)
tempz = np.zeros((len(data), 42))
for i in range(len(data)):
    for j in range(42):
        tempz[i, j] = data[i][j]

print(tempz.shape)

print(labels.shape)

# Split the data into testing and training data
x_train, x_test, y_train, y_test = train_test_split(tempz, labels, test_size=0.2, shuffle=True, stratify=labels)

# Make the Random Forest classifier model
model = RandomForestClassifier()

# Fit the training data with the classifier
model.fit(x_train, y_train)

# Get prediction for the test data
y_predict = model.predict(x_test)

# Calculate the accuracy and print it
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))


f = open('model2.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

