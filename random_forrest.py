from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report, precision_score
import pandas

np.set_printoptions(suppress=True)

df1 = pandas.read_csv('Expanded_Table.csv')

x = np.array([elem for elem in df1["Whistlings Score"]])

ids = ["PATIENT ID", "Track Name", "Rhonchus Score", "Whistlings Score", None]

temp = []

for elem in df1:
    if elem not in ids:
        temp.append(df1[elem])
temp = np.array(temp)
temp = np.transpose(temp)

x_train = x[0:90]
temp_train = temp[0:90]

clf = RandomForestClassifier(n_estimators=200)
clf.fit(temp_train, x_train)

predicted_values = clf.predict(temp[90:114])
true_values = x[90:114]

target_names = ['0', '1', '2', '3']

print(classification_report(true_values, predicted_values, target_names=target_names))
