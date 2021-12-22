from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import pandas

np.set_printoptions(suppress=True)
df1 = pandas.read_csv('Expanded_Table.csv')

x = np.array([elem for elem in df1["Rhonchus Score"]])
x_1 = np.array([elem for elem in df1["Whistlings Score"]])

features1 = np.array([elem for elem in df1["spectral_centroid"]])
features2 = np.array([elem for elem in df1["spectral_rolloff"]])
features3 = np.array([elem for elem in df1["ZCR"]])
features4 = np.array([elem for elem in df1["mfcc1"]])
features5 = np.array([elem for elem in df1["mfcc2"]])
features6 = np.array([elem for elem in df1["mfcc3"]])
features7 = np.array([elem for elem in df1["mfcc4"]])
features8 = np.array([elem for elem in df1["mfcc5"]])

final = np.column_stack((features1, features2, features3, features4, features5,
        features6, features7, features8
        ))

ids = ["PATIENT ID", "Track Name", "Rhonchus Score", "Whistlings Score", None]

temp = []

for elem in df1:
    if elem not in ids:
        temp.append(df1[elem])
temp = np.array(temp)
temp = np.transpose(temp)

# x_1 = np.array(x[0:100])
# temp = np.array(temp[0:100])

x_2 = np.array([elem for elem in df1["mfcc3"]])

x_2 = np.reshape(x_2, (len(x_2), 1))

X_train, X_test, Y_train, Y_test = train_test_split(temp, x, test_size=0.3, random_state=0)

clf = svm.SVC(kernel='linear', decision_function_shape='ovr', C=1).fit(X_train, Y_train)

print(clf.score(X_test, Y_test))

"""
Cross validation
"""

# clf = svm.SVC(kernel='linear', C=1, random_state=42)
#
# scores = cross_val_score(clf, temp, x, cv=5)
#
# print(scores.mean())

