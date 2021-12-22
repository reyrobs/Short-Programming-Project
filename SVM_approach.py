from sklearn import svm
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

clf = svm.SVC(decision_function_shape='ovo', gamma=0.1)


print(np.count_nonzero(x == 3))


