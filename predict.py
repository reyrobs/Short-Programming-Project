from sklearn import svm
import pandas

df1 = pandas.read_csv('Analyze.csv')

df2 = pandas.read_csv('newData1.csv')

rhoncusL = [[50, 50], [51, 51]]

temp = [elem for elem in df1["Rhonc L score Rounded"]]

labels = [int(elem) for elem in df1["Rhonc L score Rounded"] if elem != '#VALUE!']

labels = labels[:12]

features = []
j=0
for i in range(0, 24, 2):
    array1 = []
    for column in df2:
        if column != "filename" and column != "Patient_ID" and temp[j] != '#VALUE!':
            array1.append(df2[column][i])
    if len(array1) > 0:
        features.append(array1)
    j+=1

print(len(features))
print(len(labels))
clf = svm.SVC(gamma='auto')
clf.fit(features, labels)
print(clf.predict([[68.75254043, 117.4538352, 0.002906671, -426.1712341, 94.73357391, 77.00539398, 55.19595718,	35.74889755]]))

