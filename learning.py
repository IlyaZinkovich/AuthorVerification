import csv
from sklearn.feature_extraction.text import TfidfTransformer

sample_names = []
features = []
known = [[]]
unknown = [[]]

def read_matrix(reader, row_count):
    counter = 0
    matrix = [[]]
    for row in reader:
        matrix[len(matrix) - 1] = row
        counter = counter + 1
        if (counter == row_count):
            break
        else:            
            matrix.append([])
    return matrix
    

with open('features.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    sample_names = reader.next()
    features = reader.next()
    known = read_matrix(reader, len(sample_names))
    unknown = read_matrix(reader, len(sample_names))
    

transformer = TfidfTransformer()

import numpy as np

known = np.array(known).astype(np.int)
unknown = np.array(unknown).astype(np.int)

count = len(sample_names)

learn_samples = np.vstack((known[:count,:], unknown[:count,:]))

tfidf = transformer.fit_transform(learn_samples)

tfidf_matrix = tfidf.toarray()

for i in range(count):
    row = np.absolute(np.subtract(tfidf_matrix[i], tfidf_matrix[count + i]))
    if i == 0:
        learn = row 
    else:
        learn = np.vstack((learn, row))

data_path = './authorship-verification-dataset/'
with open(data_path + 'truth.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    truth = []
    for row in reader:
        truth.append(1 if row[1] == 'Y' else 0)
        
truth = np.array(truth)


#from sklearn.svm import SVR

#print clf.predict(learn)


#from sklearn.ensemble import RandomForestRegressor

#estimator = RandomForestRegressor(random_state=0, n_estimators=2)
#estimator = SVR(C=2.0, epsilon=0.4)
#estimator.fit(learn[:50, :], truth[:50])

#predicted = estimator.predict(learn[50:, :])

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=10)
ada = AdaBoostClassifier()

forest.fit(learn[:50, :], truth[:50])
ada.fit(learn[:50, :], truth[:50])

from sklearn.metrics import confusion_matrix
print confusion_matrix(truth[50:], forest.predict(learn[50:, :]))
print confusion_matrix(truth[50:], ada.predict(learn[50:, :]))
