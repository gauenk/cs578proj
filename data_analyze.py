from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np

input_data = "~/CS578/proj/data/data_no90s.csv"

df = pd.read_csv(input_data)

#The predicted variable is crack cocaine. 
#We have to remove the predicted variable from the feature set
cocaine = df["CRKCOC"]
df.drop("CRKCOC", 1)

#Turn data into numpy arrays
y = cocaine.as_matrix()
X = df.as_matrix()

clf = LinearSVC(penalty = "l1", dual = False)
clf.fit(X, y)