import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn import datasets

df = pd.read_csv('/Users/justinduan/Documents/MLProj1/train.csv')
df = df.dropna()
df_age = df['Age'].values
df_class = df['Pclass'].values
features = list(zip(df_age, df_class))
le = preprocessing.LabelEncoder()
label = le.fit_transform(df['Survived'].values)


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)  # 70% training and 30% test

knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))