import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import pickle
import warnings
from flask import Flask, request, render_template, jsonify



###### Data Preprocesing & Cleaning #####
df = pd.read_csv('anemia.csv')
# print(df.head())
# print()
# print(df.info())
# print()
# print(df.shape)
# print()
# print(df.isnull().sum())
# print()

# results = df['Result'].value_counts()
# results.plot(kind = 'bar', color = ['green', 'blue'])
# plt.xlabel('Result')
# plt.ylabel('Frequency')
# plt.title('Count of Result')
# plt.show()

# majorclass = df[df['Result']==0]
# minorclass = df[df['Result']==1]

# major_downsample = resample(majorclass, replace=False, n_samples=len(minorclass),random_state=45)

# df = pd.concat([major_downsample,minorclass])
# print(df['Result'].value_counts())

# results = df['Result'].value_counts()
# results.plot(kind = 'bar', color = ['green', 'blue'])
# plt.xlabel('Result')
# plt.ylabel('Frequency')
# plt.title('Count of Result')
# plt.show()

#### Exploratory Data Analysis ####
# print(df.describe())

### Univariate Analysis ###

## Gender Count Visualisation ##

# output = df['Gender'].value_counts()
# ax = output.plot(kind = 'bar', color = ['orange', 'green'])
# plt.xlabel('Gender')
# plt.ylabel('Frequency')
# plt.title('Gender Count')
# for bars in ax.containers:
#     ax.bar_label(bars)
# plt.show()

## Haemoglobin Count Visualisation ##

# VISUAL ONE #
# bx = sns.histplot(df['Hemoglobin'])
# for bars in bx.containers:
#     bx.bar_label(bars)
# plt.show()

# VISUAL TWO #
# cx = sns.displot(df['Hemoglobin'], kde=True)
# plt.show()

### Bivariate Analysis ###

## Haemoglobin, Gender & Result Visualisation ##
# plt.figure(figsize=(6,6))
# mx = sns.barplot(y = df['Hemoglobin'], x = df['Gender'], hue = df['Result'])
# mx.set(xlabel = ['Male', 'Female'])
# mx.bar_label(mx.containers[0])
# mx.bar_label(mx.containers[1])
# plt.title("Mean Haemoglobin by Gender & Result")
# plt.show()

### Multivariate Analysis ###

# sns.pairplot(df)
# plt.show()

# sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
# fig = plt.gcf()
# fig.set_size_inches(10,8)
# plt.show()


### Splitting of DATASET into TRAINING & TESTING DATA ###

X = df.drop('Result', axis = 1)
Y = df['Result']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=25)

# print(X)
# print()
# print(Y)
# print()
# print("X_Train: ", x_train.shape)
# print("Y_Train: ", y_train.shape)
# print("Y_Test: ", y_test.shape)
# print("X_Test: ", x_test.shape)

### Training of MODEL on Different ML Algortihms ###

## Logistic Regression ##
lor = LogisticRegression()
lor.fit(x_train, y_train)
y_pred = lor.predict(x_test)
# print(y_pred)
# print()
acc_score_lor = accuracy_score(y_test, y_pred)
class_repo_lor = classification_report(y_test, y_pred)
# print('Accuracy Score is: ', round(acc_score_lor*100,2),'%')
# print('Classification Report for Logistic Regression is: ')
# print(class_repo_lor)
# print()
train_pred = lor.predict(x_train)
train_acc = accuracy_score(y_train, train_pred)
test_pred = lor.predict(x_test)
test_acc = accuracy_score(y_test, test_pred)
# print("Train Accuracy:", round(train_acc*100,2),'%')
# print("Test Accuracy:",round( test_acc*100,2),'%')
# print()
# print()

## Random Forest Classifier ##
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
# print(y_pred)
# print()
acc_score_rf = accuracy_score(y_test, y_pred)
class_repo_rf = classification_report(y_test, y_pred)
# print('Accuracy Score is: ', round(acc_score_rf*100,2),'%')
# print('Classification Report for Random Forest Classifier is: ')
# print(class_repo_rf)
# print()
train_pred = rf.predict(x_train)
train_acc = accuracy_score(y_train, train_pred)
test_pred = rf.predict(x_test)
test_acc = accuracy_score(y_test, test_pred)
# print("Train Accuracy:", round(train_acc*100,2),'%')
# print("Test Accuracy:",round( test_acc*100,2),'%')
# print()
# print()

## Decision Tree Classifier ##
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
# print(y_pred)
# print()
acc_score_dtc = accuracy_score(y_test, y_pred)
class_repo_dtc = classification_report(y_test, y_pred)
# print('Accuracy Score is: ', round(acc_score_dtc*100,2),'%')
# print('Classification Report for Decision Tree Classifier  is: ')
# print(class_repo_dtc)
# print()
train_pred = dtc.predict(x_train)
train_acc = accuracy_score(y_train, train_pred)
test_pred = dtc.predict(x_test)
test_acc = accuracy_score(y_test, test_pred)
# print("Train Accuracy:", round(train_acc*100,2),'%')
# print("Test Accuracy:",round( test_acc*100,2),'%')
# print()
# print()

## Gaussian Naive Bayes ##
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
# print(y_pred)
# print()
acc_score_nb = accuracy_score(y_test, y_pred)
class_repo_nb = classification_report(y_test, y_pred)
# print('Accuracy Score is: ', round(acc_score_nb*100,2),'%')
# print('Classification Report for Gaussian Naive Bayes is: ')
# print(class_repo_nb)
# print()
train_pred = nb.predict(x_train)
train_acc = accuracy_score(y_train, train_pred)
test_pred = nb.predict(x_test)
test_acc = accuracy_score(y_test, test_pred)
# print("Train Accuracy:", round(train_acc*100,2),'%')
# print("Test Accuracy:",round( test_acc*100,2),'%')
# print()
# print()

## Support Vector Machine ##
svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
# print(y_pred)
# print()
acc_score_svm = accuracy_score(y_test, y_pred)
class_repo_svm = classification_report(y_test, y_pred)
# print('Accuracy Score is: ', round(acc_score_svm*100,2),'%')
# print('Classification Report for Support Vector Machine is: ')
# print(class_repo_svm)
# print()
train_pred = svm.predict(x_train)
train_acc = accuracy_score(y_train, train_pred)
test_pred = svm.predict(x_test)
test_acc = accuracy_score(y_test, test_pred)
# print("Train Accuracy:", round(train_acc*100,2),'%')
# print("Test Accuracy:",round( test_acc*100,2),'%')
# print()
# print()

## Gradient Boosting Classifier ##
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
y_pred = gbc.predict(x_test)
# print(y_pred)
# print()
acc_score_gbc = accuracy_score(y_test, y_pred)
class_repo_gbc = classification_report(y_test, y_pred)
# print('Accuracy Score is: ', round(acc_score_gbc*100,2),'%')
# print('Classification Report for Gradient Boosting Classifier is: ')
# print(class_repo_gbc)
# print()
train_pred = gbc.predict(x_train)
train_acc = accuracy_score(y_train, train_pred)
test_pred = gbc.predict(x_test)
test_acc = accuracy_score(y_test, test_pred)
# print("Train Accuracy:", round(train_acc*100,2),'%')
# print("Test Accuracy:",round( test_acc*100,2),'%')
# print()

### PREDICTION OF ANEMIA ###
# prediction = dtc.predict([[1, 21.6, 22.3, 30.9, 74.5]])
# print(prediction[0])

# sample = pd.DataFrame([[1, 11.6, 22.3, 33.9, 44.6]], columns=x_train.columns)

# prediction = rf.predict(sample)

# if prediction[0] == 0:
#     print("The Final Report for Anemia Analysis is: You don't have ANEMIA")
# elif prediction[0] == 1:
#     print("The Final Report for Anemia Analysis is: You have ANEMIA")
# print()

model = pd.DataFrame({'Model':['Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier',
                                'Gaussian Naive Bayes', 'Support Vector Machine', 'Gradient Boosting Classifier'],
                                  'Score':[round(acc_score_lor*100,2), round(acc_score_rf*100,2), round(acc_score_dtc*100,2), 
                                           round(acc_score_nb*100,2), round(acc_score_svm*100,2), round(acc_score_gbc*100,2)]})
# print(model)

### MODEL DEPLOYMENT ###
pickle.dump(dtc,open('model.pickle','wb'))

app = Flask(__name__, static_url_path='/Flask/static')
model = pickle.load(open('model.pickle','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = [  "POST"])
def predict():
    Gender = float(request.form['Gender'])
    Hemoglobin = float(request.form['Hemoglobin'])
    MCH = float(request.form['MCH'])
    MCHC = float(request.form['MCHC'])
    MCV = float(request.form['MCV'])

    features_values = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]])

    df = pd.DataFrame(features_values, columns = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])
    print(df)
    print()
    prediction = model.predict(df)
    result = prediction[0]

    if prediction[0] == 0:
        result = "The Final Report for Anemia Analysis is: You don't have ANEMIA..!"
    elif prediction[0] == 1:
        result = "The Final Report for Anemia Analysis is: You have ANEMIA..!"
    print()

    return jsonify({"result": result}) 

if __name__ == "__main__":
    app.run(debug=True, port=5000)


