import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import timeit

#Create your df here:
df = pd.read_csv('profiles.csv')
print(df.columns)

#combine essay columns --first replace NaNs with blanks then join together 
essay_cols = ['essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9'] 
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

#####Create religion features#####

#fill religion NaNs
df = df.dropna(subset = ['religion'])
print(df.religion.value_counts())

#get first word(religion type) from religion choices whether serious or not.
df['religion_type'] = df.religion.str.split(n=1).str[0]
print(df.religion_type.value_counts())
#give values to religion based on number of people ... no arbirtrary rating system, just a number assignment.
df['religion_vals'] = df.religion_type.map({'agnosticism':10,'other':9,'atheism':8,'christianity':7,'catholicism':6,'catholicism':5,'judaism':4,'buddhism':3,'hinduism':2,'islam':1})

##histogram of religions
plt.hist(df.religion_type, edgecolor='white')
plt.xticks(rotation = 45)
plt.title('Summarized Responses to Religion Question')
plt.xlabel('Religions')
plt.ylabel('People')
plt.show()

#breakout of male/female per religion
religion_temptations = df.groupby(['religion_type','sex']).size().unstack()
religion_temptations.plot(kind='barh',stacked=True, figsize=[16,6])

#fill drug use NaNs with 'experimented'
df['drugs'] = df['drugs'].fillna('experimented')
#print(df.drugs.value_counts())
#give drug usage values
df['drug_vals'] = df.drugs.map({'never':0,'experimented':1,'sometimes':2,'often':3})

#fill alcohol consumption NaNs with 'maybe so and maybe not'
df['drinks'] = df.drinks.fillna('maybe so and maybe not')
#map booze consumption by intensity ... maybe so and maybe not defaults to 2 ... my decision.
df['drinks_vals'] = df.drinks.map({'socially':3,'rarely':1,'often':4,'not at all':0,'maybe so and maybe not':2,'very often':5,'desperately':6})

#fill smokes NaNs
df['smokes'] = df.smokes.fillna("what momma don't know don't hurt her")
#assign values for how often they light up.
df['smokes_vals'] = df.smokes.map({'no':0,"what momma don't know don't hurt her":1,'sometimes':2,'when drinking':3,'trying to quit':4,'yes':5})

###Normalize data, define feature data and labels###
feature_data = df[['smokes_vals', 'drinks_vals','drug_vals']]
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
labels = df['religion_vals']

#####KNN Religion Classifier#####

#timer variable for KNN
total = 0
start = timeit.default_timer()

#split data between training and testing then run model
train_data, test_data, train_labels, test_labels = train_test_split(feature_data, labels, test_size = .2, random_state = 1)
classifier = KNeighborsClassifier(n_neighbors=166)
classifier.fit(train_data, train_labels)
#stop timer
stop = timeit.default_timer()
total += stop - start

#print accuracy and time
print('KNN validation accuracy:', classifier.score(test_data, test_labels))
print('Time to run KNN:', total)

###this is where I looped though neighbors.
#scores = []
#for k in range(155, 175):
#    classifier = KNeighborsClassifier(n_neighbors = k)
#    classifier.fit(train_data, train_labels)
#    scores.append(classifier.score(test_data, test_labels))    
#print(scores.index(max(scores)))

#####SVC Religion Classifier#####

#create dataset for SVM Model
dataset = df[['religion', 'smokes_vals', 'drinks_vals','drug_vals']]

#create classifier object and call fit
total = 0
start = timeit.default_timer()
#create training and validation set
training_set, validation_set = train_test_split(dataset, random_state = 1)

#split data between training and testing then run model
classifier = SVC(kernel = 'rbf', gamma = .5, C = .125)
classifier.fit(training_set[['smokes_vals','drinks_vals','drug_vals']], training_set.religion)
#stop timer
stop = timeit.default_timer()

print('SVM Validation Accuracy:',classifier.score(validation_set[['smokes_vals','drinks_vals','drug_vals']],validation_set.religion))
total += stop - start
print('Time to run SVM:', total)


#####Regression#####

#get rid of -1s as answers.
df_inc = df[df.income != -1]
print(df_inc.income.value_counts())

#get rid of NaNs in smokes and drinks columns
df_inc = df_inc.dropna(subset=['drinks', 'smokes'])

#plot income vs smoking
plt.scatter(df_inc.income, df_inc.smokes, alpha = 0.1)
plt.xlabel('Income')
plt.ylabel('Smokes')
plt.title('Income vs Smoking')
plt.show()

#plot income vs drinking
plt.scatter(df_inc.income, df_inc.drinks, alpha = 0.1)
plt.xlabel('Income')
plt.ylabel('Drinks')
plt.title('Income vs Drinking')
plt.show()

x = df_inc[['drinks_vals','drug_vals']]
y = df_inc.income

start = timeit.default_timer()
#####Multiple Linear Regression#####
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, test_size = 0.2, random_state = 1)
incMLR = LinearRegression()
incMLR.fit(x_train, y_train)
stop = timeit.default_timer()
y_predict = incMLR.predict(x_test)

print('Time to run MLR:', stop - start)
print('MLR Score:', incMLR.score(x_train, y_train))
print('MLR Score:', incMLR.score(x_test, y_test))
print('MLR Coefficients:', incMLR.coef_)

plt.scatter(y_test, y_predict)
plt.xlabel('Actual Income')
plt.ylabel('Predicted Income')
plt.title('Actual vs. Predicted Income')
plt.plot(range(200000),range(200000))
plt.show()

#####KNN Regressor#####
KNN_scores = []
for n in range(100, 200):
    incRegressor = KNeighborsRegressor(n_neighbors = n, weights = 'distance')
    incRegressor.fit(x,y)
    KNN_scores.append(incRegressor.score(x,y))

#time running of KNN regressor
total = 0
start = timeit.default_timer()
incRegressor = KNeighborsRegressor(n_neighbors = 104, weights = 'distance')
incRegressor.fit(x,y)
stop = timeit.default_timer()

print('Time to run KNN Regressor:', stop - start)
print('KNN Regressor Score:', incRegressor.score(x,y))

#plot KNN Regressor Scores to find best N
plt.plot(range(100,200), KNN_scores)
plt.xlabel('Neighbors')
plt.ylabel('Score')
plt.title('Neighbors vs Model Score')
plt.show()

#barchart groupings
#inc_smk_drnk = df_inc.groupby(['income', 'smokes', 'drinks']).size().unstack()
#inc_smk_drnk.plot(kind='barh',stacked=True, figsize=[20,20])

