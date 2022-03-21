from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import metrics


# Read data
dataset = pd.read_csv('./bank_personal_loan.csv')
dataset.head()

# Split data into 80% for training and 20% for testing
training_data = dataset.sample(frac=0.8, random_state=25)
testing_data = dataset.drop(training_data.index)

#training_data["Log_Income"] = training_data["Income"].map(lambda i: np.log(i) if i > 0 else 0) 
#testing_data["Log_Income"] = testing_data["Income"].map(lambda i: np.log(i) if i > 0 else 0) 

# Fix features
training_data["Monthly_money"] = (training_data['Income']/12) - training_data['CCAvg']
testing_data["Monthly_money"] = (training_data['Income']/12) - training_data['CCAvg']
training_data["Log_monthly_money"] = training_data["Monthly_money"].map(lambda i: np.log(i) if i > 0 else 0) 
testing_data["Log_monthly_money"] = testing_data["Monthly_money"].map(lambda i: np.log(i) if i > 0 else 0) 

training_data["Log_Mortgage"] = training_data["Mortgage"].map(lambda i: np.log(i) if i > 0 else 0) 
testing_data["Log_Mortgage"] = testing_data["Mortgage"].map(lambda i: np.log(i) if i > 0 else 0) 

# Rename features
training_data.rename({'Personal.Loan': 'PersonalLoan'}, axis=1, inplace=True)
training_data.rename({'Securities.Account': 'SecuritiesAccount'}, axis=1, inplace=True)
training_data.rename({'CD.Account': 'CDAccount'}, axis=1, inplace=True)

testing_data.rename({'Personal.Loan': 'PersonalLoan'}, axis=1, inplace=True)
testing_data.rename({'Securities.Account': 'SecuritiesAccount'}, axis=1, inplace=True)
testing_data.rename({'CD.Account': 'CDAccount'}, axis=1, inplace=True)

# Drop unfixed features
X = training_data.drop(['PersonalLoan', 'Income', 'Mortgage', 'CCAvg', 'Monthly_money'], axis = 1)
y = training_data.PersonalLoan

X_2 = testing_data.drop(['PersonalLoan', 'Income', 'Mortgage', 'CCAvg', 'Monthly_money'], axis = 1)
y_2 = testing_data.PersonalLoan

# Test and validation data
x_train, x_val, y_train, y_val = train_test_split(X,y, test_size=0.30)

# Find the best parameters
params= {"max_depth": np.arange(1,20,1),
		"n_estimators": [1, 10, 100, 200, 500, 1000],
		"criterion": ['gini', 'entropy']}
		#"min_samples_split": [1, 2, 5, 10, 50]}
grid_search_logistic = GridSearchCV(RandomForestClassifier(), params, cv=5)
grid_search_logistic.fit(x_train,y_train)

print("tuned hpyerparameters :(best parameters) ",grid_search_logistic.best_params_)
print("accuracy :",grid_search_logistic.best_score_)

# Split data into k-folds and validate model's accuracy
stratified_k_f = StratifiedKFold(n_splits = 5)

for train_i, test_i in stratified_k_f.split(X,y):
	x_train, x_val = X.iloc[train_i], X.iloc[test_i]
	y_train, y_val = y.iloc[train_i], y.iloc[test_i]

	# Fit model
	random_for_model_2 = RandomForestClassifier(max_depth = 10, n_estimators = 200)
	random_for_model_2.fit(x_train,y_train)

	# Predict validation data
	predict_val = random_for_model_2.predict(x_val)
	predict_proba = random_for_model_2.predict_proba(x_val)[:,1]

	print ('Random Forest score validation data', accuracy_score(y_val, predict_val))

	# Predict testing data
	predict_test = random_for_model_2.predict(X_2)
	print('Random Forest score testing data: ', accuracy_score(y_2,predict_test))

# Another implementation of the stratified cross validation
'''score = cross_val_score(log_reg_model, X, y, cv = stratified_k_f)
print("Cross Validation Scores: {}".format(score))
print("Avg Cross Validation score :{}".format(score.mean()))'''

# Plot the roc curve
'''fpr, tpr, _ = metrics.roc_curve(y_val, predict_proba)
roc_score = metrics.roc_auc_score(y_val, predict_proba)
plt.figure(figsize=(6,4))
plt.suptitle("Random Forest ROC curve")
plt.plot(fpr, tpr, label = "validation" + " roc score " +str(roc_score))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()

# Plot the importance features
importances=pd.Series(random_for_model_2.feature_importances_, index=X.columns)
importances.plot(kind='barh', figsize=(12,8))
plt.show()'''
