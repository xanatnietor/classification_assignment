import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import metrics


dataset = pd.read_csv('./bank_personal_loan.csv')
dataset.head()

training_data = dataset.sample(frac=0.8, random_state=25)
testing_data = dataset.drop(training_data.index)

#print(training_data.columns)

# Grouping ages from 10 to 100
#bins_age = np.arange(10, 100, 10)
#bins_exp = [2, 5, 10, 15, 20, 30, 40]
bins_income = [10,20,30,40,50,70,90,110,150,250]
training_data['category'] = np.digitize(training_data.Income, bins_income, right=True)
counts = training_data.groupby(['category']).Income.count()

#print(training_data['category'].unique())


# Select the positive values from Experience to calculate the mode
experience_var = []
for t in range(0,len(training_data['Experience'])):
	if(training_data['Experience'].iloc[t] > 0 and training_data['Age'].iloc[t] >= 20 and training_data['Age'].iloc[t] < 30):
		#print(training_data['Experience'].iloc[t] , '-',  training_data['Age'].iloc[t])
		experience_var.append(training_data['Experience'].iloc[t])

# Print the Experience values mode
#print(stats.mode(experience_var))

# Plot the data
bins_age = np.arange(10, 100, 10)
training_data['category'] = np.digitize(training_data.Age, bins_age, right=True)
counts = training_data.groupby(['category']).Age.count()

bars_ages = ('50-59', '40-49', '30-39', '20-29', '60-69')
x_pos = np.arange(len(bars_ages))
print(x_pos)
training_data['category'].value_counts(normalize=True).plot.bar(figsize=(4,3), title='Age', color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(x_pos, bars_ages)
plt.ylabel('Percentage')
plt.xlabel('Group of ages')
plt.show()

bins_income = [10,20,30,40,50,70,90,110,150,250]
training_data['category'] = np.digitize(training_data.Income, bins_income, right=True)
counts = training_data.groupby(['category']).Income.count()

bars_income = ('0-10','11-20','21-30','31-40','41-50','51-70','71-90','91-110','111-150','151+')
x_pos = np.arange(len(bars_income))
training_data['category'].value_counts(normalize=True).plot.bar(figsize=(4,3), title='Income', color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(x_pos, bars_income)
plt.ylabel('Percentage')
plt.xlabel('Group of income')
plt.show()

bins_exp = [2, 5, 10, 15, 20, 30, 40]
training_data['category'] = np.digitize(training_data.Experience, bins_exp, right=True)
counts = training_data.groupby(['category']).Experience.count()
bars_exp = ('0-2', '3-5', '6-10', '11-15', '16-20', '21-30', '31-40', '41+')
x_pos = np.arange(len(bars_exp))
training_data['category'].value_counts(normalize=True).plot.bar(figsize=(4,3), title='Experience', color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(x_pos, bars_exp)
plt.ylabel('Percentage')
plt.xlabel('Group of experience')
plt.show()

# Plot Family, Education, PersonalLoan, SecuritiesAccount, CDAccount, Online, CreditCard
bars_edu = ('No', 'Yes')
x_pos = np.arange(len(bars_edu))
training_data['Personal.Loan'].value_counts(normalize=True).plot.bar(figsize=(3,4), title='PersonalLoan', color=(0.2, 0.4, 0.6, 0.6))
plt.ylabel('Percentage')
plt.xticks(x_pos, bars_edu)
plt.xlabel('customer accepted a personal load offered')
plt.show()

bars_edu = ('Undergraduate', 'Graduate', 'Advanced')
x_pos = np.arange(len(bars_edu))
training_data['Education'].value_counts(normalize=True).plot.bar(figsize=(3,4), title='Education', color=(0.2, 0.4, 0.6, 0.6))
plt.ylabel('Percentage')
plt.xticks(x_pos, bars_edu)
plt.xlabel('most recent educational achievement level')
plt.show()

training_data.boxplot(column='Income', by = 'CreditCard')
plt.suptitle("Income vs Credit Card boxplot")
plt.show()

# Plot the income, ZIPCode, CCAvg, Mortgae
monthly_income = training_data['Income']/12 - training_data['CCAvg']
monthly_income = monthly_income.map(lambda i: np.log(i) if i > 0 else 0) 
sns.distplot(monthly_income)
plt.xlabel('Value of monthly money without credit card')
plt.suptitle("Log Monthly money")
plt.show()
monthly_income.plot.box(figsize=(4,5))
plt.suptitle("Log monthly money boxplot")
plt.xlabel('Value of monthly money without credit card')
plt.show()

plt.plot(training_data['Personal.Loan'], training_data['ZIP.Code'], '.', color='salmon')
plt.xlabel('ZIP code')
plt.suptitle("ZIP code vs Personal Loan")
plt.ylabel('Loan')
plt.show()

# Plot categorical vs loan
Family = pd.crosstab(training_data['Family'],training_data['Personal.Loan'])
Family.div(Family.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
Education = pd.crosstab(training_data['Education'],training_data['Personal.Loan'])
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
CDAccount = pd.crosstab(training_data['CD.Account'],training_data['Personal.Loan'])
CDAccount.div(CDAccount.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
CreditCard = pd.crosstab(training_data['CreditCard'],training_data['Personal.Loan'])
CreditCard.div(CreditCard.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
Online = pd.crosstab(training_data['Online'],training_data['Personal.Loan'])
Online.div(Online.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
Securities = pd.crosstab(training_data['Securities.Account'],training_data['Personal.Loan'])
Securities.div(Securities.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()

# Plot numerical vs loan
training_data.groupby('Personal.Loan')['Income'].mean().plot.bar()
plt.xlabel('Personal Loan')
plt.ylabel('Income')
plt.suptitle("Personal Loan vs Income")
plt.show()

bins_income = np.arange(9000, 100000, 1000)
#bars_income = ['50-59', '40-49', '30-39', '20-29', '60-69']
training_data['bins_income']=pd.cut(training_data['ZIP.Code'],bins_income)#,labels=bars_income)
Income_bin = pd.crosstab(training_data['bins_income'],training_data['Personal.Loan'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
plt.xlabel('ZIP Code')
plt.ylabel('Percentage')
plt.show()

#training_data = training_data.drop(['category'], axis=1)

matrix = training_data.corr()
f, ax = plt.subplots(figsize=(12,8))
sns.heatmap(matrix,vmax=.8,square=True,cmap="BuPu", annot = True)
plt.suptitle("Correlation matrix before fixed values")
plt.show()

# Cehck for null information
#print(testing_data.isnull().sum())

m = training_data['Mortgage']
training_data['Income_Log']=np.log(m, out=np.zeros_like(m), where=(m!=0))
plt.suptitle("Mortgage histogram")
training_data['Mortgage'].hist(bins=20)
plt.show()
training_data['Income_Log'].hist(bins=20)
plt.suptitle("Log income histogram")
plt.show()
#testing_data['Income_Log']=np.log(testing_data['Income'])

Q1 = training_data.quantile(0.25)
Q3 = training_data.quantile(0.75)
IQR = Q3 - Q1
#print(IQR)

print(training_data['Income'].skew())
print(training_data['Income'].describe())
training_data.boxplot(column='Income', by='Personal.Loan')
plt.show()

training_data["Log_Mortgage"] = training_data["Mortgage"].map(lambda i: np.log(i) if i > 0 else 0) 
testing_data["Log_Mortgage"] = testing_data["Mortgage"].map(lambda i: np.log(i) if i > 0 else 0) 

training_data["Monthly_money"] = (training_data['Income']/12) - training_data['CCAvg']
testing_data["Monthly_money"] = (training_data['Income']/12) - training_data['CCAvg']
training_data["Log_monthly_money"] = training_data["Monthly_money"].map(lambda i: np.log(i) if i > 0 else 0) 
testing_data["Log_monthly_money"] = testing_data["Monthly_money"].map(lambda i: np.log(i) if i > 0 else 0)

X = training_data.drop(['Income', 'Mortgage', 'CCAvg', 'Monthly_money'], axis = 1)

matrix = X.corr()
f, ax = plt.subplots(figsize=(12,8))
sns.heatmap(matrix,vmax=.8,square=True,cmap="BuPu", annot = True)
plt.suptitle("Correlation matrix with fixed features")
plt.show()

'''print(training_data['Income'].skew())
print(training_data['Log_Income'].skew())
print(training_data['Log_Income'].describe())'''

training_data['Monthly_money'].hist(bins=20)
plt.suptitle("Monthly money histogram")
plt.show()
training_data['Log_monthly_money'].hist(bins=20)
plt.suptitle("Log monthly money histogram")
plt.show()

#Plot for two numerical variables
fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(training_data['Mortgage'], training_data['Personal.Loan'])
ax.set_xlabel('Mortgage')
ax.set_ylabel('Loan')
plt.suptitle("Mortgage vs loan scatterplot")
plt.show()
