import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Load the "loan_old.csv" dataset.
data = pandas.read_csv('loan_old.csv')

#check whether there are missing values
missing_values = data.isnull().sum()
print('Missing values:\n',missing_values)

print('----------------------')

#check the type of each feature (categorical or numerical)
data_types = data.dtypes

print('Data types:\n',data_types)

print('----------------------')


#visualize a pairplot between numercial columns
sns.pairplot(data[['Income','Coapplicant_Income','Credit_History','Loan_Tenor','Max_Loan_Amount']])
plot.show()



#records containing missing values are removed
data.drop(columns=['Loan_ID'], inplace=True)
if data.isnull().values.any():
    data_cleaned_rows = data.dropna()


data_types = data_cleaned_rows.dtypes

print('----------------------')

#check whether numerical features have the same scale
print('Numerical features scale: \n')
cnt=0
for column_name in data_cleaned_rows.columns:
    if data_types.iloc[cnt] == 'int64' or data_types.iloc[cnt] == 'float64':
         print(column_name,' : ',data_cleaned_rows[column_name].max()-data_cleaned_rows[column_name].min())
    cnt += 1

print('----------------------')

label_encoder = LabelEncoder()
cnt = 0

#categorical features and targets are encoded
for column_name in data_cleaned_rows.columns:
    if data_types.iloc[cnt] == 'object':
        data_cleaned_rows.loc[data_cleaned_rows.index, column_name] = label_encoder.fit_transform(
            data_cleaned_rows[column_name])
    cnt += 1




#the features and targets are separated
x=data_cleaned_rows.drop(columns=['Max_Loan_Amount','Loan_Status'])
y=data_cleaned_rows[['Max_Loan_Amount','Loan_Status']]

#the data is shuffled and split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

#numerical features are standardized
cnt=0
for column_name in x_train.columns:
    if data_types.iloc[cnt] == 'int64' or data_types.iloc[cnt] == 'float64':
        x_train[column_name]=(x_train[column_name]-x_train[column_name].mean())/x_train[column_name].std()
        x_test[column_name] = (x_train[column_name] - x_train[column_name].mean()) / x_train[column_name].std()
    cnt += 1


# y_train['Max_Loan_Amount'] = (y_train['Max_Loan_Amount'] - y_train['Max_Loan_Amount'].mean()) / y_train['Max_Loan_Amount'].std()
# y_test['Max_Loan_Amount'] = (y_train['Max_Loan_Amount'] - y_train['Max_Loan_Amount'].mean()) / y_train['Max_Loan_Amount'].std()


#Convert data to Numpy array
x_train = x_train.to_numpy().reshape((-1,9))
x_test = x_test.to_numpy().reshape((-1,9))
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()







