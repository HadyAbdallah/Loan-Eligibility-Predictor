import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


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
#plot.show()



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
y_train_max_loan = y_train['Max_Loan_Amount']
y_test_max_loan = y_test['Max_Loan_Amount']
y_train_loan_status = y_train['Loan_Status']
y_test_loan_status = y_test['Loan_Status']


#numerical features are standardized
cnt=0
for column_name in x_train.columns:
    if data_types.iloc[cnt] == 'int64' or data_types.iloc[cnt] == 'float64':
        x_train[column_name]=(x_train[column_name]-x_train[column_name].mean())/x_train[column_name].std()
        x_test[column_name] = (x_test[column_name] - x_test[column_name].mean()) / x_test[column_name].std()
    cnt += 1

# y_train['Max_Loan_Amount'] = (y_train['Max_Loan_Amount'] - y_train['Max_Loan_Amount'].mean()) / y_train['Max_Loan_Amount'].std()
# y_test['Max_Loan_Amount'] = (y_train['Max_Loan_Amount'] - y_train['Max_Loan_Amount'].mean()) / y_train['Max_Loan_Amount'].std()


#Convert data to Numpy array
x_train = x_train.to_numpy().reshape((-1,9))
x_test = x_test.to_numpy().reshape((-1,9))
y_train_max_loan = y_train_max_loan.to_numpy()
y_test_max_loan = y_test_max_loan.to_numpy()


#Fit a linear regression model
model = linear_model.LinearRegression()
model.fit(x_train,y_train_max_loan)

print('Coefficients: \n', model.coef_, " ", model.intercept_)


#predict the loan amount
y_pred = model.predict(x_test)

r2 = r2_score(y_test_max_loan, y_pred)
print("R-squared score:", r2)

#logistic regression model
'''
Logistic regression Algorithm: σ(z)
1. Define the Sigmoid Function
2. Initialize Parameters (θ and B)
3. Compute the Linear Combination: z = θ1x1+θ2x2+…+θnx n+b
4. Apply the Sigmoid Function: y = σ(z)
5. Define the Cost Function: J(θ)=− 1/m∑[y(i)log(y)+(1−y)log(1− y)]
6. Gradient Descent
    θj=θj−α∂j/∂θj
    b = b + −α∂j/∂b
'''
def sigmoid(z):
    z = np.array(z,dtype=float)
    return 1 / (1 + np.exp(-z))
def initialize_parameters(dim):
    # Initialize weights and bias to zero
    theta = np.zeros((1, dim))
    b = 0
    return theta, b
def linear_combination(X, w, b):
    return np.dot(X, w.T) + b
def compute_cost(y, y_hat):
    m = len(y)
    return -1/m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
def predict(X, w, b):
    z = linear_combination(X, w, b)
    return sigmoid(z)
def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        # Compute linear combination
        z = linear_combination(X, w, b)
        # Apply sigmoid function and reshape
        y_hat = sigmoid(z).reshape(-1)
        # Compute cost
        cost = compute_cost(y, y_hat)
        # Compute gradients
        dw = 1/m * np.dot(X.T, (y_hat - y))
        db = 1/m * np.sum(y_hat - y)
        # Update parameters
        w -= (learning_rate * dw.T).astype(float) # Transpose dw before updating weights
        b -= learning_rate * db
        # Print cost every 100 iterations
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    return w, b
# Display the shapes of the training data (just for debugging)
#print("X_train shape:", x_train.shape)
#print("y_train shape:", y_train_loan_status.shape)

# Initialize parameters before training
w, b = initialize_parameters(x_train.shape[1])
# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000
# Train the logistic regression model
w, b = gradient_descent(x_train, y_train_loan_status, w, b, learning_rate, num_iterations)
# Print the trained parameters
print("Trained weights:", w)
print("Trained bias:", b)

#load_new analysis and preprocessing part
# Load the "loan_new.csv" dataset.
print("------------------------Loan_new.csv Part----------------------")
data = pandas.read_csv('loan_new.csv')

# check whether there are missing values
missing_values = data.isnull().sum()
print('Missing values:\n', missing_values)

print('----------------------')
data_types = data.dtypes

print('Data types:\n', data_types)

print('----------------------')

# visualize a pairplot between numerical columns
sns.pairplot(data[['Income', 'Coapplicant_Income', 'Credit_History', 'Loan_Tenor']])
# plot.show()

# records containing missing values are removed
data.drop(columns=['Loan_ID'], inplace=True)
if data.isnull().values.any():
    data_cleaned_rows = data.dropna()

data_types = data_cleaned_rows.dtypes


# check whether numerical features have the same scale
print('Numerical features scale: \n')
count = 0
for column_name in data_cleaned_rows.columns:
    if data_types.iloc[count] == 'int64' or data_types.iloc[count] == 'float64':
        print(column_name, ' : ', data_cleaned_rows[column_name].max() - data_cleaned_rows[column_name].min())
    count += 1

print('----------------------')

label_encoder = LabelEncoder()
count = 0

# categorical features are encoded
for column_name in data_cleaned_rows.columns:
    if data_types.iloc[count] == 'object':
        data_cleaned_rows.loc[data_cleaned_rows.index, column_name] = label_encoder.fit_transform(
            data_cleaned_rows[column_name])
    count += 1

# numerical features are standardized
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

