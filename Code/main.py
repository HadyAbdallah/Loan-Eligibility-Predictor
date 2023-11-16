import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score



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
print('Numerical features describe: \n')

numerical_column_name=['Income','Coapplicant_Income','Loan_Tenor']
describe_data=data_cleaned_rows[numerical_column_name].describe()
print(describe_data)


print('----------------------')

label_encoder = LabelEncoder()
cnt = 0

#categorical features and targets are encoded
for column_name in data_cleaned_rows.columns:
    if data_types.iloc[cnt] == 'object':
        data_cleaned_rows.loc[:, column_name] = label_encoder.fit_transform(data_cleaned_rows[column_name])
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

x_train_mean=x_train[numerical_column_name].mean()
x_train_std=x_train[numerical_column_name].std()
x_test[numerical_column_name] = (x_test[numerical_column_name] - x_train_mean) / x_train_std
x_train[numerical_column_name] = (x_train[numerical_column_name] - x_train_mean) / x_train_std


#Convert data to Numpy array
x_train = x_train.to_numpy().reshape((-1,9))
x_test = x_test.to_numpy().reshape((-1,9))
y_train_max_loan = y_train_max_loan.to_numpy()
y_test_max_loan = y_test_max_loan.to_numpy()
y_train_loan_status = y_train_loan_status.to_numpy()
y_test_loan_status = y_test_loan_status.to_numpy()

#Fit a linear regression model
print("linear regression model: ")
model = linear_model.LinearRegression()
model.fit(x_train,y_train_max_loan)

print('Coefficients: \n', model.coef_, " ", model.intercept_)


#predict the loan amount
y_pred = model.predict(x_test)

r2 = r2_score(y_test_max_loan, y_pred)
print("R-squared score:", r2)

print('----------------------')
print('logistic regression model:')
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
        if i % 200 == 0:
            print(f"Cost after iteration {i}: {cost}")
    return w, b
# Display the shapes of the training data (just for debugging)
#print("X_train shape:", x_train.shape)
#print("y_train shape:", y_train_loan_status.shape)

# Initialize parameters before training
w, b = initialize_parameters(x_train.shape[1])
# Set hyperparameters
learning_rate = 0.01
num_iterations = 2000
# Train the logistic regression modely_train_loan_status
w, b = gradient_descent(x_train,y_train_loan_status , w, b, learning_rate, num_iterations)
# Print the trained parameters
print("Trained weights:", w)
print("Trained bias:", b)


# calculate accuracy function
def Accuracy(X, y, w, b):
    predictions = predict(X, w, b)
    predictions_as_binary = ((predictions >= 0.5).astype(int)).reshape(-1)
    correct_predictions = (predictions_as_binary == y).sum()
    accuracy = (correct_predictions / len(y))*100
    return accuracy


accuracy = Accuracy(x_test, y_test_loan_status, w, b)
print("Accuracy: ",format(accuracy, ".2f"),'%')



#load_new analysis and preprocessing part
# Load the "loan_new.csv" dataset.
print("------------------------Loan_new.csv Part----------------------")
data = pandas.read_csv('loan_new.csv')

# check whether there are missing values
missing_values = data.isnull().sum()
print('Missing values:\n', missing_values)



# records containing missing values are removed and drop Loan_ID column
data.drop(columns=['Loan_ID'], inplace=True)
if data.isnull().values.any():
    newdata_cleaned_rows = data.dropna()



label_encoder = LabelEncoder()
count = 0

# categorical features are encoded
for column_name in newdata_cleaned_rows.columns:
    if data_types.iloc[count] == 'object':
        newdata_cleaned_rows.loc[:, column_name] = label_encoder.fit_transform(
            newdata_cleaned_rows[column_name])
    count += 1


# numerical values are standardized
newdata_cleaned_rows.loc[:,numerical_column_name] = (newdata_cleaned_rows[numerical_column_name]-x_train_mean)/x_train_std
x_new = newdata_cleaned_rows.to_numpy()


# use models to predict loan_Amount and status
loan_amount_prediction = model.predict(x_new)
loan_amount_prediction =[0 if i <0 else i for i in loan_amount_prediction]
status_prediction = predict(x_new, w, b)
status_prediction_YorN = ['Y' if prob >= 0.5 else 'N' for prob in status_prediction]

print("------------------------------------------")
print("prediction of loan Amount:\n",loan_amount_prediction)
print("---------------------------------------")
print("prediction of loan status:\n",status_prediction_YorN)
