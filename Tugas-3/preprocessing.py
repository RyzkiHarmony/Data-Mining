import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'D:\VSCode Data\Kuliah\SMT 5\Data Mining\Tugas 3\CrabAgePrediction.csv'
data = pd.read_csv(file_path)

# Checking for missing values
print("Missing values before imputation:")
print(data.isnull().sum())

# Step 1: Prepare the data for regression
# Select rows where 'Age' is not null (for training)
train_data = data[data['Age'].notnull()]

# Select features for training (dropping non-numeric and target columns)
X_train = train_data.drop(['Age', 'Sex'], axis=1)  # Dropping 'Sex' as it is categorical, and 'Age' as it is the target
y_train = train_data['Age']  # Target column (Age)

# Step 2: Train the regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Step 3: Predict missing 'Age' values
# Select rows where 'Age' is null (to fill in missing values)
missing_data = data[data['Age'].isnull()]

# Use the same features to predict the missing values
X_missing = missing_data.drop(['Age', 'Sex'], axis=1)

# Step 4: Fill in the missing 'Age' values with the predicted values
predicted_ages = reg.predict(X_missing)
data.loc[data['Age'].isnull(), 'Age'] = predicted_ages

# Step 5: Display rows where 'Age' was imputed
imputed_rows = data[data.index.isin(missing_data.index)]  # Rows where Age was originally missing
print("\nRows with imputed 'Age' values:")
print(imputed_rows)

# Optionally, save the imputed dataset to a CSV file
# data.to_csv('/mnt/data/CrabAge_filled_by_regression.csv', index=False)

# Checking for missing values after imputation
print("\nMissing values after imputation:")
print(data.isnull().sum())
