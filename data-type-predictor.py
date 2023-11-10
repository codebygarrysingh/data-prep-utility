import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Create a synthetic dataset for training
n_samples = 100000
data = {
    'Column1': [1, 2, 3, 4, 5] * (n_samples // 5)
}

# Corresponding data types
data_types = {
    'Column1': 'integer'
}

# Create a DataFrame
df = pd.DataFrame(data)

# Prepare the dataset for training
X = df
y = df['Column1']  # Target column (Integer)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a classification model (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

X_test1 = np.array([["Sample"]])

# Make predictions
y_pred_1 = model.predict(X_test1)

print(f"Predictions from y_pred_1 is: {y_pred_1}")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Classification Report:")
print(report)



arch, _ = platform.architecture()

# Define the range of integers
low_int_range = 0
high_int_range = 0

# Define the range of integers
low_float_range = 0.0
high_float_range = 0.0

# Define the range of integers
low_double_range = 0.0
high_double_range = 0.0

if arch == "32bit":
    low_int_range = -2147483648
    high_int_range = 2147483647

    # Float (single-precision) range
    low_float_range = -3.402823466e38
    high_float_range = 3.402823466e38

    # Double (double-precision) range
    low_double_range = -1.79769313486e308
    high_double_range = 1.79769313486e308

    print("You are running a 32-bit system.")
elif arch == "64bit":
    low_int_range = -9223372036854775808
    high_int_range = 9223372036854775807

    # Float (single-precision) range
    low_float_range = -3.402823466e38
    high_float_range = 3.402823466e38

    # Double (double-precision) range
    low_double_range = -1.79769313486e308
    high_double_range = 1.79769313486e308
    print("You are running a 64-bit system.")
else:
    print("Architecture information not available.")


# generate a random NumPy array with 100,000 integers within the specified range
int_size = 9223372036854775807
float_size = 3.402823466e38
X_train_int = np.random.randint(low_int_range, high_int_range, int_size)
#X_train_double = np.random.uniform(low_double_range, high_double_range, size)
X_train_float = np.random.uniform(low_float_range, high_float_range, float_size)

# Cast the arrays to 'float64' data type
#X_train_double = X_train_double.astype('float64')
#X_train_float = X_train_float.astype('float64')

# Concatenate the training sets
X_train = np.concatenate((X_train_int
                          #, X_train_double
                          , X_train_float), axis=0)

# Optionally, you can create corresponding labels for each training set
# Assuming you have labels for each type of data
y_train_int = np.array(["Integer"] * size)
#y_train_double = np.array(["Double"] * size)
y_train_float = np.array(["Float"] * size)

# Concatenate the labels
y_train = np.concatenate((y_train_int
                          #, y_train_double
                          , y_train_float), axis=0)

# let's split the training and test data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
generated_training_size = X_train.shape[0]
generated_test_size = X_test.shape[0]

# Create and train a classification model (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)

model.fit(X_train.reshape((generated_training_size,1)), 
          y_train)

# Make predictions
# y_pred = model.predict(X_test.reshape((generated_test_size,1)))


# Make predictions from sample data
X_sample = np.array([[123],[456],[3.40005403e+38]])
y_sample_pred = model.predict(X_sample)

print(f"Predictions from X_test data is: {y_sample_pred}")