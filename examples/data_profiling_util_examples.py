# Author: Garry Singh
# Date: 2023-11-16
# Description: This script has some examples for functions to profile dataframes

# Import necessary libraries
from pandas import DataFrame
from data_read_util import infer_data_types
from data_profiling_util import check_data_completeness, check_data_spread, check_data_uniqueness, check_categorical_data, calculate_correlation_matrix

# Create a sample DataFrame
data = {
    'Age': [25, 30, 35, 28, None, 32, 22],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'Salary': [50000, 60000, 75000, 55000, 65000, None, 48000],
    'City': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Houston', 'Miami', 'Seattle']
}

df = DataFrame(data)

# Define the profiling functions here (previously defined)

# Example usage of check_data_completeness:
completeness_info = check_data_completeness(df, missing_threshold=10.0)
for column, info in completeness_info.items():
    print(f"Column: {column}")
    print(f"Present Values: {info['present_values']}")
    print(f"Missing Values: {info['missing_values']}")
    print(f"Completeness Percentage: {info['completeness_percentage']:.2f}%")
    print(f"Empty Percentage: {info['empty_percentage']:.2f}%")
    print(f"Flagged: {info['flagged']}")

# Example usage of check_data_spread:
spread_info = check_data_spread(df)
for column, info in spread_info.items():
    print(f"Column: {column}")
    print(f"Spread Pattern: {info['spread_pattern']}")
    print(f"Spread Percentage: {info['spread_percentage']:.2f}%")
    print(f"Is Uniform Spread: {info['is_uniform_spread']}")
    print(f"Most Occurring Value: {info['most_occuring_value']}")
    print(f"Least Occurring Value: {info['least_occuring_value']}")

# Example usage of check_data_uniqueness:
uniqueness_info = check_data_uniqueness(df)
for column, info in uniqueness_info.items():
    print(f"Column: {column}")
    print(f"Uniqueness Percentage: {info['uniqueness_percentage']:.2f}%")
    print(f"Total Unique Values: {info['total_unique_values']}")
    print(f"Is Unique: {info['is_unique']}")

# Example usage of check_categorical_data:
categorical_info = check_categorical_data(df)
for column, info in categorical_info.items():
    print(f"Column: {column}")
    print(f"Is Categorical: {info['is_categorical']}")
    print(f"Categories: {', '.join(info['categories'])}")

# Example usage of calculate_correlation_matrix:
correlation_matrix = calculate_correlation_matrix(df)
print("Correlation Matrix:")
print(correlation_matrix)
