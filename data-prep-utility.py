# Author: Garry Singh
# Date: 2023-11-16
# Description: This script defines functions to read, analyze, and prepare a report on a DataFrame.

from pandas import DataFrame
import pandas as pd
import os

def read_data(input_file: str) -> pd.DataFrame:
    """
    Reads data from various file formats and returns a DataFrame.
    
    Parameters:
    - input_file: str, the path to the input file or a DataFrame.
    
    Returns:
    - pd.DataFrame containing the data from the input file or the provided DataFrame.
    
    Raises:
    - ValueError: If the input file format is unsupported or the input source type is unsupported.
    - Exception: If an error occurs while reading the data.
    """
    try:
        if isinstance(input_file, str):
            if input_file.endswith('.csv'):
                # Read data as string columns without inferring schema
                df = pd.read_csv(input_file)
            elif input_file.endswith(('.xls', '.xlsx')):
                # Read data as string columns without inferring schema
                df = pd.read_excel(input_file)
            elif input_file.endswith('.json'):
                df = pd.read_json(input_file)
            else:
                raise ValueError("Unsupported file format.")
        elif isinstance(input_file, pd.DataFrame):
            df = input_file
        else:
            raise ValueError("Unsupported input source type")

        return df  # Return the DataFrame

    except Exception as e:
        print(f"An error occurred while reading the input data: {str(e)}")
        return None

def infer_data_types(df: DataFrame):
    """
    Infers the data types of columns in a DataFrame.
    
    Parameters:
    - df: pd.DataFrame, the DataFrame to analyze.
    
    Returns:
    - Dictionary containing the inferred data types for each column.
    """
    data_types = df.dtypes.to_dict()

    # Dictionary to store actual data types for columns with 'object' data type
    actual_types = {}
    infer_data_type = {}

    for column, dtype in data_types.items():
        if dtype == 'object':
            # Attempt to find the actual data type for 'object' columns
            try:
                # Try converting to datetime with a specified format
                actual_data_type = pd.to_datetime(df[column], format='%Y-%m-%d %H:%M', errors='coerce').dtype
                actual_types[column] = str(actual_data_type)
            except ValueError:
                try:
                    # Try converting to numeric
                    actual_data_type = pd.to_numeric(df[column], errors='coerce').dtype
                    actual_types[column] = str(actual_data_type)
                except ValueError:
                    # Unable to infer the actual data type
                    actual_types[column] = 'str'
        else:
            # For all other data types, use the detected data type
            actual_types[column] = str(dtype)

        infer_data_type[column] = {
            "original_data_type": data_types[column],
            "inferred_data_type": actual_types[column]
        }

    return infer_data_type

def check_data_completeness(df: DataFrame):
    """
    Checks data completeness in a DataFrame.
    
    Parameters:
    - df: pd.DataFrame, the DataFrame to analyze.
    
    Returns:
    - Dictionary containing completeness information for each column.
    """
    total_rows = df.shape[0]
    completeness_results = {}

    for column in df.columns:
        non_empty_count = df[column].count()
        present_values = non_empty_count
        completeness_percentage = (non_empty_count / total_rows) * 100
        empty_percentage = 100 - completeness_percentage
        missing_values = total_rows - non_empty_count

        completeness_results[column] = {
            'present_values': present_values,
            'missing_values': missing_values,
            'completeness_percentage': completeness_percentage,
            'empty_percentage': empty_percentage
        }
    
    return completeness_results

def check_data_spread(df: DataFrame):
    """
    Checks data spread in a DataFrame.
    
    Parameters:
    - df: pd.DataFrame, the DataFrame to analyze.
    
    Returns:
    - Dictionary containing data spread information for each column.
    """
    spread_results = {}

    for column in df.columns:
        # Count the occurrences of each unique value in the column
        value_counts = df[column].value_counts()

        # Convert the value counts to a dictionary
        value_counts_dict = value_counts.to_dict()

        # Check if data spread is uniform
        is_uniform_spread = all(count == value_counts.iloc[0] for count in value_counts)

        # Calculate most occurring value
        most_occuring_value = value_counts.idxmax() if not is_uniform_spread else ''

        # Calculate least occurring value
        least_occuring_value = value_counts.idxmin() if not is_uniform_spread else ''

        is_uniform_spread_label = 'Yes' if is_uniform_spread else 'No'
        spread_percentage = df[column].nunique() / df.shape[0] * 100

        # Store the spread pattern and analytics in the dictionary
        spread_results[column] = {
            'spread_pattern': value_counts_dict,
            'spread_percentage': spread_percentage,
            'is_uniform_spread': is_uniform_spread_label,
            'most_occuring_value': most_occuring_value,
            'least_occuring_value': least_occuring_value
        }

    return spread_results

def check_data_uniqueness(df: DataFrame):
    """
    Checks data uniqueness in a DataFrame.
    
    Parameters:
    - df: pd.DataFrame, the DataFrame to analyze.
    
    Returns:
    - Dictionary containing uniqueness information for each column.
    """
    total_rows = df.shape[0]
    uniqueness_results = {}

    for column in df.columns:
        unique_values_count = df[column].nunique()
        uniqueness_percentage = (unique_values_count / total_rows) * 100
        total_unique_values = unique_values_count
        is_unique = 'Yes' if uniqueness_percentage == 100 else ''

        uniqueness_results[column] = {
            'uniqueness_percentage': uniqueness_percentage,
            'total_unique_values': total_unique_values,
            'is_unique': is_unique
        }

    return uniqueness_results

def check_categorical_data(df: DataFrame):
    """
    Checks if columns in a DataFrame contain categorical data.
    
    Parameters:
    - df: pd.DataFrame, the DataFrame to analyze.
    
    Returns:
    - Dictionary containing categorical data information for each column.
    """
    data_spread_results = check_data_spread(df)
    infer_data_type = infer_data_types(df)
    threshold_percentage = 10.00
    categorical_results = {}

    for column in df.columns:
        is_categorical_data = ''
        data_categories = ''
        if infer_data_type[column]['inferred_data_type'] == 'str':
            if data_spread_results[column]['spread_percentage'] < threshold_percentage:
                is_categorical_data = 'Yes'
                data_categories = list(data_spread_results[column]['spread_pattern'].keys())

        categorical_results[column] = {
            "is_categorical": is_categorical_data,
            "categories": data_categories
        }

    return categorical_results

def calculate_correlation_matrix(df: DataFrame):
    """
    Calculate the correlation matrix for a given DataFrame.

    Parameters:
    - df: pd.DataFrame, the DataFrame to calculate the correlation matrix for.

    Returns:
    - correlation_matrix: pd.DataFrame, the calculated correlation matrix.
    """
    # Select only numeric columns for correlation calculation
    numeric_columns = df.select_dtypes(include=['number'])
    
    if numeric_columns.empty:
        print("No numeric columns found for correlation calculation.")
        return None

    # Using the corr method to calculate the correlation matrix
    correlation_matrix = numeric_columns.corr()

    return correlation_matrix

def prepare_data_report(df: DataFrame):
    """
    Prepares a report on a DataFrame including data type inference, completeness, spread, uniqueness, and categorical data.

    Parameters:
    - df: pd.DataFrame, the DataFrame to prepare the report for.

    Returns:
    - report_df: pd.DataFrame, the prepared data report.
    """
    completeness_results = check_data_completeness(df)
    uniqueness_results = check_data_uniqueness(df)
    data_spread_results = check_data_spread(df)
    infer_data_type = infer_data_types(df)
    categorical_results = check_categorical_data(df)

    report_data = {'Column Name': [],
                   'Orig Data Type': [],
                   'Infered Data Type': [],
                   'Is Categorical': [],
                   'Is Unique': [],
                   'Unique Values': [],
                   'Categories': [],
                   'Present Values': [],
                   'Missing Values': [],
                   'Completeness Percentage': [],
                   'Empty Percentage': [],
                   'Is uniform spread': [],
                   'Spread %age': [],
                   'Most occuring value': [],
                   'Least occuring value': []}

    for col in df.columns:
        report_data['Column Name'].append(col)
        report_data['Orig Data Type'].append(infer_data_type[col]['original_data_type'])
        report_data['Infered Data Type'].append(infer_data_type[col]['inferred_data_type'])
        report_data['Is Categorical'].append(categorical_results[col]['is_categorical'])
        report_data['Is Unique'].append(uniqueness_results[col]['is_unique'])
        report_data['Unique Values'].append(uniqueness_results[col]['total_unique_values'])
        report_data['Categories'].append(categorical_results[col]['categories'])
        report_data['Present Values'].append(completeness_results[col]['present_values'])
        report_data['Missing Values'].append(completeness_results[col]['missing_values'])
        report_data['Completeness Percentage'].append(completeness_results[col]['completeness_percentage'])
        report_data['Empty Percentage'].append(completeness_results[col]['empty_percentage'])
        report_data['Is uniform spread'].append(data_spread_results[col]['is_uniform_spread'])
        report_data['Spread %age'].append(round(data_spread_results[col]['spread_percentage'], 2))
        report_data['Most occuring value'].append(data_spread_results[col]['most_occuring_value'])
        report_data['Least occuring value'].append(data_spread_results[col]['least_occuring_value'])

    report_df = DataFrame(report_data)
    return report_df

df = read_data("./data/Retail_Transactions_Dataset.csv")

results = prepare_data_report(df)

correlation_matrix = calculate_correlation_matrix(df)

print(results)
