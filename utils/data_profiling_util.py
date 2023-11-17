from pandas import DataFrame
from data_read_util import infer_data_types

def check_data_completeness(df: DataFrame, missing_threshold=0.0):
    """
    Check data completeness in a Pandas DataFrame.

    Args:
        df (DataFrame): The input DataFrame.
        missing_threshold (float, optional): Threshold for missing data percentage.
            Columns with missing data percentage above this threshold will be flagged.

    Returns:
        dict: A dictionary mapping column names to completeness information, including:
            - 'present_values': Number of non-empty (present) values.
            - 'missing_values': Number of missing values.
            - 'completeness_percentage': Percentage of present values.
            - 'empty_percentage': Percentage of missing values.
            - 'flagged': True if missing data exceeds the specified threshold, False otherwise.
    """
    total_rows = df.shape[0]
    completeness_results = {}

    for column in df.columns:
        non_empty_count = df[column].count()
        
        present_values = non_empty_count
        completeness_percentage = (non_empty_count / total_rows) * 100
        empty_percentage = 100 - completeness_percentage
        missing_values = total_rows - non_empty_count

        flagged = False
        if missing_threshold > 0.0 and empty_percentage > missing_threshold:
            flagged = True

        completeness_results[column] = {
            'present_values': present_values,
            'missing_values': missing_values,
            'completeness_percentage': completeness_percentage,
            'empty_percentage': empty_percentage,
            'flagged': flagged
        }
    
    return completeness_results


def check_data_spread(df: DataFrame):
    """
    Analyze the spread and distribution of values in each column of a Pandas DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary mapping column names to spread and distribution information, including:
            - 'spread_pattern': Dictionary of value counts for each unique value in the column.
            - 'spread_percentage': Percentage of unique values in the column.
            - 'is_uniform_spread': 'Yes' if the spread is uniform (all values are the same), 'No' otherwise.
            - 'most_occuring_value': The most occurring value in the column.
            - 'least_occuring_value': The least occurring value in the column.
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
        spread_percentage = (df[column].nunique() / df.shape[0]) * 100

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
    Analyze the uniqueness of values in each column of a Pandas DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary mapping column names to uniqueness information, including:
            - 'uniqueness_percentage': Percentage of unique values in the column.
            - 'total_unique_values': The total count of unique values.
            - 'is_unique': 'Yes' if all values in the column are unique, '' (empty string) otherwise.
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
    Analyze columns in a Pandas DataFrame to identify categorical data.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary mapping column names to categorical data information, including:
            - 'is_categorical': 'Yes' if the column is categorical, '' (empty string) otherwise.
            - 'categories': List of categories if the column is categorical, '' (empty list) otherwise.
    """
    data_spread_results = check_data_spread(df)
    infer_data_type = infer_data_types(df)
    threshold_percentage = 10.0  # You might adjust this threshold as needed

    categorical_results = {}

    for column in df.columns:
        is_categorical_data = ''
        data_categories = []

        if infer_data_type[column]['infered_data_type'] == 'str':
            if data_spread_results[column]['spread_percentage'] < threshold_percentage:
                is_categorical_data = 'Yes'
                data_categories = list(data_spread_results[column]['spread_pattern'].keys())

        if not is_categorical_data:
            # Label non-categorical columns
            is_categorical_data = 'No'

        categorical_results[column] = {
            "is_categorical": is_categorical_data,
            "categories": data_categories
        }

    return categorical_results


def calculate_correlation_matrix(df: DataFrame):
    """
    Calculate the correlation matrix for a given DataFrame.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        DataFrame: A Pandas DataFrame representing the correlation matrix.
    """
    # Using the corr method to calculate the correlation matrix
    correlation_matrix = df.corr()

    return correlation_matrix

