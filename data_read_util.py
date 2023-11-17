# Author: Garry Singh
# Date: 2023-11-17
# Description: This script defines functions to read data into a Pandas DataFrame.

from pandas import DataFrame
import pandas as pd
import os

def read_data(input_file: str) -> pd.DataFrame:
    try:
        # Check if the input_file is a valid string path
        if isinstance(input_file, str) and os.path.exists(input_file):
            if input_file.endswith('.csv'):
                # Read CSV file into a Pandas DataFrame
                df = pd.read_csv(input_file)
            elif input_file.endswith(('.xls', '.xlsx')):
                # Read Excel file into a Pandas DataFrame
                df = pd.read_excel(input_file)
            elif input_file.endswith('.json'):
                # Read JSON file into a Pandas DataFrame
                df = pd.read_json(input_file)
            else:
                raise ValueError("Unsupported file format.")
        elif isinstance(input_file, pd.DataFrame):
            # Input is already a DataFrame, so no need to read from a file
            df = input_file
        else:
            raise ValueError("Unsupported input source type")

        return df  # Return the DataFrame

    except FileNotFoundError as e:
        print(f"File not found: {input_file}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the input data: {str(e)}")
        return None

# Example usage:
# df = read_data("data.csv")
# df = read_data("data.xlsx")
# df = read_data("data.json")
# df = read_data(existing_dataframe)

def infer_data_types(df: DataFrame):
    """
    Infer actual data types for columns in a DataFrame, especially 'object' columns.

    Args:
        df (DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary mapping column names to a dictionary containing:
            - 'original_data_type': The original data type detected by Pandas.
            - 'inferred_data_type': The inferred actual data type after analysis.
    """
def infer_data_types(df: DataFrame):
    data_types = df.dtypes.to_dict()

    # Dictionary to store actual data types for columns with 'object' data type
    actual_types = {}

    infer_data_type = {}

    for column, dtype in data_types.items():
        if dtype == 'object':
            # Attempt to find the actual data type for 'object' columns
            try:
                # Try converting to numeric
                actual_data_type = pd.to_numeric(df[column]).dtype
                actual_types[column] = str(actual_data_type)
            except ValueError:
                try:
                    # Try converting to datetime
                    actual_data_type = pd.to_datetime(df[column]).dtype
                    actual_types[column] = str(actual_data_type)
                except ValueError:
                    try:
                        # Try converting to boolean
                        actual_data_type = pd.to_numeric(df[column].astype(str), errors='coerce').notna().all()
                        actual_types[column] = 'bool' if actual_data_type else 'str'
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

# Example usage:
# inferred_types = infer_data_types(df)
# for column, types in inferred_types.items():
#     print(f"Column: {column}")
#     print(f"Original Data Type: {types['original_data_type']}")
#     print(f"Inferred Data Type: {types['inferred_data_type']}")

