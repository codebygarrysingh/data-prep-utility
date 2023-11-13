import numpy as np
import pandas as pd
from pandas import DataFrame
import re
import os
from ydata_profiling import ProfileReport

def read_data(input_file: str) -> pd.DataFrame:
    try:
        if isinstance(input_file, str):
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"File not found: {input_file}")

            if input_file.endswith('.csv'):
                # Read data as string columns without inferring schema
                df = pd.read_csv(input_file, dtype=str)
            elif input_file.endswith(('.xls', '.xlsx')):
                # Read data as string columns without inferring schema
                df = pd.read_excel(input_file, dtype=str)
            elif input_file.endswith('.json'):
                df = pd.read_json(input_file, dtype=str)
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

def clean_input_data(df: DataFrame) -> DataFrame:
    try:
        # Trim string columns before further cleaning operations
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Trim string columns before further cleaning operations
        df = df.map(lambda x: x if not isinstance(x, str) else re.sub(r'(?i)null', 'None', x))

        # Replace empty strings with None
        df.replace('', None, inplace=True)

        # Replace NaN values with empty
        df.fillna(value='', inplace=True)

        df.dropna()

        return df
    except Exception as e:
        print(f"An error occurred while cleaning the input data: {str(e)}")
        return None

def infer_schema(df: DataFrame) -> dict:
    try:
        # Initialize an empty dictionary to store inferred schemas
        schema = {}

        for column in df.columns:
            # Exclude columns with only None values
            if df[column].notna().any():
                # Exclude rows with None values in the column
                column_data = df.dropna(subset=[column])
                schema[column] = pd.api.types.infer_dtype(column, skipna=True)
            else:
                # Default columns with all None values to string
                schema[column] = str

        return schema
    except Exception as e:
        print(f"An error occurred while inferring the schema: {str(e)}")
        return None


if __name__ == "__main__":
    df = read_data("C:\\Users\\CVW4MYCA3\\Downloads\\Fido-Oct27_2023-866811631-Usage.xls")

    profile = ProfileReport(df, title="Profiling Report")

    profile.to_file("profiling_report.html")