# Author: Garry Singh
# Date: 2023-11-16
# Description: This script defines functions to read, analyze, and prepare a report on a DataFrame.

from pandas import DataFrame
from data_read_util import read_data, infer_data_types
from data_profiling_util import check_data_completeness, check_data_spread, check_data_uniqueness, check_categorical_data, calculate_correlation_matrix

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