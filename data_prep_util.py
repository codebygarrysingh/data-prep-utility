# Author: Garry Singh
# Date: 2023-11-16
# Description: This script calls data prep functions including data reading, profiling and reporting

from pandas import DataFrame
from data_read_util import read_data
from data_profiling_util import calculate_correlation_matrix
from data_report_util import prepare_data_report

def main():
    df = read_data("./data/Retail_Transactions_Dataset.csv")

    results = prepare_data_report(df)

    correlation_matrix = calculate_correlation_matrix(df)

    print("Data Profiling Results:")
    print(results)

    print("\nCorrelation Matrix:")
    print(correlation_matrix)

if __name__ == "__main__":
    main()
