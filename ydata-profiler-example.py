import pandas as pd
from ydata_profiling import ProfileReport

# Load your dataset
data = pd.read_csv("./data/Retail_Transactions_Dataset.csv")

# Create a profile report
profile = ProfileReport(data, title="Profiling Report")


# Generate the report
profile.to_file("data_profile.html")