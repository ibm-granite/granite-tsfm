import argparse
import os
import re

import pandas as pd


parser = argparse.ArgumentParser(description="TTM pretrain arguments.")
# Adding a positional argument
parser.add_argument(
    "--result_dir",
    "-rd",
    type=str,
    required=True,
    help="Directory containing results after running benchmarking script.",
)
args = parser.parse_args()

# Path to the main folder containing subfolders
main_folder_path = args.result_dir

# List to collect dataframes
all_data = []

# Iterate through all items in the main folder
for folder_name in os.listdir(main_folder_path):
    # Check if the folder name matches the pattern fl-XX_
    print(folder_name)
    match = re.search(r"fl-(\d+)_", folder_name)
    c_match = re.search(r"cl-(\d+)_", folder_name)
    print(match, c_match)
    if match:
        # Extract XX from the folder name
        XX = int(match.group(1))

        cl = int(c_match.group(1))

        print("reading", XX, cl)
        folder_path = os.path.join(main_folder_path, folder_name)

        # Check if results_zero_few.csv exists in the folder
        csv_file_path = os.path.join(folder_path, "results_zero_few.csv")
        if os.path.exists(csv_file_path):
            # Load the CSV file
            df = pd.read_csv(csv_file_path)

            # Add a new column 'FL' with value XX
            df["FL"] = XX
            df["CL"] = cl

            # Append the dataframe to the list
            all_data.append(df)

# Concatenate all dataframes into one
final_df = pd.concat(all_data, ignore_index=True)

custom_order = ["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"]


final_df["dataset_sorted"] = pd.Categorical(final_df["dataset"], categories=custom_order, ordered=True)

final_df = final_df.sort_values(by=["CL", "dataset_sorted", "FL"], ascending=[True, True, True])


# Save to a new CSV file or process further
out_file_name = os.path.basename(os.path.normpath(main_folder_path))
final_df[["dataset", "CL", "FL", "zs_mse", "fs5_mse"]].to_csv(f"combined_{out_file_name}.csv", index=False)

final_df = final_df.drop(columns=["Unnamed: 0", "dataset_sorted", "FL"])
cols = final_df.columns
cols_index = [
    "dataset",
    "CL",
]
cols_others = [_ for _ in cols if _ not in cols_index]
cols_ord = cols_index + cols_others
final_df = final_df[cols_ord]
avg_df = final_df.groupby(["dataset", "CL"], as_index=False, sort=False).mean()
avg_df = avg_df.round(decimals=3)
avg_df["CL"] = avg_df["CL"].astype(int)
avg_df.to_csv(
    f"combined_avg_{out_file_name}.csv",
    index=False,
)


print(f"All CSV files have been combined and saved as combined_{out_file_name}.csv")
print(f"Average scores per-dataset are as combined_avg_{out_file_name}.csv")
