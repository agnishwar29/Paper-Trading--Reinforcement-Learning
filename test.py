import os
import time

import pandas as pd
import matplotlib.pyplot as plt


# df = pd.read_csv(r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset\NIFTY50_all.csv")
# fig, ax = plt.subplots()
# ax.plot(df['Close'])
#
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.show()

dataset_dir = r"D:\PycharmProjects\Paper Trading -- Reinforcement Learning\Dataset"

# List all files in the dataset directory
dataset_files = os.listdir(dataset_dir)

# Determine the number of rows and columns for the subplots
num_datasets = len(dataset_files)
num_rows = int(num_datasets ** 0.5)  # Square root of num_datasets
num_cols = int((num_datasets + num_rows - 1) / num_rows)

# Create the subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Flatten the subplots array to handle both 1D and 2D subplot cases
if num_datasets == 1:
    axs = [axs]
else:
    axs = axs.flatten()

# Plot each dataset in a separate subplot
for i, file in enumerate(dataset_files):
    df = pd.read_csv(os.path.join(dataset_dir, file))
    ax = axs[i]

    ax.plot(df['Close'])
    ax.set_title(file)

    # Add labels and titles
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')

# Remove empty subplots if num_datasets is not a perfect square
if len(axs) > num_datasets:
    for j in range(num_datasets, len(axs)):
        fig.delaxes(axs[j])

plt.tight_layout()  # Adjust spacing between subplots
plt.show()