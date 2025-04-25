#test the code to function for parsing the text strings
# coding: utf-8
"""
This script performs the following tasks:
1. Sets up a working directory named 'my_python_project'.
2. Checks for the existence of a CSV file named 'data.csv' in the working directory.
3. Reads the CSV file and displays its contents if it exists.
4. Defines a `cleansing` function to:
    - Read specific columns from the CSV file.
    - Display the first few rows of the selected columns.
    - Provide information about the DataFrame.
5. Adds new columns to the cleansed DataFrame:
    - A column based on the sum of 'column1' and 'column3'.
    - A column containing the first 7 characters of 'column2'.
    - A column containing the last 5 characters of 'column2'.
    - A column extracting text between "EX=" and "," in 'column2' using regex.
    - Two columns extracting specific substrings ('xxxx' and 'yy') from 'column2' using regex.
Regex Explanation:
- `str.extract(r'EX=([^,]*)')`: This regex pattern is used to extract text between "EX=" and the next comma (",") in a string.
  - `EX=`: Matches the literal string "EX=".
  - `([^,]*)`: Captures any sequence of characters that are not a comma (","). The parentheses `()` create a capturing group, and the `[^,]` is a negated character class that matches any character except a comma. The `*` quantifier allows for zero or more occurrences of such characters.
  - The `str.extract()` method returns the captured group as a new column in the DataFrame.
Note:
- Replace "column1", "column2", and "column3" with actual column names in the CSV file.
- Ensure the 'data.csv' file exists in the working directory with the appropriate structure before running the script.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from email.message import EmailMessage
import smtplib


# Set up a working directory
working_directory = "data/"

# Create the directory if it doesn't exist
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

print(f"Working directory '{working_directory}' is set up.")

# Path to the data.csv file
data_file_path = os.path.join(working_directory, "thick.csv")

# Check if the file exists before reading
if os.path.exists(data_file_path):
    # Read the CSV file
    data = pd.read_csv(data_file_path)
    print("Data read successfully:")
    print(data.head())  # Display the first few rows of the data
else:
    print(f"File '{data_file_path}' does not exist.")

#####good to read file######

    def cleansing(columns_to_read):
        # Check if the file exists before reading
        if os.path.exists(data_file_path):
            # Read only the specified columns from the CSV file
            try:
                data = pd.read_csv(data_file_path, usecols=columns_to_read)
                print("Selected columns read successfully:")
                print(data.head())  # Display the first few rows of the selected columns
                return data
            except ValueError as e:
                print(f"Error reading columns: {e}")
        else:
            print(f"File '{data_file_path}' does not exist.")
            return None
        

        # Display information about the DataFrame
        if data is not None:
            print("DataFrame information:")
            print(data.info())


# Call the cleansing function to read and display the data
columns_to_read = ["column1", "column2", "column3"] # Replace with actual column names. define as global variable
cleansed_data = cleansing(columns_to_read)

if cleansed_data is not None:
    # Add a new column based on the sum of column1 and column3
    cleansed_data['new_column'] = cleansed_data['column1'] + cleansed_data['column3']
    print("New column added successfully:")
    print(cleansed_data.head())


    # Add a new column based on the first 7 letters of column2
    cleansed_data['short_column2'] = cleansed_data['column2'].str[:7]
    print("New column based on the first 7 letters of column2 added successfully:")
    print(cleansed_data.head())

    # Add a new column based on the last 5 letters of column2
    cleansed_data['last_5_column2'] = cleansed_data['column2'].str[-5:]
    print("New column based on the last 5 letters of column2 added successfully:")
    print(cleansed_data.head())


    # Add a new column based on regex pattern to extract text between "EX=" and ","
    cleansed_data['extracted_text'] = cleansed_data['column2'].str.extract(r'EX=([^,]*)')
    print("New column based on regex pattern added successfully:")
    print(cleansed_data.head())

    # Add new columns to extract 'xxxx' and 'yy' from the string in column2
    cleansed_data['extracted_xxxx'] = cleansed_data['column2'].str.extract(r'EX=([^,]*)')
    cleansed_data['extracted_yy'] = cleansed_data['column2'].str.extract(r'FS=([^,]*)')
    print("New columns 'extracted_xxxx' and 'extracted_yy' added successfully:")
    print(cleansed_data.head())

    def plot_heatmap(cleansed_data, x_col, y_col, value_col):
        
    # Use the global constant columns_to_read to pass the column names dynamically
        if len(columns_to_read) >= 3:
            x_col, y_col, value_col = columns_to_read[:3]
            plot_heatmap(cleansed_data, x_col=x_col, y_col=y_col, value_col=value_col)
        else:
            print("Insufficient columns in columns_to_read to plot the heatmap.")
        """
        Plots a heatmap using the given DataFrame.

        Parameters:
        - data: DataFrame containing the data.
        - x_col: Column name for x-coordinates.
        - y_col: Column name for y-coordinates.
        - value_col: Column name for the values to plot.

        Returns:
        - None
        """
        if data is not None:
            try:
                # Pivot the DataFrame to create a grid for the heatmap
                heatmap_data = data.pivot(index=y_col, columns=x_col, values=value_col)
                
                # Plot the heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(heatmap_data, cmap="viridis", annot=False, cbar=True)
                plt.title("Heatmap")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.show()
            except KeyError as e:
                print(f"Error creating heatmap: {e}")
        else:
            print("Data is None. Cannot plot heatmap.")

    # Call the function to plot the heatmap
    plot_heatmap(cleansed_data, x_col="column1", y_col="column2", value_col="column3")


#standard deviation analysis

import pandas as pd
     import numpy as np
     import matplotlib.pyplot as plt

     # Load data (replace with your actual data loading)
     data = pd.read_csv('surface_data.csv')

     # Calculate standard deviation of surface heights
     standard_deviation = data['height'].std()

     print(f"Standard Deviation of surface flatness: {standard_deviation}")

     # Visualize (optional)
     plt.hist(data['height'], bins=20)
     plt.xlabel('Surface Height')
     plt.ylabel('Frequency')
     plt.title('Surface Height Distribution')
     plt.show()

#RMSE analysis

     import pandas as pd
     import numpy as np
     import matplotlib.pyplot as plt

     # Load data (replace with your actual data loading)
     data = pd.read_csv('surface_data.csv')

     # Assume 'height' column contains surface heights
     surface_heights = data['height']

     # Calculate the mean height (as an approximation of a plane)
     mean_height = np.mean(surface_heights)

     # Calculate the squared differences from the mean
     squared_differences = (surface_heights - mean_height)**2

     # Calculate the root mean squared error
     rmse = np.sqrt(np.mean(squared_differences))

     print(f"Root Mean Squared Error (RMSE) of surface flatness: {rmse}")

     # Visualize (optional)
     plt.scatter(range(len(surface_heights)), surface_heights, label="Surface Height")
     plt.plot(range(len(surface_heights)), [mean_height] * len(surface_heights), color="red", linestyle='-', label= "Mean Height (Plane)", linewidth=2)
     plt.xlabel("Data Points")
     plt.ylabel("Surface Height")
     plt.title("Surface Height vs. Mean Height")
     plt.legend()
     plt.show()


     import numpy as np

def calculate_std_dev(data, ddof=0):
  """Calculates the standard deviation of a dataset.

  Args:
    data: A list or numpy array of numerical data.
    ddof: Delta Degrees of Freedom. 0 for population std dev, 1 for sample std dev.

  Returns:
      The standard deviation of the data.
  """
  return np.std(data, ddof=ddof)

def calculate_rmse(predictions, targets):
    """Calculates the Root Mean Squared Error (RMSE).

    Args:
        predictions: A list or numpy array of predicted values.
        targets: A list or numpy array of actual values.

    Returns:
        The RMSE value.
    """
    return np.sqrt(np.mean((np.array(predictions) - np.array(targets))**2))
