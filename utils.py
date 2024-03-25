import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score

def one_hot_encode(df, exclude_columns=None):
    """
    Perform one-hot encoding on object and category columns of a DataFrame, with the option to exclude specified columns. 
    For columns with two categories, encode them as 0 and 1. For columns with more than two categories, 
    perform one-hot encoding and drop the 'None' category if present, otherwise use drop_first, 
    ensuring all results are integers.
    
    Parameters:
    - df: DataFrame to encode.
    - exclude_columns: List of column names to exclude from one-hot encoding.
    
    Returns:
    - DataFrame with one-hot encoded columns.
    """
    if exclude_columns is None:
        exclude_columns = []

    # Select columns that are of type object or category, excluding specified columns
    object_or_category_columns = df.select_dtypes(include=['object', 'category']).columns
    columns_to_encode = [col for col in object_or_category_columns if col not in exclude_columns]

    for column in columns_to_encode:
        unique_values = df[column].nunique()
        
        if unique_values == 2:
            # Convert two categories to 0 and 1 explicitly and ensure integer type
            df[column] = pd.Categorical(df[column]).codes
            df[column] = df[column].astype(int)
            
        elif unique_values > 2:
            # Check if "None" is one of the categories
            if 'None' in df[column].unique():
                # Perform one-hot encoding and drop the "None" category
                dummies = pd.get_dummies(df[column], prefix=column).astype(int)
                dummies.drop(column + '_None', axis=1, inplace=True, errors='ignore')  # Use errors='ignore' in case "None" category doesn't exist
                df = pd.concat([df.drop(column, axis=1), dummies], axis=1)
            else:
                # Perform one-hot encoding with drop_first=True and ensure integer type
                dummies = pd.get_dummies(df[column], prefix=column, drop_first=True).astype(int)
                df = pd.concat([df.drop(column, axis=1), dummies], axis=1)

    return df


def find_uniq_vals(df):
    """
    Print unique values for each column of object or category data type in a DataFrame.
    
    Parameters:
    - df: pandas DataFrame.
    """
    # Select columns of object or category datatype
    object_or_category_columns = df.select_dtypes(include=['object', 'category']).columns

    # Print unique values for each object or category column
    for column in object_or_category_columns:
        print(f"Unique values in {column}:")
        print(df[column].unique())
        print("\n")

def analyze_missing_vals(df):
    """
    Calculate the percentage of missing values for each column and the actual number of missing values.
    
    Parameters:
    - df: pandas DataFrame.
    
    Returns:
    - DataFrame with columns for the percentage of missing values and the actual number of missing values,
      for columns that have missing values, sorted by the percentage of missing values in descending order.
    """
    # Calculate the percentage of missing values for each column
    missing_percentage = df.isnull().mean() * 100
    # Calculate the actual number of missing values for each column
    missing_count = df.isnull().sum()
    
    # Combine the two series into a DataFrame
    missing_df = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
    
    # Filter out columns that have missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    # Sort by the percentage of missing values in descending order
    missing_df = missing_df.sort_values(by='Missing Percentage', ascending=False)
    
    return missing_df

# Univariate Analysis Functions
def plot_distribution(data, column, title, xlabel, ylabel='Frequency'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_boxplot(data, column, title, xlabel, position, nrows, ncols):
    """
    Plots a boxplot for a given column in a specified subplot position.
    
    Parameters:
    - data: pandas DataFrame containing the data.
    - column: The name of the column to plot.
    - title: The title of the plot.
    - xlabel: The label for the x-axis.
    - position: Position of the subplot in the grid.
    - nrows: Number of rows in the subplot grid.
    - ncols: Number of columns in the subplot grid.
    """
    plt.subplot(nrows, ncols, position)
    sns.boxplot(x=data[column])
    plt.title(title)
    plt.xlabel(xlabel)


def plot_countplot(data, column, title, xlabel, ylabel='Count', rotation=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotation:
        plt.xticks(rotation=rotation)
    plt.show()

# Bivariate Analysis Functions
def plot_scatter(data, x_column, y_column, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_column, y=y_column, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_correlation_matrix(data, title):
    corr_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
    plt.figure(figsize=(25, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdYlGn')
    plt.title(title)
    plt.show()

def plot_boxplot_categorical(data, x_column, y_column, title, xlabel, ylabel, rotation=None):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=x_column, y=y_column, data=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotation:
        plt.xticks(rotation=rotation)
    plt.show()

# Additional Analysis Function
def plot_pairplot(data, title="Pair Plot"):
    sns.pairplot(data)
    plt.title(title)
    plt.show()

def eda_categorical(data):
    """
    Perform exploratory data analysis on categorical columns (both object and category dtypes)
    of a DataFrame by plotting the frequency of values in each categorical column on a single figure.
    
    Parameters:
    - data: DataFrame to analyze.
    """
    # Select columns of both object and category datatype
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    n = len(categorical_columns)
    
    # Calculate the total number of rows needed for the subplots, aiming for 3 plots per row
    total_rows = (n + 2) // 3
    
    # Create a single figure and axes grid for all plots
    fig, axes = plt.subplots(nrows=total_rows, ncols=3, figsize=(20, 6 * total_rows))
    
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    
    for i, column in enumerate(categorical_columns):
        sns.countplot(x=column, data=data, ax=axes[i])
        axes[i].set_title(f'Frequency of {column}')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)
    
    # Hide any unused axes if the number of plots isn't a perfect multiple of 3
    for j in range(i + 1, total_rows * 3):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def eda_numeric(data):
    """
    Perform exploratory data analysis on numeric columns of a DataFrame.
    
    Parameters:
    - data: DataFrame to analyze.
    """    
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    n = len(numeric_columns)
    
    # Define the number of rows needed in the subplot grid
    total_rows = (n + 2) // 3  # Round up to ensure all variables are included
    
    fig, axes = plt.subplots(total_rows, 6, figsize=(20, 5 * total_rows))  # 6 columns per row
    
    for i, column in enumerate(numeric_columns):
        row = i // 3
        hist_col = (i % 3) * 2  # Determine the column for the histogram (0, 2, 4)
        box_col = hist_col + 1  # Box plot immediately follows the histogram
        
        # Plot histogram
        sns.histplot(data[column], kde=True, ax=axes[row, hist_col])
        axes[row, hist_col].set_title(f'Histogram of {column}')
        
        # Plot box plot
        sns.boxplot(x=data[column], ax=axes[row, box_col])
        axes[row, box_col].set_title(f'Box Plot of {column}')
    
    # Hide any remaining, unused subplots
    for j in range(i + 1, total_rows * 3):
        unused_row = j // 3
        unused_hist_col = (j % 3) * 2
        axes[unused_row, unused_hist_col].set_visible(False)
        axes[unused_row, unused_hist_col + 1].set_visible(False)

    plt.tight_layout()
    plt.show()

def bivariate_eda_categorical(data, target_column):
    """
    Perform bivariate exploratory data analysis on categorical columns of a DataFrame.
    
    Parameters:
    - data: DataFrame to analyze.
    - target_column: The name of the target column for the y-axis in the boxplots.
    """
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    n = len(categorical_columns)
    
    if n == 0:
        print("No categorical columns to plot.")
        return
    
    # Determine the total number of rows needed, ensuring at least 1 row
    total_rows = max((n + 2) // 3, 1)
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(nrows=total_rows, ncols=3, figsize=(20, 6 * total_rows))
    
    # Flatten the axes array for easier 1D indexing if more than one row is created
    if total_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Ensure axes is always a list for consistency

    for i, column in enumerate(categorical_columns):
        sns.boxplot(x=column, y=target_column, data=data, ax=axes[i])
        axes[i].set_title(f'Distribution of {target_column} by {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel(target_column)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Hide any remaining, unused subplots
    for j in range(i + 1, total_rows * 3):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def bivariate_eda_numeric(data, target_column):
    """
    Perform bivariate exploratory data analysis on categorical columns of a DataFrame.
    
    Parameters:
    - data: DataFrame to analyze.
    - target_column: The name of the target column for the y-axis in the boxplots.
    """    
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = numeric_columns.drop(target_column)  # Exclude the target variable itself
    n = len(numeric_columns)
    
    # Determine the total number of rows needed
    total_rows = max((n + 2) // 3, 1)  # Ensure there's at least one row
    
    # Create a single figure with a grid of subplots
    fig, axes = plt.subplots(nrows=total_rows, ncols=3, figsize=(20, 6 * total_rows))
    
    # Flatten the axes array for easier 1D indexing if more than one row is created
    if total_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Ensure axes is always a list for consistency

    for i, column in enumerate(numeric_columns):
        sns.scatterplot(x=column, y=target_column, data=data, ax=axes[i])
        axes[i].set_title(f'{target_column} vs. {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel(target_column)
    
    # Hide any remaining, unused subplots
    for j in range(i + 1, total_rows * 3):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_performance_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test):
    """
    Calculate and return performance metrics for training and testing sets.
    
    Parameters:
    - y_true_train: Actual values for the training set.
    - y_pred_train: Predicted values for the training set.
    - y_true_test: Actual values for the test set.
    - y_pred_test: Predicted values for the test set.
    - exp_transform: Boolean indicating whether to apply exponential transformation (default: False).
    
    Returns:
    - DataFrame with performance metrics for both training and testing sets.
    """

    # Reverse log transformation
    y_true_train = np.exp(y_true_train)
    y_pred_train = np.exp(y_pred_train)
    y_true_test = np.exp(y_true_test)
    y_pred_test = np.exp(y_pred_test)
        
    # Calculating metrics
    train_mse = mean_squared_error(y_true_train, y_pred_train)
    test_mse = mean_squared_error(y_true_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_true_train, y_pred_train)
    test_r2 = r2_score(y_true_test, y_pred_test)
    
    # Creating a DataFrame to display metrics
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'R-squared'],
        'Train': [train_mse, train_rmse, train_r2],
        'Test': [test_mse, test_rmse, test_r2]
    })
    
    return metrics_df

def calculate_performance_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test):
    """
    Calculate and return performance metrics for training and testing sets, including RSS, formatted to two decimal places.
    
    Parameters:
    - y_true_train: Actual values for the training set.
    - y_pred_train: Predicted values for the training set.
    - y_true_test: Actual values for the test set.
    - y_pred_test: Predicted values for the test set.
    - exp_transform: Boolean indicating whether to apply exponential transformation (default: False).
    
    Returns:
    - DataFrame with performance metrics for both training and testing sets, formatted to two decimal places.
    """
    # Reverse log transformation
    y_true_train = np.exp(y_true_train)
    y_pred_train = np.exp(y_pred_train)
    y_true_test = np.exp(y_true_test)
    y_pred_test = np.exp(y_pred_test)
        
    # Calculating metrics
    train_mse = mean_squared_error(y_true_train, y_pred_train)
    test_mse = mean_squared_error(y_true_test, y_pred_test)
    train_rss = np.sum((y_true_train - y_pred_train) ** 2)
    test_rss = np.sum((y_true_test - y_pred_test) ** 2)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_true_train, y_pred_train)
    test_r2 = r2_score(y_true_test, y_pred_test)
    
    # Formatting metrics to two decimal places
    metrics_formatted = {
        'Metric': ['R-squared', 'RMSE', 'MSE', 'RSS'],
        'Train': [f"{train_r2:.2f}", f"{train_rmse:.2f}", f"{train_mse:.2f}", f"{train_rss:.2f}"],
        'Test': [f"{test_r2:.2f}", f"{test_rmse:.2f}", f"{test_mse:.2f}", f"{test_rss:.2f}"]
    }
    
    # Creating a DataFrame to display metrics
    metrics_df = pd.DataFrame(metrics_formatted)
    
    return metrics_df

def get_coefficients_table(model, feature_names):
    """
    Generate a table of feature names and their corresponding coefficients from a Ridge Regression model,
    formatted to normal notation up to five decimal places.
    
    Parameters:
    - model: The trained Ridge Regression model.
    - feature_names: A list of feature names.
    
    Returns:
    - A pandas DataFrame with two columns: 'Feature' and 'Coefficient', sorted by the absolute values of the coefficients
      in descending order, with a correctly reset index, and coefficients in normal notation rounded to five decimal places.
    """
    # Extract the coefficients from the model
    coefs = model.coef_
    
    # Pair each feature name with its coefficient
    coef_pairs = zip(feature_names, coefs)
    
    # Create a DataFrame from the pairs
    coef_df = pd.DataFrame(coef_pairs, columns=['Feature', 'Coefficient'])
    
    # Round the coefficients to five decimal places
    coef_df['Coefficient'] = coef_df['Coefficient'].round(5)
    
    # Sort the DataFrame by the absolute values of the coefficients in descending order
    coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)
    
    # Reset the DataFrame index to reflect the new order and drop the old index
    coef_df = coef_df.reset_index(drop=True)
    
    return coef_df

def plot_prediction_fit(y_true, y_pred, title='Model Fit'):
    """
    Plot the actual vs. predicted values and a line representing perfect predictions.
    
    Parameters:
    - y_true: The actual values.
    - y_pred: The model's predicted values.
    - title: Title for the plot (default: 'Model Fit').
    """
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of actual vs. predicted values
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Predictions')
    
    # Perfect predictions line
    perfect_predictions = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(perfect_predictions, perfect_predictions, color='red', label='Perfect Fit')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.show()

def plot_error_terms(res, bins=20, title='Error Terms', xlabel='Errors'):
    """
    Plot the distribution of error terms (residuals) with a histogram and density curve.
    
    Parameters:
    - bins: Number of bins for the histogram (default: 20).
    - title: Title for the plot (default: 'Error Terms').
    - xlabel: Label for the X-axis (default: 'Errors').
    """
    
    # Initialize plot
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(res, bins=bins, kde=True)  # kde=True adds the density curve
    
    # Plot formatting
    fig.suptitle(title, fontsize=20)  # Plot heading
    plt.axhline(y=0, color='r', linestyle=':')
    plt.xlabel(xlabel, fontsize=18)  # X-label
    plt.ylabel('Residuals', fontsize=18)  # Y-label
    plt.grid(True)
    
    plt.show()

def plot_qqplot(res, title='QQ Plot of Residuals'):
    """
    Plot a QQ plot of the residuals to assess normality.
    
    Parameters:
    - residuals: The residuals (error terms) from the model.
    - title: Title for the plot (default: 'QQ Plot of Residuals').
    """

    fig = sm.qqplot(res, fit=True, line='45')
    plt.title(title, fontsize=20)
    plt.xlabel('Theoretical Quantiles', fontsize=18)
    plt.ylabel('Sample Quantiles', fontsize=18)
    plt.grid(True)
    plt.show()

def plot_r2_vs_alpha(cv_scores):
    ## Plotting R2 score vs alpha values
    plt.plot(cv_scores['param_alpha'], cv_scores['mean_train_score'], label='Train')
    plt.plot(cv_scores['param_alpha'], cv_scores['mean_test_score'], label='Test')
    plt.xlabel('alpha')
    plt.ylabel('neg_mean_absolute_error')
    plt.xscale('log')
    plt.legend()
    plt.show()