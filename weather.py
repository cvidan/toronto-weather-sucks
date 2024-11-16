import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


DATA_DIR = './data/'
IMAGES_DIR = './images/'


def csv_to_dataframe(file: str):
    df = pd.read_csv(DATA_DIR + file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    return df


def count_value_in_column(df: DataFrame, column_name: str, value):
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")

    return df[column_name].value_counts().get(value, 0)


def percentage_value_in_column(df: DataFrame, column_name: str, value):
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")
    if len(df) == 0:
        raise ZeroDivisionError("Cannot calculate percentage on empty DataFrame")

    count = count_value_in_column(df, column_name, value)
    percentage = (count / len(df)) * 100
    
    return round(percentage, 1)


def diff_dataframes_by_column(df1: DataFrame, df2: DataFrame, column_name: str):
    aligned_df = df1[column_name].align(df2[column_name])
    return aligned_df[0] - aligned_df[1]
    
    
def plot_histogram(df: DataFrame, output_file: str, title: str, x_label: str, bin_size: int):
    plt.figure(figsize=(12, 6))

    raw_min = np.floor(df.min())
    min_val = raw_min - (raw_min % bin_size)
    raw_max = np.ceil(df.max())
    max_val = raw_max + (bin_size - (raw_max % bin_size)) if raw_max % bin_size else raw_max
    bins = np.arange(min_val, max_val + bin_size, bin_size)
    
    plt.hist(df, bins=bins, color='skyblue', edgecolor='black')
    
    plt.axvline(x=0, color='black', linestyle='solid')
    mean_diff = df.mean()
    plt.axvline(x=mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.1f}')
    
    plt.title(title, pad=20)
    plt.xlabel(x_label)
    plt.ylabel('Frequency (days)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    stats_text = f'Mean: {df.mean():.1f}\n'
    stats_text += f'SD: {df.std():.1f}\n'
    stats_text += f'p10: {np.percentile(df, 10):.1f}\n'
    stats_text += f'p90: {np.percentile(df, 90):.1f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = IMAGES_DIR + output_file
    plt.savefig(output_path)
    plt.close()
    print(f"Histogram saved to '{output_path}'")

    
if __name__ == "__main__":
    # From https://www.visualcrossing.com/
    manhattan_csv = "manhattan 2019-10-01 to 2024-10-01.csv"
    toronto_csv = "toronto 2019-10-01 to 2024-10-01.csv"
    
    try:
        manhattan_df = csv_to_dataframe(manhattan_csv)
        toronto_df = csv_to_dataframe(toronto_csv)
        
        manhattan_num_clear_days = count_value_in_column(manhattan_df, 'conditions', 'Clear')
        manhattan_pct_clear_days = percentage_value_in_column(manhattan_df, 'conditions', 'Clear')
        toronto_num_clear_days = count_value_in_column(toronto_df, 'conditions', 'Clear')
        toronto_pct_clear_days = percentage_value_in_column(toronto_df, 'conditions', 'Clear')
        headers = ['City', '% clear days', '# clear days']
        data = [
            ['Manhattan', manhattan_pct_clear_days, manhattan_num_clear_days],
            ['Toronto', toronto_pct_clear_days, toronto_num_clear_days]
        ]
        print(tabulate(data, headers=headers, tablefmt='grid'))
        
        feelslike_diff_df = diff_dataframes_by_column(manhattan_df, toronto_df, 'feelslike')
        plot_histogram(
            feelslike_diff_df,
            'diff_feelslike.png', 
            'Difference between daily Manhattan and Toronto "feels like" temperature (°C)',
            'Temperature (°C)',
            1
        )
        
        windspeed_diff_df = diff_dataframes_by_column(manhattan_df, toronto_df, 'windspeed')
        plot_histogram(
            windspeed_diff_df,
            'diff_windspeed.png',
            'Difference between daily Manhattan and Toronto wind speed (km/h)',
            'Wind speed (km/h)',
            5
        )
        
    except FileNotFoundError:
        print("Error: One or both CSV files not found. Please check the file paths.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
