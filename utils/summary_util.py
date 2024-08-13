import re

import pandas as pd


def create_winning_table(df, with_std=False, decreasing_order=False):
    # Function to extract the mean value from a string in the format "mean(std)"
    def extract_mean(value):
        match = re.match(r'([\d\.]+)\(.*\)', value)
        return float(match.group(1)) if match else None

    mean_df = df.copy()
    # Extract the mean values for comparison
    if with_std:
        mean_df = mean_df.drop(columns=['Type', 'dataset_name']).applymap(extract_mean)
    else:
        mean_df = mean_df.drop(columns=['Type', 'dataset_name'])
    # print(mean_df)
    # Identify the winning method for each dataset
    if decreasing_order:
        max_means = mean_df.max(axis=1)
        winners = mean_df.eq(max_means, axis=0)

    else:
        min_means = mean_df.min(axis=1)
        winners = mean_df.eq(min_means, axis=0)
        # print(mean_df)
        # print(min_means)
    # print(winners)
    # Add the winners to the original DataFrame
    total_win_counts = winners.sum(axis=0).astype(int)
    type_win_counts = df[['Type']].join(winners).groupby('Type').sum().astype(int)

    # Count the number of wins for each method within each type
    # type_win_counts_df = df.groupby('Type')['Winner'].value_counts().unstack(fill_value=0)
    total_win_counts_df = total_win_counts.to_frame().T
    # print(type_win_counts)
    # Align the win counts with the original columns (excluding 'Type' and 'Winner')
    total_win_counts_df = total_win_counts_df.reindex(columns=df.drop(columns=['Type']).columns, fill_value=0)
    type_win_counts_df = type_win_counts.reindex(columns=df.drop(columns=['Type']).columns, fill_value=0)
    total_win_counts_df['Type'] = ['all']
    type_win_counts_df['Type'] = type_win_counts_df.index
    # print(type_win_counts_df)
    # Append the win counts to the original DataFrame
    df_with_wins = pd.concat([df, type_win_counts_df, total_win_counts_df])
    # df_with_wins = df_with_wins.reset_index()
    return df_with_wins