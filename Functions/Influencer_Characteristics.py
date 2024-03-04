# Description: This file contains the functions to calculate the influencer characteristics
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
import ast
import networkx as nx
import pandas as pd


def find_common_elements_and_plot(set_1, set_2, set_3, name_1, name_2, name_3, title):
    # Find common elements between sets
    common_in_1_and_2 = set_1.intersection(set_2)
    common_in_1_and_3 = set_1.intersection(set_3)
    common_in_2_and_3 = set_2.intersection(set_3)

    # Merge all common elements
    all_common_elements = common_in_1_and_2.union(common_in_1_and_3, common_in_2_and_3)

    # Create a dictionary to store the count of sets in which each element is present
    element_presence_count = {}

    # Check for each common element in the sets and update the count
    for element in all_common_elements:
        count = 0
        if element in set_1:
            count += 1
        if element in set_2:
            count += 1
        if element in set_3:
            count += 1
        element_presence_count[element] = count

    # Plot using a Venn Diagram
    venn3([set_1, set_2, set_3], (name_1, name_2, name_3))
    plt.title(title)
    plt.show()


def plot_instances_by_week(dataframe, date_column, title='Evolution of Instances by Week', x_label='Date',
                           y_label='Number of Instances', rotation=45):
    # Convert date column to datetime format
    dataframe[date_column] = pd.to_datetime(dataframe[date_column])

    # Group by the day and count instances for each week
    count_by_week = dataframe.resample('W-Mon', on=date_column).size().reset_index(name='count')

    # Plot the data
    plt.plot(count_by_week[date_column], count_by_week['count'], marker='', linestyle='-', linewidth=1)

    # Customize the plot
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=rotation)

    # Show the plot
    plt.show()


def plot_instances_together(df1, df2, df3, timecol1, timecol2, timecol3, label1, label2, label3, title):
    # Convert to datetime format for each DataFrame
    df1[timecol1] = pd.to_datetime(df1[timecol1])
    df2[timecol2] = pd.to_datetime(df2[timecol2])
    df3[timecol3] = pd.to_datetime(df3[timecol3])

    # Group by the week and count instances for each week for each DataFrame
    count_by_week1 = df1.resample('W-Mon', on=timecol1).size().reset_index(name='count1')
    count_by_week2 = df2.resample('W-Mon', on=timecol2).size().reset_index(name='count2')
    count_by_week3 = df3.resample('W-Mon', on=timecol3).size().reset_index(name='count3')

    # Plot each DataFrame separately
    plt.plot(count_by_week1[timecol1], count_by_week1['count1'], label=label1, linestyle='-', linewidth=1)
    plt.plot(count_by_week2[timecol2], count_by_week2['count2'], label=label2, linestyle='-', linewidth=1)
    plt.plot(count_by_week3[timecol3], count_by_week3['count3'], label=label3, linestyle='-', linewidth=1)

    # Customize the plot if needed
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def calculate_time_between_posts(dataframe, username_column, date_column):
    # Get unique usernames
    unique_usernames = dataframe[username_column].unique()

    # Calculate time difference for each influencer
    timediff_list = []
    for username in unique_usernames:
        user_data = dataframe[dataframe[username_column] == username]
        time_diff = user_data[date_column].sort_values().diff().mean()
        timediff_list.append({'Username': username, 'Time Difference': time_diff})

    # Create DataFrame from list of dictionaries
    timediff_df = pd.DataFrame(timediff_list)

    return timediff_df


def plot_time_between_posts(df1, df2, df3, label1, label2, label3, title,lower_limit=0, upper_limit=1000):
    # Combine the data into a list of arrays
    data_to_plot = [
        df1["Time Difference"].dt.total_seconds().dropna() / (60 * 60 * 24),
        df2["Time Difference"].dt.total_seconds().dropna() / (60 * 60 * 24),
        df3["Time Difference"].dt.total_seconds().dropna() / (60 * 60 * 24)
    ]

    # Create a boxplot for all the data
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=[label1, label2, label3])

    # Customize the plot
    plt.title(title)
    plt.xlabel('Data Source')
    plt.ylabel('Days')
    plt.ylim(lower_limit, upper_limit)
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


def draw_hashtag_network(dataframe, hashtags_column, title, sample_size=None, k_value=0.2, node_size=2,
                         edge_color='grey', alpha=0.7, edge_weight=0.1):
    dataframes = pd.DataFrame()
    # Convert the column containing lists of hashtags to lists
    dataframes[hashtags_column] = dataframe[hashtags_column].apply(
        lambda x: [tag.strip() for tag in x.split(",")] if x else [])

    # Filter out rows with non-empty lists of hashtags
    filtered_df = dataframes[dataframes[hashtags_column].apply(lambda x: bool(x))].reset_index(drop=True)
    if sample_size is None:
        sample_size = 1000 if len(filtered_df) > 1000 else len(filtered_df)
    sample_size = sample_size
    random_sample = filtered_df.sample(n=sample_size, random_state=42)

    G = nx.Graph()
    for words_list in random_sample[hashtags_column]:
        co_occurrences = Counter(
            [(word1, word2) for i, word1 in enumerate(words_list) for j, word2 in enumerate(words_list) if i < j])
        for edge, weight in co_occurrences.items():
            G.add_edge(edge[0], edge[1], weight=weight)

    # Calculate the spring layout
    pos = nx.spring_layout(G, k=k_value)

    # Draw the graph with specified layout, considering edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=False, node_size=node_size, edge_color=edge_color,
            width=[weight * edge_weight for weight in edge_weights], alpha=alpha)

    plt.title(title)
    plt.show()


def draw_tag_network(dataframe, tags_column, title, sample_size=None, k_value=0.5, node_size=2,
                     edge_color='grey', alpha=0.7, edge_weight=0.1):
    dataframes = pd.DataFrame()
    dataframes[tags_column] = dataframe[tags_column].apply(lambda x: [tag.strip() for tag in x.split(",")] if x else [])
    filtered_df = dataframes[dataframes[tags_column].apply(lambda x: bool(x))].reset_index(drop=True)
    if sample_size is None:
        sample_size = 1000 if len(filtered_df) > 1000 else len(filtered_df)
    sample_size = sample_size
    random_sample = filtered_df.sample(n=sample_size, random_state=42)

    G = nx.Graph()
    # Assuming tags_column is a DataFrame column containing lists of words in each row
    for i, row1 in random_sample.iterrows():
        for j, row2 in random_sample.iterrows():
            if i < j:
                # Check if there is any common word between the two rows
                common_words = set(row1[tags_column]) & set(row2[tags_column])
                if common_words:
                    co_occurrences = Counter(common_words)
                    for word, weight in co_occurrences.items():
                        G.add_edge(i, j, word=word, weight=weight)

    k_value = k_value  # You can adjust this value to control the repulsion
    pos = nx.spring_layout(G, k=k_value)

    # Draw the graph with specified layout
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=False, node_size=node_size, edge_color=edge_color,
            width=[weight * edge_weight for weight in edge_weights], alpha=alpha)

    plt.title(title)
    plt.show()


def calculate_post_count(df, inf_df, username_col):
    post_count_per_user = df.groupby(username_col).size()
    inf_df["post_count"] = inf_df["username"].map(post_count_per_user)


def calculate_avg_hashtag_per_post(df, inf_df, username_col, hashtag_col):
    avg_hashtag_per_post_per_user = df.groupby(username_col)[hashtag_col].apply(lambda x: x.apply(len).sum() / len(x))
    inf_df["avg_hashtag_per_post"] = inf_df["username"].map(avg_hashtag_per_post_per_user)


def calculate_sd_hashtag_per_post(df, inf_df, username_col, hashtag_col):
    sd_hashtag_per_post_per_user = df.groupby(username_col)[hashtag_col].apply(lambda x: x.apply(len).std())
    inf_df["avg_hashtag_per_post"] = inf_df["username"].map(sd_hashtag_per_post_per_user)


def calculate_avg_tag_per_post(df, inf_df, username_col, tag_col):
    avg_tag_per_post_per_user = df.groupby(username_col)[tag_col].apply(lambda x: x.apply(len).sum() / len(x))
    inf_df["avg_tag_per_post"] = inf_df["username"].map(avg_tag_per_post_per_user)


def calculate_sd_tag_per_post(df, inf_df, username_col, tag_col):
    sd_tag_per_post_per_user = df.groupby(username_col)[tag_col].apply(lambda x: x.apply(len).std())
    inf_df["sd_tag_per_post"] = inf_df["username"].map(sd_tag_per_post_per_user)


def calculate_avg_caption_length_per_user(df, inf_df, username_col, caption_col):
    avg_caption_length_per_user = df.groupby(username_col)[caption_col].apply(lambda x: x.str.len().mean())
    inf_df["avg_caption_length_per_user"] = inf_df["username"].map(avg_caption_length_per_user)


def calculate_sd_caption_length_per_user(df, inf_df, username_col, caption_col):
    sd_caption_length_per_user = df.groupby(username_col)[caption_col].apply(lambda x: x.str.len().std())
    inf_df["sd_caption_length_per_user"] = inf_df["username"].map(sd_caption_length_per_user)
