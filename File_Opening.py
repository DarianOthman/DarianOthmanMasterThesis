import pandas as pd
import csv
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
import ast
import networkx as nx


def read_data(file_path):
    # Read in Instagram data
    with open(file_path, 'r', encoding='utf-8') as f:
        reader_ins = csv.reader(f, delimiter=',')
        rows_ins = list(reader_ins)

    # Making the Instagram dataframe ENGLISH
    df_ins = pd.DataFrame(rows_ins)

    # Making the first row the header
    new_header = df_ins.iloc[0]
    df_ins = df_ins[1:]
    df_ins.columns = new_header

    # Filtering rows with non-empty 'relative_position_1st_disclosure' column
    df_ins = df_ins[df_ins["relative_position_1st_disclosure"] != ""]

    return df_ins


def get_unique_values(dataframe, column_name):
    # Convert values in the specified column to uppercase and remove spaces
    dataframe[column_name] = dataframe[column_name].apply(lambda x: x.replace(" ", "").upper())

    # Create a set with unique values from the specified column
    unique_values = set(dataframe[column_name].unique())

    return unique_values


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


import pandas as pd
import matplotlib.pyplot as plt


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


def plot_time_between_posts(df1, df2, df3, label1, label2, label3, title):
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
    plt.ylim(0, 1000)
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()


def draw_hashtag_network(dataframe, hashtags_column, title, sample_size=1000, k_value=0.2, node_size=2,
                         edge_color='grey', width=0.1, alpha=0.7):
    # Convert the column containing lists of hashtags to lists
    dataframe[hashtags_column] = dataframe[hashtags_column].apply(lambda x: ast.literal_eval(x) if x else "")

    # Filter out rows with non-empty lists of hashtags
    filtered_df = dataframe[dataframe[hashtags_column].apply(lambda x: bool(x))].reset_index(drop=True)

    sample_size = sample_size
    random_sample = filtered_df.sample(n=sample_size, random_state=42)

    G = nx.Graph()
    for words_list in random_sample[hashtags_column]:
        G.add_edges_from(
            [(word1, word2) for i, word1 in enumerate(words_list) for j, word2 in enumerate(words_list) if i < j])

    # Calculate the spring layout
    pos = nx.spring_layout(G, k=k_value)

    # Draw the graph with specified layout
    nx.draw(G, pos, with_labels=False, node_size=node_size, edge_color=edge_color, width=width, alpha=alpha)
    plt.title(title)
    plt.show()


def draw_tag_network(dataframe, tags_column, title, sample_size=1000, k_value=0.2, node_size=2, edge_color='grey',
                     width=0.1, alpha=0.7):
    dataframe[tags_column] = dataframe[tags_column].apply(lambda x: ast.literal_eval(x) if x else "")
    filtered_df = dataframe[dataframe[tags_column].apply(lambda x: bool(x))].reset_index(drop=True)

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
                    G.add_edge(i, j, common_words=list(common_words))

    k_value = k_value  # You can adjust this value to control the repulsion
    pos = nx.spring_layout(G, k=k_value)

    # Draw the graph with specified layout
    nx.draw(G, pos, with_labels=False, node_size=node_size, edge_color=edge_color, width=width, alpha=alpha)
    plt.title(title)
    plt.show()
