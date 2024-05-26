# Description: This file contains the functions to calculate the influencer characteristics
from matplotlib_venn import venn3
from matplotlib_venn import venn2
import emoji
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


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
    return all_common_elements


def get_common_elements(set_1_en, set_2_en, set_3_en):
    # Merge all common elements for English
    common_usernames = set_1_en.intersection(set_2_en, set_3_en)

    return common_usernames


def get_common_elements_all_languages(set_1_en, set_2_en, set_3_en, set_1_nl, set_2_nl, set_3_nl):
    common_in_1_and_2_en = set_1_en.intersection(set_2_en)
    common_in_1_and_3_en = set_1_en.intersection(set_3_en)
    common_in_2_and_3_en = set_2_en.intersection(set_3_en)

    # Merge all common elements for English
    all_common_elements1_en = common_in_1_and_2_en.union(common_in_1_and_3_en, common_in_2_and_3_en)

    common_in_1_and_2_nl = set_1_nl.intersection(set_2_nl)
    common_in_1_and_3_nl = set_1_nl.intersection(set_3_nl)
    common_in_2_and_3_nl = set_2_nl.intersection(set_3_nl)

    # Merge all common elements for Dutch
    all_common_elements2_nl = common_in_1_and_2_nl.union(common_in_1_and_3_nl, common_in_2_and_3_nl)

    # Get common elements across both English and Dutch
    common_elements_both_languages = all_common_elements1_en.intersection(all_common_elements2_nl)

    return common_elements_both_languages


def plot_common_in_languages(set_1_en, set_2_en, set_3_en, set_1_nl, set_2_nl, set_3_nl):
    common_in_1_and_2_en = set_1_en.intersection(set_2_en)
    common_in_1_and_3_en = set_1_en.intersection(set_3_en)
    common_in_2_and_3_en = set_2_en.intersection(set_3_en)

    # Merge all common elements for English
    all_common_elements1_en = common_in_1_and_2_en.union(common_in_1_and_3_en, common_in_2_and_3_en)

    common_in_1_and_2_nl = set_1_nl.intersection(set_2_nl)
    common_in_1_and_3_nl = set_1_nl.intersection(set_3_nl)
    common_in_2_and_3_nl = set_2_nl.intersection(set_3_nl)

    # Merge all common elements for Dutch
    all_common_elements2_nl = common_in_1_and_2_nl.union(common_in_1_and_3_nl, common_in_2_and_3_nl)

    # Plot Venn diagram
    venn2([all_common_elements1_en, all_common_elements2_nl], ('English', 'Dutch'))
    plt.title("Common Influencers in English and Dutch")
    plt.show()


def filter_rows_by_count(df, column_name, min_count):
    # Get the counts of each value in the specified column
    value_counts = df[column_name].value_counts()

    # Create a mask for values that appear at least the specified minimum count
    mask = value_counts[df[column_name]].values >= min_count

    # Filter the DataFrame
    result = df.loc[mask]

    return result


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


def plot_time_between_posts(df1, df2, df3, label1, label2, label3, title, lower_limit=0, upper_limit=1000):
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


def draw_network(dataframe, column, filename):
    for i in range(len(dataframe)):
        dataframe[column].iloc[i] = set(dataframe[column].iloc[i])
    G = nx.Graph()
    for i in range(len(dataframe)):
        for j in range(i + 1, len(dataframe)):  # Avoid duplicate pairs and self-loops
            # Compute the intersection set
            intersection_set = dataframe[column].iloc[i].intersection(dataframe[column].iloc[j])
            # Check if intersection set is non-empty
            if intersection_set:
                for k in range(len(list(intersection_set))):
                    G.add_edge(i, j, label=list(intersection_set)[k])
    nx.write_graphml(G, filename)
    return G


def calculate_post_count(df, inf_df, username_col):
    post_count_per_user = df.groupby(username_col).size()
    inf_df["post_count"] = inf_df["username"].map(post_count_per_user)


def calculate_avg_hashtag_per_post(df, inf_df, username_col, hashtag_col):
    avg_hashtag_per_post_per_user = df.groupby(username_col)[hashtag_col].apply(lambda x: x.apply(len).sum() / len(x))
    inf_df["avg_hashtag_per_post"] = inf_df["username"].map(avg_hashtag_per_post_per_user)


def calculate_sd_hashtag_per_post(df, inf_df, username_col, hashtag_col):
    sd_hashtag_per_post_per_user = df.groupby(username_col)[hashtag_col].apply(lambda x: x.apply(len).std())
    inf_df["sd_hashtag_per_post"] = inf_df["username"].map(sd_hashtag_per_post_per_user)


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


def extract_unique_hashtags(dataframe, column_name):
    list_hashtags = dataframe[column_name].tolist()

    # Flatten list of lists
    flat_list = [tag.strip() for sublist in list_hashtags for tag in str(sublist).split(',')]

    # Remove empty strings
    flat_list = [tag for tag in flat_list if tag]

    # Convert to set to get unique values
    unique_hashtags = set(flat_list)
    return unique_hashtags


def extract_mentions(parts):
    at_mentions = [word for word in str(parts).split() if str(word).startswith('@')]
    return ' '.join(at_mentions)


# Function to extract URLs starting with 'http'
def extract_urls(raw_text):
    url = [word for word in raw_text.split() if str(word).startswith('http')]
    return ' '.join(url)


# Function to extract hashtags starting with '#'
def extract_hashtags(raw_text):
    hashtags = [word for word in raw_text.split() if str(word).startswith('#')]
    return ' '.join(hashtags)


# Function to extract emojis
def extract_emojis(raw_text):
    spaced_sentence = ' '.join(raw_text)
    emojis = [word for word in spaced_sentence.split() if emoji.is_emoji(word)]
    emojis = list(set(emojis))
    return ' '.join(emojis)


def split_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: x.split() if isinstance(x, str) else x)
    return df


def plot_unique(df1ai, df2ai, df3ai, df1real, df2real, df3real, column_name):
    # Extracting data
    unihasins_en_ai = len(extract_unique_hashtags(df1ai, column_name))
    unihasins_nl_ai = len(extract_unique_hashtags(df2ai, column_name))
    unihastt_en_ai = len(extract_unique_hashtags(df3ai, column_name))

    unihasins_en = len(extract_unique_hashtags(df1real, column_name))
    unihasins_nl = len(extract_unique_hashtags(df2real, column_name))
    unihastt_en = len(extract_unique_hashtags(df3real, column_name))

    platforms_ai = ['Instagram English', 'Instagram Dutch', 'TikTok English']
    unique_counts_ai = [unihasins_en_ai, unihasins_nl_ai, unihastt_en_ai]
    unique_counts = [unihasins_en, unihasins_nl, unihastt_en]

    plt.figure(figsize=(10, 6))
    p1 = plt.bar(platforms_ai, unique_counts_ai, color='skyblue', label='AI')
    p2 = plt.bar(platforms_ai, unique_counts, color='orange', bottom=unique_counts_ai, label='Non-AI')

    plt.xlabel('Platform')
    plt.ylabel(f'Number of Unique {column_name.capitalize()}')
    plt.title(f'Number of Unique {column_name.capitalize()} per Platform (AI vs. Non-AI)')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.legend()
    plt.tight_layout()

    plt.show()


def get_unique_counts(df1, df2, df3, column_name):
    unihasins1 = extract_unique_hashtags(df1, column_name)
    unihasins2 = extract_unique_hashtags(df2, column_name)
    unihasins3 = extract_unique_hashtags(df3, column_name)

    unique_counts = {
        'Platform': ['Instagram', 'TikTok', 'YouTube'],
        'Unique Counts': [len(unihasins1), len(unihasins2), len(unihasins3)]
    }

    df_unique_counts = pd.DataFrame(unique_counts)
    return df_unique_counts


def calculate_avg_per_post(df, hashtag_col):
    avg_hashtag_per_post = df[hashtag_col].apply(len).sum() / len(df)
    return avg_hashtag_per_post


def get_avg_counts(df1, df2, df3, column_name):
    unihasins1 = calculate_avg_per_post(df1, column_name)
    unihasins2 = calculate_avg_per_post(df2, column_name)
    unihasins3 = calculate_avg_per_post(df3, column_name)

    unique_counts = {
        'Platform': ['Instagram', 'TikTok', 'YouTube'],
        'Unique Counts': [unihasins1, unihasins2, unihasins3]
    }

    df_unique_counts = pd.DataFrame(unique_counts)
    return df_unique_counts


def get_percentage_counts(df1, df2, df3, column_name):
    unihasins1 = df1[column_name].apply(len).sum() / len(df1) / len(extract_unique_hashtags(df1, column_name))
    unihasins2 = df2[column_name].apply(len).sum() / len(df2) / len(extract_unique_hashtags(df2, column_name))
    unihasins3 = df3[column_name].apply(len).sum() / len(df3) / len(extract_unique_hashtags(df3, column_name))

    unique_counts = {
        'Platform': ['Instagram', 'TikTok', 'YouTube'],
        'Unique Counts': [(unihasins1), (unihasins2), (unihasins3)]
    }

    df_unique_counts = pd.DataFrame(unique_counts)
    return df_unique_counts


def get_percentage_counts_total(df1, df2, df3, df7, df8, df9, column_name):
    unihasins1 = len(extract_unique_hashtags(df1, column_name)) / (
                len(extract_unique_hashtags(df1, column_name)) + len(extract_unique_hashtags(df7, column_name)))
    unihasins2 = len(extract_unique_hashtags(df2, column_name)) / (
                len(extract_unique_hashtags(df2, column_name)) + len(extract_unique_hashtags(df8, column_name)))
    unihasins3 = len(extract_unique_hashtags(df3, column_name)) / (
                len(extract_unique_hashtags(df3, column_name)) + len(extract_unique_hashtags(df9, column_name)))

    unique_counts = {
        'Platform': ['Instagram', 'TikTok', 'YouTube'],
        'Unique Counts': [(unihasins1), (unihasins2), (unihasins3)]
    }

    df_unique_counts = pd.DataFrame(unique_counts)
    return df_unique_counts


# Function to extract unique hashtags and calculate average ratio
def extract_hashtag_ratio(row, unique_hashtags):
    row_hashtags = set(row)
    ratio = len(row_hashtags) / len(unique_hashtags) * 100
    return ratio


def extract_unique_hashtags_ratio(dataframe, column_name):
    list_hashtags = dataframe[column_name].tolist()
    flat_list = [tag.strip() for sublist in list_hashtags for tag in str(sublist).split(',')]
    flat_list = [tag for tag in flat_list if tag]
    unique_hashtags = set(flat_list)

    # Calculate ratio for each row
    ratio_list = [extract_hashtag_ratio(row, unique_hashtags) for row in list_hashtags]

    # Calculate the average ratio
    average_ratio = sum(ratio_list) / len(ratio_list)

    return average_ratio


def check_for_ad(caption):
    adlist = ['promo',
              'partner',
              '#ad',
              'gift',
              '#partner',
              'sponsor',
              'merch',
              'gifted',
              ' #ad',
              '#sponsored',
              '#partnership',
              '#collab',
              '#merch',
              '#spon',
              '#advertisement',
              '#gifted',
              ' partner',
              ' gift',
              ' #partner',
              'ambassador',
              'advertising',
              'advertisement',
              ' #merch',
              '#sponsor',
              ' #collab',
              'promotion',
              ' #sponsor',
              'advert',
              ' merch',
              ' #spon',
              'spon',
              ' #sponsored',
              ' sponsor',
              ' #ambassador',
              ' promo',
              ' gifted',
              ' promotion',
              ' advertising']
    for keyword in adlist:
        if keyword in caption:
            return 1
    return 0

def check_for_ad_dutch(caption):
    adlist = ['#ad',
     '#partner',
     'partner',
     '#spon',
     '#merch',
     'merch',
     ' #merch',
     ' #spon',
     ' #partner',
     ' gift',
     'gift',
     ' #ad',
     'reclame',
     '#collab',
     'advertentie',
     '#promotion',
     'promo',
     ' #gifted',
     '#ambassador',
     'gifted',
     'spon',
     ' #reclame',
     '#gifted',
     '#sponsor',
     'sponsor',
     '#advertentie',
     '#partnership',
     ' advertentie',
     ' partner',
     '#reclame',
     'ambassador',
     ' promo',
     ' sponsor',
     'promotion',
     'advertorial',
     ' reclame',
     ' merch']
    for keyword in adlist:
        if keyword in caption:
            return 1
    return 0

def ad_column(df, column):
    df["ad"]=0
    for i in range(len(df)):
        df["ad"].iloc[i]=check_for_ad(df[column].iloc[i])
    return df

def ad_column_dutch(df, column):
    df["ad"]=0
    for i in range(len(df)):
        df["ad"].iloc[i]=check_for_ad_dutch(df[column].iloc[i])
    return df