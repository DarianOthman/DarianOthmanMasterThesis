import pandas as pd
import csv
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
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

def plot_instances_by_week(dataframe, date_column, title='Evolution of Instances by Week', x_label='Date', y_label='Number of Instances', rotation=45):

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
    
def plot_instances_together(df1,df2,df3,timecol1,timecol2,timecol3,label1,label2,label3,title):
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
