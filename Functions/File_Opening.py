import pandas as pd
import csv


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

    return df_ins


def get_unique_values(dataframe, column_name):
    # Convert values in the specified column to uppercase and remove spaces
    dataframe[column_name] = dataframe[column_name].apply(lambda x: x.replace(" ", "").upper())

    # Create a set with unique values from the specified column
    unique_values = set(dataframe[column_name].unique())

    return unique_values


def merge_and_drop_rows(df1, df2, on_column):
    # Merge the two DataFrames on the specified column
    merged_df = pd.merge(df1, df2, on=on_column, how='left', indicator=True)

    # Drop rows where the ID is present in df1
    result_df = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)

    return result_df
