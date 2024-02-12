# Description: This file contains the functions to calculate the influencer characteristics
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