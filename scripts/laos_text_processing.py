from datasets import load_dataset
import pandas as pd
import os
import csv
import unicodedata

ds = load_dataset("wannaphong/LaoNewsClassification")

# Save splits to CSVs in ../data/
ds["train"].to_csv("../data/lao_news_2019_10K/train.csv", index=False)
ds["test"].to_csv("../data/lao_news_2019_10K/test.csv", index=False)


current_dir = os.path.expanduser("~/projects/NLP/NLP-language-detection/data")

df = pd.read_csv(f"{current_dir}/lao_news_2019_10K/train.csv")

# Only take 9000 lines as training set
df = df.head(9000)

# Collapse line breaks
df["title"] = df["title"].str.replace(r"\s+", " ", regex=True).str.strip()            

# Add labels
df["label"] = "lo"         

df.to_csv(f"{current_dir}/clean_data/train.txt", mode='a', index=False, header=False, columns=["title", "label"], sep="\t", quoting=csv.QUOTE_NONE)


# Check number of actual lines in the file
with open(f"{current_dir}/clean_data/train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
print("Lines in file:", len(lines))

# Do the same thing with the test set
df_test = pd.read_csv(f"{current_dir}/lao_news_2019_10K/test.csv")

df_test = df_test.head(1000)
df_test


# Collapse line breaks
df_test.loc[:, "title"] = df_test["title"].str.replace(r"\s+", " ", regex=True).str.strip() 

df_test.loc[:, "title"] = df['title'].apply(lambda x: ''.join(
    char for char in x 
    if unicodedata.category(char)[0] != 'C' or char in ' \n\r\t'
))

df_test.to_csv(f"{current_dir}/clean_data/test.txt", mode='a', index=False, header=False, columns=["title"], sep="\t", quoting=csv.QUOTE_NONE)


# Check number of actual lines in the file
with open(f"{current_dir}/clean_data/test.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
print("Lines in file:", len(lines))

