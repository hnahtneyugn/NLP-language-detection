import pandas as pd
import os
import csv
import unicodedata
import re


working_dir = os.path.expanduser("~/projects/NLP/NLP-language-detection/data")


with open(f"{working_dir}/lao_news_2019_300K/train.txt", encoding="utf-8") as f:
    lines = f.read().splitlines()   # keep each line as one string

df = pd.DataFrame(lines, columns=["text"])


df_train = df.iloc[:240000].copy()
df_test = df.iloc[240000:].copy()


def clean_text(s):
    s = ''.join(
        ch for ch in s
        if (
            unicodedata.category(ch)[0] in ['L', 'N']   # Letters & Numbers
            or ch in " .,!?;:'\"-–()[]{}"               # basic punctuation
            or ch in "\n\r\t "                          # whitespace
        )
    )
    # Collapse multiple whitespace into a single space
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Clean the train data
df_train["text"] = df_train["text"].apply(clean_text)

# Add labels
df_train["label"] = "lo"         

df_train.to_csv(f"{working_dir}/clean_data/train.txt", mode='a', index=False, header=False, columns=["text", "label"], sep="\t", quoting=csv.QUOTE_NONE)


# Clean the test data
df_test["text"] = df_test["text"].apply(clean_text)

df_test.to_csv(f"{working_dir}/clean_data/test.txt", mode='a', index=False, header=False, columns=["text"], sep="\t", quoting=csv.QUOTE_NONE)

