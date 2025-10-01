import pandas as pd
import os
import csv
import unicodedata


working_dir = os.path.expanduser("~/projects/NLP/NLP-language-detection/data")
langs = {"ara": "ar", "deu": "de", "eng": "en", "fra": "fr", "rus": "ru", "vie": "vi", "lao": "lo"}


for lang in langs[:-1]:
    with open(f"{working_dir}/{lang}_news_2019_10K/{lang}_news_2019_10K-sentences.txt") as file:
        lines = file.readlines()

    sentences = [line.strip().split(sep="\t", maxsplit=1)[1]  for line in lines]

    df = pd.DataFrame(data=sentences[:9000], columns=["text"])

    # Remove unicode control characters like U+200E
    df['text'] = df['text'].apply(lambda x: ''.join(
        char for char in x 
        if unicodedata.category(char)[0] != 'C' or char in ' \n\r\t'
    ))

    # Add labels
    df["label"] = langs[lang]

    # Write data to train set
    df.to_csv(f"{working_dir}/clean_data/train.txt", mode='a', index=False, header=False, columns=["text", "label"], sep="\t", quoting=csv.QUOTE_NONE)

    df_test = pd.DataFrame(data=sentences[9000:], columns=["text"])

    # Remove unicode control characters like U+200E
    df_test['text'] = df_test['text'].apply(lambda x: ''.join(
        char for char in x 
        if unicodedata.category(char)[0] != 'C' or char in ' \n\r\t'
    ))

    # Write data to test set
    df_test.to_csv(f"{working_dir}/clean_data/test.txt", mode='a', index=False, header=False, columns=["text"], sep="\t", quoting=csv.QUOTE_NONE)


# Create true label file
with open(f"{working_dir}/clean_data/test_labels.txt", "w", encoding="utf-8") as f:
    for lang in langs:
        for _ in range(1000):
            f.write(f"{langs[lang]}\n")

