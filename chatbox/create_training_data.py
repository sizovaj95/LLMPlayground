import pandas as pd
import re


def create_conversations_data(conv_sequences: pd.Series, lines_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(columns=["Statement", "Reply"])
    for j, sequence in enumerate(conv_sequences):
        if j % 100 == 0:
            print(f"{j}/{len(conv_sequences)}")
        codes = re.findall(r"L\d{1,7}", sequence)
        texts1, texts2 = [], []
        for i, code in enumerate(codes[:-1]):
            try:
                text1 = lines_dict[code]
                text2 = lines_dict[codes[i+1]]
                texts1.append(text1)
                texts2.append(text2)
            except KeyError:
                break
        df_ = pd.DataFrame({"Statement": texts1, "Reply": texts2})
        df = pd.concat([df, df_])
    df.reset_index(inplace=True, drop=True)
    return df


def clean_text(text: str) -> str:
    text = re.subn(r"\s{2,}", " ", text)[0]
    text = re.subn(r"-{2,}", "-", text)[0]
    text = re.subn(r"\.\.\.", ".", text)[0]
    text = re.subn(r"\.\.\?", "?", text)[0]
    text = re.subn(r"\s+\?", "?", text)[0]
    text = re.subn(r"\s'\s", "'", text)[0]
    return text


def filter_data():
    data = pd.read_csv("data/movie_conversations_data.csv")
    data.dropna(inplace=True)
    data['longest_text_length'] = data[['Statement', 'Reply']].apply(
        lambda row: max(len(row['Statement'].split()), len(row['Reply'].split())), axis=1)
    data['longest_text_length'] = data['longest_text_length'].apply(lambda x: x * 1.5)
    data = data[data['longest_text_length'] < 35]
    data = data.drop(['longest_text_length'], axis=1)
    data['Statement'] = data['Statement'].apply(lambda x: clean_text(x))
    data['Reply'] = data['Reply'].apply(lambda x: clean_text(x))
    data.to_csv("data/movie_conversations_data_filtered.csv", index=False)


def main():
    with open("data/movie_lines.tsv",  "rb") as f:
        file = f.read()
    movie_lines = re.split(r"(?=L\d{1,7}\tu\d{1,6}\tm\d{1,6}\t.{1,30}?\t)", file.decode())
    movie_lines = [l for l in movie_lines if len(re.findall(r"L\d", l)) == 1]
    lines_dict = {}
    for line in movie_lines:
        if match := re.search(r"(L\d{1,7})\tu\d{1,6}\tm\d{1,6}\t.{1,30}?\t(.+)", line):
            code = match[1]
            text = match[2].strip("\r")
            text = re.subn("\t", " ", text)[0]
            lines_dict[code] = text
    movie_conversations = pd.read_csv("data/movie_conversations.tsv", sep='\t',
                                      names=["Person1", "Person2", "MovieID", "LineCodes"])
    conv_sequences = movie_conversations["LineCodes"]
    data = create_conversations_data(conv_sequences, lines_dict)
    data.to_csv("data/movie_conversations_data.csv", index=False)


if __name__ == "__main__":
    filter_data()
    # main()
