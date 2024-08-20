"""Text taken from https://www.kaggle.com/datasets/ashishsinhaiitr/lord-of-the-rings-text
Manually removed Foreword"""


import pandas as pd
from pathlib import Path
import os
import re
import numpy as np
import math

import constants as co
import util

TEXT = "text"


def clean_text(txt: str) -> str:
    txt = txt.strip()
    txt = re.subn(r"_(.+?)_", r"\1", txt)[0]
    txt = re.sub(r"^Chapter \d{1,3}$", '', txt)
    txt = re.sub(r"^BOOK [A-Z]{1,2}$", '', txt)
    if len(txt.split()) == 1:
        txt = ''
    return txt


def remove_chapters(paras: list[str]):
    paras = [par for par in paras if par]
    paras_words = [len(par.split()) for par in paras]
    paras_tup = zip(paras, paras_words)
    _5 = math.ceil(np.quantile(paras_words, q=0.05))
    paras = [par[0] for par in paras_tup if par[1] >= _5]
    return paras


def batch_iterator(size: int = 100):
    folder = Path(__file__).parent.resolve() / 'lotr_text'
    for file in os.scandir(folder):
        print(f"Working with {file.name}")
        with open(file.path, "r") as f:
            text = f.read()
        paras = text.split('\n')
        paras = [par for par in paras if par]
        for i in range(0, len(paras), size):
            yield paras[i: i+size]


def main():
    iterator = batch_iterator()
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        for i, par in enumerate(batch):
            batch[i] = clean_text(par)
        batch = [par for par in batch if par]
        df = pd.DataFrame(batch, columns=[TEXT])
        if os.path.exists(co.DATA_FOLDER / co.LOTR_TEXT_TRAIN_DATA):
            df.to_csv(co.DATA_FOLDER / co.LOTR_TEXT_TRAIN_DATA, index=False, mode="a", header=False)
        else:
            df.to_csv(co.DATA_FOLDER / co.LOTR_TEXT_TRAIN_DATA, index=False)


def clean_data_from_irregular_characters():
    data = pd.read_csv(co.DATA_FOLDER/ co.LOTR_TEXT_TRAIN_DATA)
    data[TEXT] = data[TEXT].apply(lambda x: util.replace_all_non_compliant_chars(x))
    data.to_csv(co.DATA_FOLDER / co.LOTR_TEXT_TRAIN_DATA, index=False)


def make_data_into_equal_batches():
    data = pd.read_csv(co.DATA_FOLDER / co.LOTR_TEXT_TRAIN_DATA)
    text = data[TEXT]
    text_str = ' '.join(text)
    sentences = iter(text_str.split('. '))
    batches = []
    while True:
        try:
            batch = [next(sentences)]
        except StopIteration:
            break
        while len('. '.join(batch)) <= 300:
            try:
                sentence = next(sentences)
                batch.append(sentence)
            except StopIteration:
                break
        text_ = '. '.join(batch)
        if not text_.endswith('.'):
            text_ += '.'
        batches.append(text_)
    new_df = pd.DataFrame(batches, columns=[TEXT])
    new_df.to_csv(co.DATA_FOLDER / "lotr_train_data_equal.csv", index=False)


if __name__ == "__main__":
    make_data_into_equal_batches()
    df1 = pd.read_csv(co.DATA_FOLDER / co.LOTR_TEXT_TRAIN_DATA)
    df2 = pd.read_csv(co.DATA_FOLDER / "lotr_train_data_equal.csv")
    df1
