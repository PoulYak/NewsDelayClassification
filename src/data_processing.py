import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

nltk.download('punkt')


class NewsDataset(Dataset):
    """Class of news to train and validate"""

    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, stemmer: SnowballStemmer,
                 target_col="label", sentence_raw="sentence", stem_flg: bool = False, clean_flg: bool = False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.stemmer = stemmer
        self.sentence_col = "sentence_stem"
        self.target_col = target_col
        self.sentence_raw = sentence_raw

        df[self.sentence_col] = df[self.sentence_raw].copy()
        if clean_flg:
            df[self.sentence_col] = df[self.sentence_col].apply(self.clean_text)
        if stem_flg:
            df[self.sentence_col] = df[self.sentence_col].apply(self.stem_text)

    def clean_text(self, text: str) -> str:
        """Delete any symbols except words

        :param text: str:
        :param text: str:

        """
        words = nltk.word_tokenize(text)
        words = [word for word in words if word.isalpha()]
        cleaned_text = ' '.join(words)
        return cleaned_text

    def stem_text(self, text: str) -> str:
        """Stem text method

        :param text: str:
        :param text: str:

        """
        words = nltk.word_tokenize(text)
        stemmed_words = [self.stemmer.stem(word) for word in words]
        stemmed_text = ' '.join(stemmed_words)

        return stemmed_text

    def __len__(self) -> int:
        return len(self.df)

    def raw_text_from(self, idx: int) -> str:
        """Raw text from dataset

        :param idx: int:
        :param idx: int:

        """
        return self.df[self.sentence_raw].iloc[idx]

    def __getitem__(self, idx: int) -> dict:
        text = self.df[self.sentence_col].iloc[idx]
        label = self.df[self.target_col].iloc[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class NewsDatasetTest(NewsDataset):
    """Class of news to test"""

    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, stemmer, target_col="label",
                 sentence_raw="text", stem_flg: bool = False, clean_flg: bool = False):
        super().__init__(df, tokenizer, max_len, stemmer, sentence_raw=sentence_raw)

    def __getitem__(self, idx) -> dict:
        text = self.df[self.sentence_col].iloc[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def add_augmented_data(dataframe: pd.DataFrame, filename: str, label: int):
    """Method added news kind of label was generated using chatGPT

    :param dataframe: pd.DataFrame:
    :param filename: str:
    :param label: int:

    """
    augmented_labels = {"sentence": [], "label": []}
    with open(filename, "r") as fin:
        for line in fin.readlines():
            augmented_labels["sentence"].append(line)
            augmented_labels["label"].append(label)
    df = pd.concat([dataframe, pd.DataFrame(augmented_labels)], axis="rows").reset_index(drop=True)
    return df


def get_dataset_dataloader(data: pd.DataFrame, tokenizer: BertTokenizer, batch_size: int, shuffle: bool,
                           num_workers: int, max_len: int, stemmer: SnowballStemmer):
    """Get dataset and dataloader in train/val cases

    :param data: pd.DataFrame:
    :param tokenizer: BertTokenizer:
    :param batch_size: int:
    :param shuffle: bool:
    :param num_workers: int:
    :param max_len: int:
    :param stemmer: SnowballStemmer:

    """
    params = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }

    dataset = NewsDataset(data, tokenizer, max_len, stemmer, stem_flg=False, clean_flg=True)
    loader = DataLoader(dataset, **params)

    return dataset, loader


def get_test_dataset_dataloader(data: pd.DataFrame, tokenizer: BertTokenizer, batch_size: int, shuffle: bool,
                                num_workers: int, max_len: int, stemmer: SnowballStemmer):
    """Get dataset and dataloader in test case

    :param data: pd.DataFrame:
    :param tokenizer: BertTokenizer:
    :param batch_size: int:
    :param shuffle: bool:
    :param num_workers: int:
    :param max_len: int:
    :param stemmer: SnowballStemmer:

    """
    params = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }

    dataset = NewsDatasetTest(data, tokenizer, max_len, stemmer, sentence_raw="text", stem_flg=False, clean_flg=True)
    loader = DataLoader(dataset, **params)

    return dataset, loader
