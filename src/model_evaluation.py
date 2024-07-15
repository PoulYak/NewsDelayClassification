import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from typing import Dict, List
from tqdm import tqdm


def test(model: nn.Module, loader: DataLoader, device: str):
    """Test of model

    :param model: nn.Module:
    :param loader: DataLoader:
    :param device: str:

    """
    model.eval()
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0), total=len(loader)):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids).flatten()
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())

    return fin_outputs


def load_model(checkpoint_path, model_name, device, model_class):
    model = model_class(model_name=model_name)

    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
    model.eval()
    return model


import nltk


def clean_text(text):
    words = nltk.word_tokenize(str(text))
    words = [word for word in words if word.isalpha()]
    cleaned_text = ' '.join(words)

    return cleaned_text


def predict(text, model, tokenizer, max_len):
    text = [clean_text(sentence) for sentence in text]
    print(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.cpu()


class NewsSomeDetector:
    def __init__(self, news_df: pd.DataFrame, tokenizer: BertTokenizer, model, max_len: int = 256, device: str = "cuda",
                 batch_size: int = 16):
        self.model = model
        self.news_df = news_df
        self.max_len = max_len
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.model = model
        self.news2sentences: Dict[int, List[int]] = {}
        self.sentence2news: Dict[int, int] = {}
        self.sentences = self.__split_by_sentences()
        self.outputs = None

    def clean_text(self, text):
        words = nltk.word_tokenize(str(text))
        words = [word for word in words if word.isalpha()]
        cleaned_text = ' '.join(words)

        return cleaned_text

    def __clean_split_text(self, news: str):
        """Clean and split text to sentences by .!?\n"""
        news = self.clean_text(news)
        window_length = 200
        overlap_length = 100
        words = news.split()

        sentences = []
        sentence = []
        next_sentence = []
        for i in range(len(words)):
            if len(" ".join(sentence)) < window_length:
                if len(" ".join(sentence)) >= overlap_length:
                    next_sentence.append(words[i])
                sentence.append(words[i])

            else:
                sentences.append(" ".join(sentence))
                sentence = next_sentence
                next_sentence = []
                sentence.append(words[i])
        if sentence:
            sentences.append(" ".join(sentence))
        return sentences

    def __split_by_sentences(self):
        sentences: List[str] = []
        for news_idx in range(len(self.news_df)):
            title = self.news_df.title.iloc[news_idx]
            text = self.news_df.text.iloc[news_idx]
            for sentence in self.__clean_split_text(str(title)) + self.__clean_split_text(str(text)):
                sentence_idx = len(sentences)
                sentences.append(sentence.strip())
                self.sentence2news[sentence_idx] = news_idx
                self.news2sentences.get(news_idx, []) + [sentence_idx]
        return sentences

    def run(self):
        outputs = None
        for i in tqdm(range(0, len(self.sentences), self.batch_size)):
            inputs = self.__preprocess(self.sentences[i:i + self.batch_size])

            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():  # Disable gradient calculation

                if outputs is None:
                    outputs = self.model(**inputs).cpu()
                else:
                    outputs = np.concatenate([outputs, self.model(**inputs).cpu()], axis=0)
        self.outputs = outputs.flatten()

    def predict(self):
        outputs = self.outputs
        news_outputs = np.zeros(len(self.news_df))

        for i in range(len(outputs)):
            news_idx = self.sentence2news[i]
            news_idx = int(news_idx)
            news_outputs[news_idx] = max(news_outputs[news_idx], outputs[i])

        return np.array(news_outputs)

    def __preprocess(self, texts: List[str]):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return encodings
