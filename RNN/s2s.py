import torch
import re
import unicodedata
import random
import time
import math
import numpy as np
import torch.nn as nn
import lightning as L

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from io import open
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tbparse import SummaryReader
from itertools import chain

SOS_token = 0
EOS_token = 1
max_length = 10

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def normalizeStringRu(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-яА-Я!?]+", r" ", s)
    return s.strip()


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = (
        open("%s_%s_vocab.txt" % (lang1, lang2), encoding="utf-8")
        .read()
        .strip()
        .split("\n")
    )

    # Split every line into pairs and normalize
    pairs = [l.split("\t")[:2] for l in lines]
    eng = [normalizeString(s[0]) for s in pairs]
    rus = [normalizeStringRu(s[1]) for s in pairs]
    pairs = list(zip(rus, eng))

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs
    
    


eng_prefixes = (
    "i am ",
    "i m ",
    "he is",
    "he s ",
    "she is",
    "she s ",
    "you are",
    "you re ",
    "we are",
    "we re ",
    "they are",
    "they re ",
)


def filterPair(p):
    return (
        len(p[0].split(" ")) < max_length
        and len(p[1].split(" ")) < max_length
        and p[1].startswith(eng_prefixes)
    )


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
    

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def get_dataloaders(batch_size):
    input_lang, output_lang, pairs = prepareData("eng", "rus", False)

    n = len(pairs)
    input_ids = np.zeros((n, max_length), dtype=np.int32)
    target_ids = np.zeros((n, max_length), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, : len(inp_ids)] = inp_ids
        target_ids[idx, : len(tgt_ids)] = tgt_ids

    # Prepare train/val/test split of sentance pairs
    all_pairs_idx = np.random.permutation(len(input_ids))
    train_len = int(0.8 * len(input_ids))
    val_len = int(0.15 * len(input_ids))
    train_pairs, val_pairs, test_pairs = np.split(
        ary=all_pairs_idx, indices_or_sections=[train_len, train_len + val_len]
    )

    # Prepare datasets
    datasets = {}
    for split, pair_ids in zip(
        ["train", "val", "test"], [train_pairs, val_pairs, test_pairs]
    ):
        datasets[split] = TensorDataset(
            torch.LongTensor(input_ids[pair_ids, ...]),
            torch.LongTensor(target_ids[pair_ids, ...]),
        )
    # Prepare dataloaders
    train_dataloader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True
    )

    val_dataloader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)

    test_dataloader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
    return (
        input_lang,
        output_lang,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        test_pairs,
    )
    



class Seq2SeqPipeline(L.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        exp_name="baseline",
        criterion=nn.NLLLoss(),
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters()), lr=0.001
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        input_tensor, target_tensor = batch

        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, _, _ = self.decoder(
            encoder_outputs, encoder_hidden, target_tensor
        )

        loss = self.criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)
        )

        self.log("Loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_tensor, target_tensor = batch

        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        decoder_outputs, _, _ = self.decoder(
            encoder_outputs, encoder_hidden, target_tensor
        )

        loss = self.criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)
        )

        self.log("Loss/val", loss, prog_bar=True)
        
        
def tbparse_visual(log_path):
    reader = SummaryReader(log_path)
    df = reader.scalars

    plt.figure(figsize=(12, 4))
    for tag in df.tag.unique():
        if "Loss" in tag:
            tag_data = df.query("tag == @tag").sort_values(by="step")
            plt.plot(tag_data.step, tag_data.value, label=tag)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.show()
    
def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor.to(device))
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs.to(device), encoder_hidden.to(device)
        )

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn
