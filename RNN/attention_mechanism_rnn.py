#pip install -q lightning tbparse

import torch 
import torch.nn as nn
import random as rnd
import numpy as np
import torch.nn.functional as F
import urllib.request
import os
import sys
import importlib.util

import lightning as L

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)

set_random_seed(42)

# Code for sequence to sequence translation
# Download s2s.py if it doesn't exist
s2s_path = os.path.join(os.path.dirname(__file__), "s2s.py")
if not os.path.exists(s2s_path):
    urllib.request.urlretrieve("https://edunet.kea.su/repo/EduNet_NLP-web_dependencies/L04/s2s.py", s2s_path)

vocab_path = os.path.join(os.path.dirname(__file__), "eng_rus_vocab.txt")
if not os.path.exists(vocab_path):
    urllib.request.urlretrieve("https://edunet.kea.su/repo/EduNet_NLP-web_dependencies/datasets/eng_rus_vocab.txt", vocab_path)

# Import from local s2s.py file instead of installed package
spec = importlib.util.spec_from_file_location("s2s_local", s2s_path)
s2s_module = importlib.util.module_from_spec(spec)
sys.modules["s2s_local"] = s2s_module
spec.loader.exec_module(s2s_module)

# Import functions from local module
PrepareData = s2s_module.prepareData
get_dataloaders = s2s_module.get_dataloaders
Seq2SeqPipeline = s2s_module.Seq2SeqPipeline

input_lang, output_lang, pairs = PrepareData("eng", "rus", False)
print(rnd.choice(pairs))


# Encoder / Decoder architect

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return(output, hidden)

SOS_token = 0
EOS_token = 1
max_length = 10

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        device = next(self.parameters()).device
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device
        ).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(max_length):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # teacher forcing
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # no teacher forcing
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach() # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return (
            decoder_outputs,
            decoder_hidden,
            None,
        ) # We return 'None' for consistecy in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


# Attention Layer

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p = 0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        device = next(self.parameters()).device
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device
        ).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # teacher forcing
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # no teacher forcing
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach() # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

# Training the model
            
hidden_size = 512
batch_size = 256

(
    input_lang,
    output_lang,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    test_pair_ids,
) = get_dataloaders(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)


L.seed_everything(42)

checkpoint_callback = ModelCheckpoint(monitor="Loss/val", mode="min", filename="best")

exp_name = f"baseline"
trainer = Trainer(
    max_epochs=100,
    logger=TensorBoardLogger(save_dir="logs/seq2seq", name=exp_name),
    num_sanity_val_steps=1,
    callbacks=[checkpoint_callback],
    log_every_n_steps=5,
)

pipeline = Seq2SeqPipeline(encoder=encoder, decoder=decoder)

"""
# Train the model
trainer.fit(
    model=pipeline,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
"""
# download pre-saved training logs 

base_path = f"logs/seq2seq/{exp_name}"
if os.path.exists(base_path):
    last_version = sorted(os.listdir(base_path))[-1]
    ckpt_path = f"{base_path}/{last_version}/checkpoints/best.ckpt"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location=device)

        print(f"Checkpoint has been loaded from {ckpt_path}")
        print(f"Best model has been saved on the {checkpoint['epoch']} epoch")

        state_dict_encoder = {}
        state_dict_decoder = {}
        for key in checkpoint["state_dict"].keys():
            if key.startswith("encoder."):
                state_dict_encoder[key[len("encoder.") :]] = checkpoint["state_dict"][key]
            elif key.startswith("decoder."):
                state_dict_decoder[key[len("decoder.") :]] = checkpoint["state_dict"][key]

        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

        encoder.load_state_dict(state_dict_encoder)
        decoder.load_state_dict(state_dict_decoder)
    else:
        print(f"Checkpoint file not found: {ckpt_path}")
else:
    print(f"Logs directory not found: {base_path}. Please train the model first.")


# Testing the model

evaluate = s2s_module.evaluate

def evaluateRandomly(encoder, decoder, n=10):
    encoder.eval()
    decoder.eval()

    eng = []
    pred_tokens = []

    with torch.no_grad():
        for i in range(n):
            pair_id = rnd.choice(test_pair_ids)
            pair = pairs[pair_id]
            print("RUS", pair[0])
            print("ENG", pair[1])
            output_words, _ = safeEvaluate(encoder, decoder, pair[0], input_lang, output_lang)
            eng.append(pair[1])
            pred_tokens.append(output_words[:-1]) # remove EOS token
            output_sentence = " ".join(output_words)
            print("pred_tokens", output_sentence)
            print("")
        return eng, pred_tokens

# Visualize the attention mechanism

import matplotlib.pyplot as plt
import re

def normalizeSentence(sentence):
    """Normalize sentence by separating punctuation from words"""
    # Add spaces around punctuation marks
    sentence = re.sub(r"([.!?,;:])", r" \1 ", sentence)
    # Remove extra spaces
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence.strip()

def safeIndexesFromSentence(lang, sentence):
    """Safely convert sentence to indexes, skipping words not in vocabulary"""
    indexes = []
    for word in sentence.split(" "):
        if word in lang.word2index:
            indexes.append(lang.word2index[word])
        # Skip words/punctuation not in vocabulary
    return indexes

def safeTensorFromSentence(lang, sentence):
    """Safely convert sentence to tensor, skipping words not in vocabulary"""
    indexes = safeIndexesFromSentence(lang, sentence)
    if len(indexes) == 0:
        # If no valid words, return a tensor with just EOS token
        indexes = [EOS_token]
    else:
        indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)

def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Get attention matrix
    attn_matrix = attentions.cpu().numpy()
    
    # Get dimensions
    input_words = input_sentence.split(" ")
    n_output = len(output_words)
    n_input = len(input_words)
    
    # Display attention matrix
    cax = ax.matshow(attn_matrix[:n_output, :n_input], cmap="bone")
    fig.colorbar(cax)

    # Set up axes with correct number of ticks
    ax.set_xticks(range(n_input))
    ax.set_yticks(range(n_output))
    
    ax.set_xticklabels(input_words, rotation=90)
    ax.set_yticklabels(output_words)
    
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.show()

def safeEvaluate(encoder, decoder, sentence, input_lang, output_lang):
    """Safe version of evaluate that handles unknown words"""
    device = next(encoder.parameters()).device
    with torch.no_grad():
        input_tensor = safeTensorFromSentence(input_lang, sentence).to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(
            encoder_outputs, encoder_hidden
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

def evaluateAndShowAttention(input_sentence):
    # Normalize the input sentence before evaluation
    normalized_sentence = normalizeSentence(input_sentence)
    output_words, attentions = safeEvaluate(
        encoder, decoder, normalized_sentence, input_lang, output_lang
    )
    print("input =", input_sentence)
    print("normalized =", normalized_sentence)
    print("output =", " ".join(output_words))
    showAttention(normalized_sentence, output_words, attentions[0, : len(output_words) :])

evaluateAndShowAttention("я рад, что у тебя все получилось")