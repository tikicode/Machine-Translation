#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way. 
"""


from __future__ import unicode_literals, print_function, division
import numpy as np
import heapq
from collections import namedtuple
from tqdm import tqdm
import math
from torch import optim
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import argparse
import logging
import random
import time
from io import open

import matplotlib
# if you are running on the gradx/ugradx/ another cluster,
# you will need the following line
# if you run on a local machine, you can comment it out
matplotlib.use('agg')

logging.getLogger('matplotlib.font_manager').disabled = True
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid,
# it can be very easy to confict with other people's jobs.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15

# file scope variable for indicating translation
is_translating = False


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s',
                 src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s',
                 tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################


def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################

class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 7yy
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.LSTM = nn.LSTM(hidden_size, hidden_size)
        self.combine = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input, hidden):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        embedded = self.embedding(input)
        if len(embedded.shape) == 4:
            embedded = embedded.squeeze(dim=2)
        output, hidden = self.LSTM(embedded, hidden)
        return output, hidden

    def get_initial_hidden_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class Attention(nn.Module):
    """ Implementing multiplicative attention
    """

    def __init__(self, hidden_size, n_heads=16, dropout_p=0.1):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.value = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

        self.relu = nn.ReLU()

    def forward(self, query, kv):
        if len(kv.shape) == 2:
            kv = kv.unsqueeze(dim=1)
        batch_size = kv.shape[1]

        k = self.key(kv).permute(1, 0, 2)
        v = self.value(kv).permute(1, 0, 2)
        q = self.query(query).permute(1, 0, 2)

        q = q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.permute(0, 1, 3, 2))
        attn_weights = torch.div(attn_weights, torch.sqrt(
            torch.tensor([self.head_dim]).to(device)))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        context = torch.matmul(self.dropout(attn_weights), v)
        context = context.permute(0, 2, 1, 3).contiguous(
        ).view(-1, batch_size, self.hidden_size)

        attn_weights = torch.sum(
            attn_weights.squeeze(0), dim=0).unsqueeze(dim=0)

        return attn_weights.unsqueeze(dim=0), context


class AttnDecoderRNN(nn.Module):
    """the class for the decoder 
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.emb = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.GRU = nn.GRU(2 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights

        Dropout (self.dropout) should be applied to the word embeddings.
        """
        embedding = self.emb(input)
        embedding = self.dropout(embedding)
        (hidden, cell) = hidden

        attn_weights, context = self.attn(hidden, encoder_outputs)
        rnn_input = torch.cat([embedding, context], 2)
        output, hidden = self.GRU(rnn_input, hidden)
        output = self.out(output)
        log_softmax = F.log_softmax(output, dim=-1)

        return log_softmax, (hidden, cell), attn_weights

    def get_initial_hidden_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, batch_size, max_length=MAX_LENGTH):
    encoder_hidden = encoder.get_initial_hidden_state(batch_size)
    encoder_hidden = (encoder_hidden, encoder_hidden)

    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    loss = 0
    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    decoder_hidden = encoder_hidden

    decoder_input = torch.tensor(
        [[SOS_index]], device=device).repeat(1, batch_size)

    for idx in range(target_tensor.shape[0]):
        decoder_output, decoder_hidden, _ = decoder(
            decoder_input, decoder_hidden, encoder_output)
        for di in range(batch_size):
            loss += criterion(decoder_output[0,
                              di].unsqueeze(0), target_tensor[idx][di])
        decoder_input = target_tensor[idx].transpose(1, 0)

    loss.backward()
    optimizer.step()

    return loss.item()  # total loss


######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, beam_search, max_length=MAX_LENGTH, beam_width=5):
    """
    runs translation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state(1)
        encoder_hidden = (encoder_hidden, encoder_hidden)

        encoder_outputs = torch.zeros(
            max_length, encoder.hidden_size, device=device)

        raw_encoder_output, encoder_hidden = encoder(
            input_tensor, encoder_hidden)
        encoder_output_simplified = torch.squeeze(raw_encoder_output)

        for ei in range(input_length):
            current_encoder_output = encoder_output_simplified[ei, :]
            encoder_outputs[ei] = current_encoder_output

        decoder_input = torch.tensor([[SOS_index]], device=device)
        decoder_hidden = encoder_hidden
        decoder_attentions = torch.zeros(max_length, max_length)
        decoded_words = []
        
        if not beam_search:
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_index:
                    decoded_words.append(EOS_token)
                    break
                else:
                    decoded_words.append(tgt_vocab.index2word[topi.item()])
                decoder_input = topi.squeeze(dim=0).detach()
            return decoded_words, decoder_attentions[:di + 1]

        hypothesis = namedtuple(
            "hypothesis", "hidden, input, attention, prev, logprob, word")
        init_hyp = hypothesis(
            decoder_hidden, decoder_input, None, None, 0, None)
        stacks = [[] for i in range(max_length + 1)]
        stacks[0].append((0, init_hyp))
        for di in range(max_length):
            for h in sorted(stacks[di], key=lambda x: -x[0])[:beam_width]:
                _, h = h
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    h.input, h.hidden, encoder_outputs)
                log_prob, ind = decoder_output.data.topk(beam_width)
                ind = ind.squeeze()
                log_prob = log_prob.squeeze()
                for i in range(beam_width):
                    if ind[i].item != SOS_index:
                        word = None
                        if ind[i].item() == EOS_index:
                            word = EOS_token
                        else:
                            word = tgt_vocab.index2word[ind[i].item()]
                        new_hyp = hypothesis(decoder_hidden,
                                             ind[i].unsqueeze(
                                                 0).unsqueeze(0).detach(),
                                             decoder_attention, h, h.logprob + torch.sum(log_prob[i]).item(), word)
                        stacks[di +
                               1].append((get_score(new_hyp.logprob, di), new_hyp))

        winner = max(stacks[15], key=lambda x: -x[0])
        attention = []
        node = winner[1]

        while node:
            decoded_words.append(node.word)
            attention.append(node.attention)
            node = node.prev

        decoded_words = decoded_words[::-1][1:]
        attention = attention[::-1][1:]

        try:
            index = decoded_words.index("<EOS>")
            decoded_words = decoded_words[:index+1]
        except ValueError:
            pass

        for i in range(len(attention)):
            decoder_attentions[i] = attention[i]

        return decoded_words, decoder_attentions[:di + 1]


def get_score(logprob, length):
    return logprob / float(length + 1e-6) + 1


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, beam_search, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(
            encoder, decoder, pair[0], src_vocab, tgt_vocab, beam_search)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, beam_search, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(
            encoder, decoder, pair[0], src_vocab, tgt_vocab, beam_search)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions):
    """visualize the attention mechanism. And save it to a file. 
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    fig = plt.figure()

    input_sentence_split = input_sentence.split() + ['<EOS>']
    output_words_split = output_words

    attentions_np = attentions.numpy()[:, :len(input_sentence_split)]
    ax = fig.add_subplot(1, 1, 1)
    im = ax.matshow(attentions_np, cmap='gray')

    ax.set_xticks(range(len(input_sentence_split)))
    ax.set_yticks(range(len(output_words_split)))

    ax.set_xticklabels(input_sentence_split, rotation=90)
    ax.set_yticklabels(output_words_split)

    plt.tight_layout()
    plt.savefig('plots/'+input_sentence+'.png')


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab, beam_search):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab, beam_search)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


def create_mini_batches(train_pairs, batch_size, src_vocab, tgt_vocab):
    input_tensors, target_tensors = [], []
    for pair in range(batch_size):
        input_tensor, target_tensor = tensors_from_pair(
            src_vocab, tgt_vocab, random.choice(train_pairs))
        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)
    input_tensors = nn.utils.rnn.pad_sequence(input_tensors)
    target_tensors = nn.utils.rnn.pad_sequence(target_tensors)
    return input_tensors.to(device), target_tensors.to(device)

######################################################################


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=100, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=float,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by  target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    ap.add_argument('--batch_size', default=4, type=int,
                    help='training batch size')
    ap.add_argument('--beam_search', dest='beam_search', action='store_true', help='use beam search')
    ap.add_argument('--no-beam_search', dest='beam_search', action='store_false', help='no beam search')
    ap.set_defaults(beam_search=False)

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(
        args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    # .parameters() returns generator
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    for iter_num in tqdm(range(args.n_iters)):
        # mini batching
        input_tensor, target_tensor = batches = create_mini_batches(
            train_pairs, args.batch_size, src_vocab, tgt_vocab)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion, args.batch_size)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0 and iter_num != 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(
                encoder, decoder, dev_pairs, src_vocab, tgt_vocab, args.beam_search, n=2)
            translated_sentences = translate_sentences(
                encoder, decoder, dev_pairs, src_vocab, tgt_vocab, args.beam_search)
            references = [[clean(pair[1]).split(), ]
                          for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(
        encoder, decoder, test_pairs, src_vocab, tgt_vocab, args.beam_search)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention(
        "on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab, args.beam_search)
    translate_and_show_attention(
        "j en suis contente .", encoder, decoder, src_vocab, tgt_vocab, args.beam_search)
    translate_and_show_attention(
        "vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab, args.beam_search)
    translate_and_show_attention(
        "c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab, args.beam_search)


if __name__ == '__main__':
    main()
