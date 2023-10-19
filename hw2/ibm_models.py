#!/usr/bin/env python
import optparse
import sys
import string
import nltk
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
nltk.download("punkt")
nltk.download("wordnet")

e_stemmer = PorterStemmer()
f_stemmer = nltk.stem.SnowballStemmer("french")

def train_model_one(bitext, iters):
    probs = defaultdict(lambda : 1/len(bitext))
    count_e = defaultdict(float)
    fe_count = defaultdict(float)

    for i in range(iters):
        for f_sent, e_sent in bitext:
            for f_i in f_sent:
                norm_z = 0
                for e_j in e_sent:
                    norm_z += probs[(f_i,e_j)]
                for e_j in e_sent:
                    c = probs[(f_i,e_j)] / norm_z # Expected Count
                    fe_count[(f_i,e_j)] += c
                    count_e[e_j] += c
        for fe in fe_count:
            probs[(fe[0],fe[1])] = fe_count[fe] / count_e[fe[1]] # Normalize
    return probs


def train_model_two(bitext, iters):
    e_total = defaultdict(lambda: 0.0)
    t_probs = train_model_one(bitext, 5)

    en_vocab = []
    fr_vocab = []
    for (f_sent, e_sent) in bitext:
        en_vocab.extend(e_sent)
        fr_vocab.extend(f_sent)

    en_vocab = set(en_vocab)
    fr_vocab = set(fr_vocab)

    max_e = max([len(e_sent) for _, e_sent in bitext])
    max_f = max([len(f_sent) for f_sent, _ in bitext])

    q = np.zeros((max_f+1, max_e+1, max_f+1, max_e+1), dtype=float)
    for (f_sent, e_sent) in bitext:
        f_len = len(f_sent)
        e_len = len(e_sent)
        for i in range(f_len):
            for j in range(e_len):
                q[i,j,f_len,e_len] = 1/(f_len + 1)
 
    for i in range(iters):
        f_total = defaultdict(float)
        fe_count = defaultdict(float)
        q_count = np.zeros((max_f+1, max_e+1, max_f+1, max_e+1), dtype=float)
        q_total = np.zeros((max_e+1, max_f+1, max_e+1), dtype=float)
        for (f_sent, e_sent) in bitext:
            f_len = len(f_sent)
            e_len = len(e_sent)
            for (j, e_j) in enumerate(e_sent):
                for (i, f_i) in enumerate(f_sent):
                    e_total[e_j] += t_probs[(f_i,e_j)] * q[i,j,f_len,e_len] # Normalize
            
            for (j, e_j) in enumerate(e_sent):
                for (i, f_i) in enumerate(f_sent):
                    c = t_probs[(f_i,e_j)] * q[i,j,f_len,e_len] / e_total[e_j]
                    q_count[i,j,f_len,e_len] += c
                    q_total[j,f_len,e_len] += c
                    fe_count[(f_i,e_j)] += c
                    f_total[f_i] += c

        q = defaultdict(lambda: 0.0)
        t_probs = defaultdict(lambda: 0.0)

        for f in fr_vocab:
            for e in en_vocab:
                t_probs[(f,e)] = fe_count[(f,e)] / f_total[f]
                    
        for (f_sent, e_sent) in bitext:
            f_len = len(f_sent)
            e_len = len(e_sent)
            for (j, e_j) in enumerate(e_sent):
                for (i, f_i) in enumerate(f_sent):
                    q[i,j,f_len,e_len] = q_count[i,j,f_len,e_len] / q_total[j,f_len,e_len]

    return t_probs, q

if __name__ == "__main__":
    # Read in command line arguments
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
    optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
    optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-m", "--model", dest="model", default="two", help="IBM Model to run (default = two)")
    optparser.add_option("-s", "--stemming", action="store_true", dest="stem", default=False)

    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (opts.train, opts.french)
    e_data = "%s.%s" % (opts.train, opts.english)

    sys.stderr.write("Training with IBM Model 1...")

    bitext = []
    if opts.stem:
        with open(f_data) as f_file, open(e_data) as e_file:
            for f_sentence, e_sentence in zip(f_file, e_file):
                f_sentence = f_sentence.strip()
                e_sentence = e_sentence.strip()
                f_tokens =f_sentence.lower().split()
                e_tokens= e_sentence.lower().split()
                f_tokens = [f_stemmer.stem(token) for token in f_tokens]
                e_tokens =  [e_stemmer.stem(token) for token in e_tokens]
                bitext.append([f_tokens, e_tokens])
    else:
        bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))]

    # Use bitext for training and alignment
    bitext = bitext[:opts.num_sents]

    probs = defaultdict(float)

    if opts.model == "one":
        probs = train_model_one(bitext, 10)
    elif opts.model == "two":
        probs, q = train_model_two(bitext, 5)
    else:
        probs = train_model_two(bitext, 5)

    # Alignment
    for (f, e) in bitext:
        for (i, f_i) in enumerate(f): 
            best_prob = 0
            best_j = 0
            for (j, e_j) in enumerate(e):
                if probs[(f_i,e_j)] > best_prob:
                    best_prob = probs[(f_i,e_j)]
                    best_j = j
            sys.stdout.write("%i-%i " % (i,best_j))
        sys.stdout.write("\n")
