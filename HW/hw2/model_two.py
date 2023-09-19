#!usr/bin/env python
import optparse
import sys
from collections import defaultdict
import numpy as np

def train_model_two(bitext):
    max_f = max([len(f_sent) for f_sent, e_sent in bitext])
    max_e = max([len(e_sent) for f_sent, e_sent in bitext])

    t_probs = {}
    for f_sent, e_sent in bitext:
        for f_i in set(f_sent):
            t_probs[f_i] = defaultdict(lambda : 1/len(bitext))

    q = {}
    for i in range(max_f):
        for j in range(max_e):
            q[(i,j)] = defaultdict(lambda : 1/len(bitext))
    
    for i in range(5):
        e_align_any = defaultdict(float)
        fe_count = defaultdict(float)
        q_count = {}
        q_total = {}
        for (f_sent, e_sent) in bitext:
            for i in range(max_f):
                for j in range(max_e):
                    q_count[(i,j)] = defaultdict(float)
                    q_total[j] = defaultdict(float)

        norm = defaultdict(float)
        for (f_sent, e_sent) in bitext:
            for (j, e_j) in enumerate(e_sent):
                norm[j] = 0
                for (i, f_i) in enumerate(f_sent):
                    norm[j] += t_probs[f_i][e_j] * q[(i,j)][(len(f_sent) - 1, len(e_sent)  - 1)]
        
        for (f_sent, e_sent) in bitext:
            for (j, e_j) in enumerate(e_sent):
                for (i, f_i) in enumerate(f_sent):
                    c = t_probs[f_i][e_j] * q[(i,j)][(len(f_sent) - 1, len(e_sent)  - 1)] / norm[j]
                    fe_count[(f_i,e_j)] += c
                    e_align_any[f_i] += c
                    q_count[(i,j)][(len(f_sent) - 1, len(e_sent)  - 1)] += c
                    q_total[j][(len(e_sent) - 1, len(f_sent)  - 1)] += c
    

        t_probs = {}
        for f_sent, e_sent in bitext:
            for f_i in set(f_sent):
                t_probs[f_i] = defaultdict(float)
        
        for (f_sent, e_sent) in bitext:
            for (i, f_i) in enumerate(f_sent):
                for (j, e_j) in enumerate(e_sent):
                        t_probs = fe_count[(i,j)] / e_align_any[f_i]

        q = {}
        for i in range(max_f):
            for j in range(max_e):
                q[(i,j)] = defaultdict(float)

        for i in range(max_f):
            for j in range(max_e):
                for i_i in range(max_f):
                    for j_j in range(max_e):
                        q[(i,j)][(i_i,j_j)] = q_count[(i,j)][(i_i,j_j)] / q_total[j][(j_j,i_i)]

    return t_probs, q

# Read in command line arguments
optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with HMM...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]

t_probs = train_model_two(bitext)

# Alignment
for (f, e) in bitext:
    for (i, f_i) in enumerate(f): 
        best_prob = 0
        best_j = 0
        for (j, e_j) in enumerate(e):
            if t_probs[f_i][e_j] > best_prob:
                best_prob = t_probs[f_i][e_j]
                best_j = j
        sys.stdout.write("%i-%i " % (i,best_j))
    sys.stdout.write("\n")