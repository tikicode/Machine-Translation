#!usr/bin/env python
import optparse
import sys
from collections import defaultdict
import numpy as np

def iteration(previous_probs, bitext):
    probs = previous_probs
    count_e = defaultdict(int)
    fe_count = defaultdict(int)
    for f_sent, e_sent in bitext:
        for f_i in set(f_sent):
            for e_j in set(e_sent):
                fe_count[(f_i,e_j)] = 0
        for e_j in set(e_sent):
            count_e[e_j] = 0

    for f_sent, e_sent in bitext:
        for f_i in f_sent:
            norm_z = 0
            for e_j in e_sent:
                norm_z += probs[(f_i,e_j)]
            for e_j in e_sent:
                c = probs[(f_i,e_j)] / norm_z
                fe_count[(f_i,e_j)] += c
                count_e[e_j] += c
        for f_i in set(f_sent):
            for e_j in set(e_sent):
                probs[(f_i,e_j)] = fe_count[(f_i,e_j)] / count_e[e_j]
    return probs

def train(probs, bitext):
    iters = 0
    converged = False
    count_e = {}
    prev_probs = probs
    while not converged:
        iters += 1
        probs = iteration(prev_probs, bitext)
        converged = check_converged(prev_probs, probs)
        prev_probs = probs
    return probs, iters

def check_converged(prev_probs, probs):
    perp1 = 0
    perp2 = 0
    for x,y in zip(prev_probs.values(), probs.values()):
        perp1 -= np.log2(x)
        perp2 -= np.log2(y)
    return abs(perp1 - perp2) < 3


# Read in command line arguments
optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with IBM Model 1...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]
# Create a matrix that matches all French translations to their 
# English counterparts and assign a default probability 
init = 0
for f_sent, e_sent in bitext:
    init += len(f_sent)
probs = defaultdict(float)
for f_sent, e_sent in bitext:
    for f_i in set(f_sent):
        for e_j in set(e_sent):
            probs[(f_i,e_j)] = init

# Run EM algorithm
probs, iters = train(probs, bitext)

# alignment
for (f, e) in bitext:
    for (i, f_i) in enumerate(f): 
        best_prob = 0
        best_j = 0
        for (j, e_j) in enumerate(e):
            if probs[(f_i,e_j)] > best_prob:
                best_prob = probs[(f_i,e_j)]
                best_j = e_j
        sys.stdout.write("%i-%i " % (i,j))
sys.stdout.write("\n")
            
