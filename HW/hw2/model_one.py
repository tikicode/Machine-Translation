#!usr/bin/env python
import optparse
import sys
from collections import defaultdict
import numpy as np

def iteration(previous_probs, bitext):
    probs = previous_probs
    count_e = defaultdict(float)
    fe_count = defaultdict(float)

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

def train_model_one(probs, bitext, iters):
    prev_probs = probs
    # converged = False
    # while not converged: 
    # Attemped to use convergence based on perplexity score
    # but results were worse
    for i in range(iters):
        probs = iteration(prev_probs, bitext)
        # converged = check_converged(prev_probs, probs)
        prev_probs = probs
    return probs

def check_converged(prev_probs, probs):
    # Check convergence by change in perplexity
    perp1 = 0
    perp2 = 0
    for x,y in zip(prev_probs.values(), probs.values()):
        if (not isinstance(x, float) and not isinstance(y, float)):
            for xs, ys in zip(x.values(), y.values()):
                perp1 -= np.log2(xs)
                perp2 -= np.log2(ys)
    return abs(perp1 - perp2) < 0.001

if __name__ == "__main__":
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
    probs = defaultdict(lambda : 1/len(bitext))

    # Run EM algorithmx
    probs = train_model_one(probs, bitext, 10)

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
