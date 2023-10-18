There are four python programs here (`-h` for usage):

- `./align` aligns words using DICE score.

- `./model_one.py` aligns words using Expectation Maximization

- `./check-alignments` checks that the entire dataset is aligned, and
  that there are no out-of-bounds alignment points.

- `./score-alignments` computes alignment error rate.

The commands work in a pipeline. For instance:

   > ./align -t 0.9 -n 1000 | ./check | ./grade -n 5

   > ./ibm_models -n 1000 -m one -s | ./check | ./grade -n 5

In order to run the ibm models, please download the requirements 
using the command:

  > pip install -r requirements.txt

To run the a specific ibm model between one and three, please use
the `-m` argument followed by `one` or `two`. The rest 
of the command line arguments are exactly the same as those in 
`align`. Additionally, to include stemming, please run with the '-s'
option.

The `data` directory contains a fragment of the Canadian Hansards,
aligned by Ulrich Germann:

- `hansards.e` is the English side.

- `hansards.f` is the French side.

- `hansards.a` is the alignment of the first 37 sentences. The 
  notation i-j means the word as position i of the French is 
  aligned to the word at position j of the English. Notation 
  i?j means they are probably aligned. Positions are 0-indexed.
