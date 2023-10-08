There are two python programs here (-h for usage):

- `decode` translates input sentences from French to English.
- `grade` computes the model score of a translated sentence.

These commands work in a pipeline. For example:

    > python decode | python compute-model-score

There is also a module:

- `model.py` implements very simple interfaces for language models
 and translation models, so you don't have to. 

You can finish the assignment without modifying this file at all. 


The `data` directory contains files derived from the Canadian Hansards,
originally aligned by Ulrich Germann:

- `input`: French sentences to translate.

- `tm`: a phrase-based translation model. Each line is in the form:

    French phrase ||| English phrase ||| log_10(translation_prob)

- `lm`: a trigram language model file in ARPA format.

    log_10(ngram_prob)   ngram   log_10(backoff_prob)

The language model and translation model are computed from the data 
in the align directory, using alignments from the Berkeley aligner.

> decode-ext is the best file which is future cost implementation with score of -1300 approx
> decode-beam-search is beam search implementation
> translations have the translated sentences from future cost which was runned using python decode-ext > translations

