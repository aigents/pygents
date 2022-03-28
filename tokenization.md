# TODO

## First

- Diary PROGREESS
  - Tokenizer.ipynb
  - TokenizerTest.ipynb
- if "generated quoted words" lexicon improves the situation!? 
- use p+ and p- to generate dp+, dp-, ddp+, ddp- and tokeniize based on EITHER of + and - as in case of ddf+, ddf- 
  - also try sums (|) and productions (&) across p+ and p- metrics with different N=[1..7] and directions +/-
- tokenize based on true lexicon.txt, count frequencies of non-present words, see what to do next
- when counting smaller ngrams, denominate them for being part of larger ngrams?  
- Corpora stats review
- Explore "surprizeness" measure to split!? 
- merge tokens in a way minimizing freedom 
- consider other metrics
- find best parameters tokenizer parameters and find SOTA 
- implement semi-supervised tokenizer, trained on word/character corpus (SST))
- beat UT SOTA with SUT 
- token/ngram graph analysis and scenario mining for tokenization and morphology.  

## Next

- pre-train freq model for Tokenization on corpus, including A) words B) individual delimiters, C) generated numbers, D) generated dates
- tokenize by clustering words in the sentence by gram counts - using MUTUAL INFORMATION!!!
- how to split endings delimiters away from words!?
- inhibit frequencies from higher-order to lower-order?
- decapitalization?
- decode '\u200b'


# DONE

- Trained 7 on different corpora https://github.com/aigents/pygents/blob/main/notebooks/nlp/Tokenizer-Corpora.ipynb
  - Brown (B) - 6M
  - Gutenberg Children (GC) - 29M
  - Gutenberg Adullt (GA) - 140M
  - Social Media (SM) - 65M
- Explored frequencies on SM corpus https://github.com/aigents/pygents/blob/main/notebooks/nlp/Tokenizer.ipynb
  - All N-grams (n=[1..7])
    - top 1-gram ' ' - gets outstanding score, next are 't' and 'e' (from 'the')
    - top 2-gram 'in' - after ' t' and 'e ' (from 'the')
    - top 3-grams 'the' and 'ing' - along with ' th' and 'he ' (from 'the')
    - top 4-grams ' the', 'the ', 'ing ', ' to ', 'http'
    - top 5-grams ' the', 'https'
    - top 6/7-grams 'https:', 'https:/'
  - Token N-grams based on space-tokenizer
    - top 'the', 'to', 'and', 'a', 'of', ...
  - Logarithmic distributions still apppear Zipfian 
  - Conditional probabilities on N-to-N+1-gram transitions forward p+ and backward p- 
    - appear correlated with spaces and morphology (both!)
    - also have sums (|) and productions (&) across p+ and p- metrics with different N=[1..7] and directions +/-
    - TODO
- Explored word counts
  - TODO
- Explored models based on different metrics according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2655800/ https://github.com/aigents/pygents/blob/main/notebooks/nlp/TokenizerTest.ipynb
  - "Freedom" TODO...
- Trying Brown (B) Gutenberg Children (GC) and Gutenberg Adult corpora with ddf+ or ddf- metrics (top F1 on tokens with no spaces)  https://github.com/aigents/pygents/blob/main/notebooks/nlp/TokenizerTest-Runs.ipynb
  - B => F1=0.91 (n=[1,2], t=0.4) - the best!!!
  - GC, GA, GC+GA => F1=0.78 (n=[1], t=0.4-0.8)
  - B+GC+GA => F1=0.91 (n=[1,2], t=0.4) - same as on B!
  - SM => F1=0.78 (n=[1], t=0.2-0.8)
  - B+GC+GA+SM => F1=0.78 (n=[1], t=0.2-0.8) - same as on SM!


# References

## Tokenization

- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2655800/
- https://lena-voita.github.io/nlp_course/language_modeling.html
- https://en.wikipedia.org/wiki/Perplexity
- https://github.com/singnet/language-learning/issues/255
- https://medium.com/mlearning-ai/word-embeddings-wordpiece-and-language-agnostic-bert-labse-98c7626878c7

- https://github.com/natasha/razdel - razdel tries to mimic segmentation of these 4 datasets: SynTagRus, OpenCorpora, GICRYA and RNC.
- https://www.kaggle.com/c/text-normalization-challenge-english-language
- https://www.kaggle.com/c/text-normalization-challenge-russian-language
