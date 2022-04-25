# Unsupervised Text Segmentaion and Tokenization

- Original task singnet/language-learning#255

## Tasks

- see if not only "increase of freedom on short-grams" is token-breaking, but also "decrease of freedom on long-grams"!? 
- use p+ and p- to generate dp+, dp-, ddp+, ddp- and tokenize based on EITHER of + and - as in case of ddf+, ddf- 
  - also try sums (|) and productions (&) across p+ and p- metrics with different N=[1..7] and directions +/-
      - https://github.com/aigents/pygents/blob/main/notebooks/nlp/TokenizerTest.ipynb
  - ON HOLD, no luck so far
- evaluate all metrics based on same corpus and "referenced" sentence, see sources of errors
- beat unsupervized tokenizer (UT) SOTA with semi-supervised tokenizer (SST) 
  - implement semi-supervised tokenizer, trained on word/character corpus (SST))
  - pre-train freq model for Tokenization on corpus, including A) words B) individual delimiters, C) generated numbers, D) 
    - tokenize based on true lexicon.txt ("curriculum learning" concept), count frequencies of non-present words, see what to do next
- when counting smaller ngrams based on p+/p-, denominate them for being part of larger ngrams?  
  - "inhibit frequencies" (or rather ""boost) from higher-order to lower-order?
  - https://github.com/aigents/pygents/blob/main/notebooks/nlp/TokenMining.ipynb
- Explore "surprizeness" measure to split as extension to "freedom"/"uncertainty"!?
- further token/ngram graph analysis and scenario mining for tokenization and morphology extending to sentence segmentation  
    - tokenize by clustering words in the sentence 
      - by gram counts - using MUTUAL INFORMATION!!! (does not work? double-check)
      - merge tokens in a way minimizing "freedom"/"uncertainty" (maximaly certain tree or MCT)
- model graph analysis with relationships
    - prev/next (sequential)
    - part/whole (morphology)
    - intention/extension (class hierarchy - vowels and consonants, suffixes and prefixe)
- corpora stats review
- consider other metrics

## Problems

- how to split endings quotes delimiters away from regular words, keeping the slashes, points and periods being parts of websites and numbers as parrt of tokens!?
- unsupervised decapitalization/capitalization?
- how to decode special chars like '\u200b' from input corpus data (other than just ignoring like we do now)

## Results

- Trained N-gram models with N=7 on different corpora
  - https://github.com/aigents/pygents/blob/main/notebooks/nlp/Tokenizer-Corpora.ipynb
  - Brown (B) - 6M
  - Gutenberg Children (GC) - 29M
  - Gutenberg Adullt (GA) - 140M
  - Social Media (SM) - 65M
- Explored frequencies on SM corpus
  - https://github.com/aigents/pygents/blob/main/notebooks/nlp/Tokenizer.ipynb
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
- Explored models based on different metrics according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2655800/ using SM corpus only
  - Conditional probabilities on N-to-N+1-gram transitions forward p+ and backward p- 
    - https://github.com/aigents/pygents/blob/main/notebooks/nlp/Tokenizer.ipynb
    - appear correlated with spaces and morphology (both!)
    - also have sums (|) and productions (&) across p+ and p- metrics with different N=[1..7] and directions +/-
  - Transitional "freedom" (uncertainty) forward p+ and backward p- (on gram-to-char and gram-to-gram basis for different N-s)
    - https://github.com/aigents/pygents/blob/main/notebooks/nlp/Tokenizer.ipynb
    - https://github.com/aigents/pygents/blob/main/notebooks/nlp/TokenizerTest.ipynb
    - appear more impressively connected with punctuation than p+ or p- 
    - also have sums (|) and productions (&) across f+ and f- metrics with different N=[1..7] and directions +/- - all appear more impressive than based on p+ and p-
    - also have deviations ddf+ and ddf- capped above zero - appear even more impressively connected with punctuation, so used in tokenizatioon  
- Explored MI using SM corpus, applied to bigram according to https://arxiv.org/pdf/cmp-lg/9805009.pdf (page 40)
  - https://github.com/aigents/pygents/blob/main/notebooks/nlp/Tokenizer.ipynb (see "counts2mis")
  - Hoping to cluster tokens based on pointise mutual information (PMI) did not lead to any promising results  
- Tried extended "generated quoted words" lexicon to improves the situation with F1 (detach framing double quotes " from the words)
  - Does not help to "resolve" doublequotes, detaching them away from words, the Brown corpus has enough quotes connected with word letters, but since the spaces are adjacent to many other punctuations, they have f+/f- much higher than doublequotes, so can not be resolved by threshold compared to word letters.
- Tried Brown (B) Gutenberg Children (GC) and Gutenberg Adult corpora to train models based on ddf+ or ddf- metrics (top F1 on tokens with no spaces) tested on B corpus https://github.com/aigents/pygents/blob/main/notebooks/nlp/TokenizerTest-Runs.ipynb
  - B => F1=0.91 (n=[1,2], t=0.4) - the best (most errors are caused with unability to detach framing double quotes " from the words)!!!
  - GC, GA, GC+GA => F1=0.78 (n=[1], t=0.4-0.8)
  - B+GC+GA => F1=0.91 (n=[1,2], t=0.4) - same as on B!
  - SM => F1=0.78 (n=[1], t=0.2-0.8)
  - B+GC+GA+SM => F1=0.78 (n=[1], t=0.2-0.8) - same as on SM!
- Improved the "freedom" models removing the low-frequency "tails" for each of the corpora 
  - F1=0.99 on Brown (and Brown + Gutenberg Children+Adult) with Brown 10 lines test set
      - https://github.com/aigents/pygents/blob/main/notebooks/nlp/TokenizerTest.ipynb
      - https://github.com/aigents/pygents/blob/main/notebooks/nlp/TokenizerTest-Runs.ipynb
  - F1=0.96 on Brown (and Brown + Gutenberg Children+Adult) with Brown 100 lines test set 
      - https://github.com/aigents/pygents/blob/main/notebooks/nlp/TokenizerTest-Runs-100.ipynb
- explored freedom-based models with all possible combinations of grams 1-7 for better F1 with larger test set of 100 lines from B
    - https://github.com/aigents/pygents/blob/main/notebooks/nlp/TokenizerTest-Runs-100.ipynb
    - larger corpus does not make better, the best is the smallest B, adding GC+GA to it does not improve (F1=0.96), adding SM to it makes it a bit worse (F1=0.93)


## References

### Corpora

- Multi-Lingual News from Common Crawl
    - https://data.statmt.org/
        - https://data.statmt.org/news-crawl/README
    - https://magichub.com/
        - https://magichub.com/datasets/chinese-english-parallel-corpus-finance/
    - https://www.amazon.science/blog/amazon-releases-51-language-dataset-for-language-understanding
- English 
    - Brown Corpus
        - http://www.sls.hawaii.edu/bley-vroman/brown_nolines.txt
    - Gutenberg Corpus
        - https://www.gutenberg.org/
- Russian
    - Inventory
        - https://nlpub.ru/%D0%A0%D0%B5%D1%81%D1%83%D1%80%D1%81%D1%8B
        - https://github.com/natasha/corus#usage
    - RusAge - text books
        - https://www.kaggle.com/datasets/oldaandozerskaya/fiction-corpus-for-agebased-text-classification
    - Twitter, need to extract froom SQL
        - http://study.mokoron.com/
    - Wiki, need to extract from XML
        - https://linguatools.org/tools/corpora/wikipedia-monolingual-corpora/
- Chinese (Simplified/Mandarin)
    - Lexicon
        - http://www.chineselexicaldatabase.com/download.php
    - CLUE
        - https://github.com/brightmart/nlp_chinese_corpus

### Tokenization

- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2655800/
- https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=6983&context=etd
- https://lena-voita.github.io/nlp_course/language_modeling.html
- https://en.wikipedia.org/wiki/Perplexity
- https://github.com/singnet/language-learning/issues/255
- https://medium.com/mlearning-ai/word-embeddings-wordpiece-and-language-agnostic-bert-labse-98c7626878c7

- https://github.com/natasha/razdel - razdel tries to mimic segmentation of these 4 datasets: SynTagRus, OpenCorpora, GICRYA and RNC.
- https://www.kaggle.com/c/text-normalization-challenge-english-language
- https://www.kaggle.com/c/text-normalization-challenge-russian-language
