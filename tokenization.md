# TODO

## First

- Diary
- if "generated quoted words" lexicon improves the situation!? 
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

- Metrics...
- "Freeedom" ...
- Trying Brown (B) Gutenberg Children (GC) and Gutenberg Adult corpora with ddf+ or ddf- metrics (top F1 on tokens woth no spaces)
-- B => F1=0.91 (n=[1,2], t=0.4)
-- GC, GA, GC+GA => F1=0.78 (n=[1], t=0.4-0.8)
-- B+GC+GA => F1=0.91 (n=[1,2], t=0.4)


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
