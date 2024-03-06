# Sentence/phrase/fragment text segmentation

## The plan

- Get train corpus - Brown
  - eliminate line feeds, tabs and repeating spaces
    - en
    - zh
    - ru
  - sentencenize by periods (for curriculum learning!!!???)
    - en
    - zh
    - ru
- Create test corpus - magicdata 100
  - sentencenize by periods
    - en
    - zh
    - ru
  - including periods/punctuation
  - excluding periods/punctuation
- tokenize both coprpora
  - including periods/punctuation
  - excluding periods/punctuation
- Count graphs on test corpus
  - including periods/punctuation
  - excluding periods/punctuation
- Build profiles for conditional priobability (CP) and transition freedom (TF) 
  - including periods/punctuation
  - excluding periods/punctuation
- sentencenize 
  - including periods/punctuation
  - excluding periods/punctuation
- conclude for best segmentation
- build token categories and compare "best fitting"
  - space metrics: CP/TF, pruning threshold, N-gram
  - clustering method - agglomerative on jaccard/cosine 
  - clustering metric - silhouette
  - ground truth - overlap with word categories (eg. parts of speech, POS, category of word, COW)
  - learning curve
    - batch
    - curriculum (incremental with increasing sentence length!?)
    - incremental reducing vector space to word categories instead of words
- conclude for best categorization
- 

    
## References
- https://www.researchgate.net/publication/236687462_Fuzzy_Formal_Concept_Analysis_and_Algorithm
