# Contents
1. _requirements.txt_ - list of dependencies to be installed under _Python 3.11.11_ environment, such as using _venv_ and _pip_
2. _a api.py, learn.py, plot.py, text.py, util.py_ - program modules used by the following notebooks 
    1. _a_api.py_ - model processing code
    2. _plot.py_ - plotting utilities
    3. _text.py_ - text utilitis
    4. _util.py_ - randeom utilities
3. Notebooks to be run in the following order:
    1. _overfitting_combined*.ipynb_ - overfitting experiment with no punctuation removed (initial study)
    2. _overfitting_combined*cleaned.ipynb_ - overfitting experiment with punctuation removed (cleaner and final results)
    3. _split_combined*.ipynb_ - split cross-validation experiment with no punctuation removed (initial study)
    4. _split_combined*cleaned.ipynb_ - split cross-validation experiment with punctuation removed (cleaner and final results)
    5. _comparing_llms.ipynb_ - Jupyter notebook for detection experiment using LLMs, saving the intermediate results to file _llm_evaluation_results_ using _pickle_ format and module
    6. _comparing_models.ipynb_ - Jupyter notebook for detection experiment comparing ours models against baseline and LLMs
4. Data files:
    1. ./data/corpora/English/distortions/halilbabacan - "binary" dataset
6. Model files
    1. _../../data/models/distortions/ours_ - baseline model created based on earlier work (Bollen et. al., 2021; Raheman et. al., 2022; Arinicheva & Kolonin, 2025)
    2. _../../data/models/distortions/overfitting_combined_ - interpretable models created in the course of our study during the "overfitting" experiments
    3. _../../data/models/distortions/split_combined_ - interpretable models created in the course of our study during the "cross-validation" experiments
