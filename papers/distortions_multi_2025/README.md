# Supplementary Code for paper "Interpretable Recognition of Cognitive Distortions in Natural Language Texts" 
1. _requirements.txt_ - list of dependencies to be installed under _Python 3.11.13_ environment, such as using _venv_ and _pip_
2. _a api.py, learn.py, plot.py, text.py, util.py_ - program modules used by the following notebooks 
    1. _api.py_ - model processing code
    2. _plot.py_ - plotting utilities
    3. _text.py_ - text processing utilitis
    4. _util.py_ - random utilities
    5. _recognition_evaluators.py_ - pipeline code for learning and recognition for hyper-parameter rearch across multiple data splits   
3. Notebooks for the models :
    1. _shreevastava.ipynb_ - experimental code for the first real field dataset (Shreevastava 2021) 
    2. _babacan.ipynb_ - experimental code for the second half-synthetic dataset (Babacan 2024)
6. Model files
    1. _../../data/models/distortions/ours_ - baseline model created based on earlier work (Bollen et. al., 2021; Raheman et. al., 2022; Arinicheva & Kolonin, 2025)
    2. _../../data/models/distortions/shreevastava2021_ - interpretable model created in the course of our study based on the first real field dataset (Shreevastava 2021), for three independent (overlapping) train splits (80% of the entire dataset) with inclusion threshold (IT) 0 based on "FCR" selection metric (SM)
