# Contents
1. _requirements.txt_ - list of dependencies to be installed under _Python 3.11.11_ environment, such as using _venv_ and _pip_
2. _a api.py, learn.py, plot.py, text.py, util.py_ - program modules used by the following notebooks 
    1. a_api.py - model processing code
    2. plot.py - plotting utilities
    3. text.py - text untilitis
    4. util.py - randeom utilities
3. Notebooks to be run in the following order:
    1. _overfitting_combined*.ipynb_ - overfitting experiment with no punctuation removed (initial study)
    2. _overfitting_combined*cleaned.ipynb_ - overfitting experiment with punctuation removed (cleaner and final results)
    3. _split_combined*.ipynb_ - split cross-validation experiment with no punctuation removed (initial study)
    4. _split_combined*cleaned.ipynb_ - split cross-validation experiment with punctuation removed (cleaner and final results)
    5. _comparing_llms.ipynb_ - Jupyter notebook for detection experiment using LLMs, saving the intermediate results to file _llm_evaluation_results_ using _pickle_ format and module
    6. _comparing_models.ipynb_ - Jupyter notebook for detection experiment comparing ours models against baseline and LLMs
4. Source files
5. Model files
    1. ../../data/models/distortions
