# Data
The following data files were used in the course of this study or were generated based on its results:

1. `./data/corpora/English/distortions/halilbabacan` — "Binary" dataset, according to Babacan (2023) (https://huggingface.co/datasets/halilbabacan/autotrain-data-cognitive\_distortions), Babacan et al. (2023) (https://ssrn.com/abstract=4582307), and Babacan et al. (2025) (10.35234/fumbd.1469178).
2. `./data/models/distortions/ours` — baseline interpretable model created based on earlier work: Bollen et al., 2021 (10.1073/pnas.2102061118); Raheman et al., 2022 (10.48550/arXiv.2204.10185); Arinicheva & Kolonin, 2025 (10.1007/978-3-031-80463-2\_31).
3. `./data/models/distortions/overfitting_combined` — interpretable models created during the "overfitting" experiments.
4. `./data/models/distortions/split_combined` — interpretable models created during the "cross-validation" experiments.

# Code
The following code is supplied in the presented repository and can be used to reproduce the results of our study and extend the experiments. All experiments were performed using Python 3.11.11, with external dependencies specified in the `requirements.txt` file, including version numbers. The code located in the `./papers/distortions_binary_2025/` folder should be used in the following order to reproduce the results:

1. `requirements.txt` — list of dependencies to be installed under Python 3.11.11 environment, such as using `venv` and `pip`.
2. `a_api.py, learn.py, plot.py, text.py, util.py` — program modules used by the following notebooks:
   - `a_api.py` — model processing code
   - `plot.py` — plotting utilities
   - `text.py` — text utilities
   - `util.py` — random utilities
3. Notebooks to be run in the following order:
   1. `overfitting_combined*.ipynb` — overfitting experiment with no punctuation removed (initial study)
   2. `overfitting_combined*cleaned.ipynb` — overfitting experiment with punctuation removed (cleaner and final results)
   3. `split_combined*.ipynb` — split cross-validation experiment with no punctuation removed (initial study)
   4. `split_combined*cleaned.ipynb` — split cross-validation experiment with punctuation removed (cleaner and final results)
   5. `comparing_llms.ipynb` - Jupyter notebook for detection experiment using LLMs, saving the intermediate results to file `llm_evaluation_results` using `pickle` format and module
    6. `comparing_models.ipynb` - Jupyter notebook for detection experiment comparing ours models against baseline and LLMs