# Interpretable learning for detection of cognitive distortions from natural language texts

## Description  
This repository provides the data and code used in the study *Interpretable learning for detection of cognitive distortions from natural language texts*.  
It contains baseline and experimental models, corpora, and Jupyter notebooks required to reproduce the results.  

## Dataset Information  
The following datasets were used in the study or generated based on its results:  

1. `./data/corpora/English/distortions/halilbabacan` — “Binary” dataset, according to Babacan (2023) (https://huggingface.co/datasets/halilbabacan/autotrain-data-cognitive_distortions), Babacan et al. (2023) (https://ssrn.com/abstract=4582307), and Babacan et al. (2025) (10.35234/fumbd.1469178).  
2. `./data/models/distortions/ours` — baseline interpretable model created based on earlier work: Bollen et al., 2021 (10.1073/pnas.2102061118); Raheman et al., 2022 (10.48550/arXiv.2204.10185); Arinicheva & Kolonin, 2025 (10.1007/978-3-031-80463-2\_31).   
3. `./data/models/distortions/overfitting_combined` — interpretable models created during the first phase "overfitting" experiments on "combined" dataset.  
4. `./data/models/distortions/split_combined` — interpretable models created during the first phase "cross-validation" experiments on "combined" dataset.  
5. `./data/models/distortions/babacan_shreevastava_binary` — interpretable binary model created during the second phase on "combined" dataset.
6. `./data/models/distortions/babacan_shreevastava_multiclass` — interpretable multi-class model created during the second phase on "combined" dataset.
7. `./data/models/distortions/babacan_2023_binary` — interpretable binary model created during the second phase on Babakan (2023) dataset.
8. `./data/models/distortions/shreevastava_2021_binary` — interpretable binary model created during the second phase on Shreevastava (2021) dataset.
9. `./data/models/distortions/shreevastava_2021_multiclass` — interpretable multi-class model created during the second phase on Shreevastava (2021) dataset.

## Code Information  
The repository contains the following code to reproduce the study results.  
All experiments were performed using **Python 3.11.11** with dependencies specified in `requirements.txt` (including version numbers).

- `requirements.txt` — list of dependencies to be installed under Python 3.11.11 environment, such as using `venv` and `pip`. 
- Program modules:  
  - `a_api.py` — model processing code  
  - `learn.py` — learning functions 
  - `plot.py` — plotting utilities  
  - `text.py` — text utilities  
  - `util.py` — random utilities  

## Usage Instructions  
To reproduce the experiments, the notebooks located in `./papers/distortions_binary_2025/` should be run in the following order:  

1. First Phase, "Combined" dataset
  1. `overfitting_combined*.ipynb` — overfitting experiment without punctuation removed (initial study).  
  2. `overfitting_combined*cleaned.ipynb` — overfitting experiment with punctuation removed (cleaner and final results).  
  3. `split_combined*.ipynb` — split cross-validation experiment without punctuation removed (initial study).  
  4. `split_combined*cleaned.ipynb` — split cross-validation experiment with punctuation removed (cleaner and final results).  
  5. `comparing_llms.ipynb` — detection experiment using LLMs, saving intermediate results to `llm_evaluation_results` file in `pickle` format.  
  6. `comparing_models.ipynb` — comparison of our models against baselines and LLMs, testing on cross-validation test splits.
2. Second Phase
  1. "Combined" dataset
    1. `babacan_shreevastava_binary_view.ipynb` - "binary view" experiment, hyperparameter search, traning on 80%, validation on 10%.
    2. `babacan_shreevastava_multiclass_view.ipynb` - "multi-class view" experiment, hyperparameter search, traning on 80%, validation on 10%.
    3. `comparing_models_babacan_shreevastava.ipynb` - comparison of our models against baselines and LLMs, testing on 10%.
  2. "Binary" dataset (Babakan, 2023) 
    1. `babacan_binary_view.ipynb` - "binary view" experiment, hyperparameter search, traning on 80%, validation on 10%.
    2. `comparing_llms_babacan.ipynb` - detection experiment using LLMs, saving intermediate results to `llm_evaluation_results_babacan` file in `pickle` format. 
  3. "Multi-class" dataset (Shreevastava, 2021)
    1. `shreevastava_binary_view.ipynb` - "binary view" experiment, hyperparameter search, traning on 80%, validation on 10%.
    2. `shreevastava_multiclass_view.ipynb` - "multi-class view" experiment, hyperparameter search, traning on 80%, validation on 10%.
    3. `comparing_llms_shreevastava.ipynb` - detection experiment using LLMs, saving intermediate results to `llm_evaluation_results_shreevastava` file in `pickle` format.
    4. `comparing_models_shreevastava.ipynb` - comparison of our models against baselines and LLMs, testing on 10%.

## Requirements  
- Python 3.11.11  
- Dependencies listed in `requirements.txt`  

## Citation  
If you use these datasets or code, please cite the following works:  

- Babacan (2023): https://huggingface.co/datasets/halilbabacan/autotrain-data-cognitive_distortions  
- Babacan et al. (2023): https://ssrn.com/abstract=4582307  
- Babacan et al. (2025): 10.35234/fumbd.1469178  
- Bollen et al. (2021): 10.1073/pnas.2102061118  
- Raheman et al. (2022): 10.48550/arXiv.2204.10185  
- Arinicheva & Kolonin (2025): 10.1007/978-3-031-80463-2_31  

## License  
This project is licensed under the MIT License. See the LICENSE file for details.