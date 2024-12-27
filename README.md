# pygents

Machine Learning experiments in Python for Aigents project - at the moment, primarily fundamental NLP research grounding further production development in https://github.com/aigents/aigents-java-nlp under the umbrella of Interpretable Natural Language Processing paradigm, see https://aigents.github.io/inlp/

## Setting up Jupyter on remote Ubuntu server in the cloud and run it locally

### Do this on remote server in the cloud (or if you want to access Jupyter locally):

#### Using Python 3.11.9

1. Make sure you have Python3 pip and vitualenv installed, see https://1cloud.ru/help/linux/ustanovka-jupyter-notebook-na-ubuntu-18-04 *(do this only once)*
1. git clone https://gitlab.com/aigents/pygents.git *(do this only once)*
1. cd pygents
1. virtualenv env *or* python -m venv env *(do this only once)*
1. . env/bin/activate *(. ./env/Scripts/activate [- if under Windows](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html))*
1. pip install -r pygents/requirements.txt *(do this only once)*
1. sudo iptables -A INPUT -p tcp --dport 8887 -j ACCEPT *(do this only once, need only to deploy on cloud server)*
1. jupyter notebook --no-browser --port=8887 *(if you are doing this on your local machine, just access http://localhost:8887/ locally in the browser)*

### Do this on your local machine (if you want to access remote Jupyter):

1. `ssh -i <yourkey>.pem -N -f -L localhost:9999:localhost:8887 <yourusername>@<yourhost>` (do this in terminal)
1. http://localhost:9999/ (access this in the browser)

## Matters of Study

### Unsupervised Text Segmentaion and Tokenization 

1. Original task https://github.com/singnet/language-learning/issues/255
1. Current progress [notebooks/nlp/tokenization/README.md](./notebooks/nlp/tokenization/README.md)

### Unsupervised Language Grammar and Ontology Learning 

1. Original task https://github.com/singnet/language-learning/issues/220
2. Curent progress http://langlearn.singularitynet.io/

### Unsupervised Text Generation 

1. See https://aigents.github.io/inlp/2021/

### Unsupervised Question Answering  and Conversational Intelligence

1. See https://aigents.github.io/inlp/2021/

## References
https://hostadvice.com/how-to/how-to-install-jupyter-notebook-on-ubuntu-18-04-vps-or-dedicated-server/
https://www.datacamp.com/community/tutorials/installing-jupyter-notebook
