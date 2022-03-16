# pygents
Machine Learning experiments in Python for Aigents project

## Setting up Jupyter on remote Ubuntu server in the cloud and run it locally

### Do this on remote server in the cloud:

1. Make sure you have Python3 pip and vitualenv installed, see https://1cloud.ru/help/linux/ustanovka-jupyter-notebook-na-ubuntu-18-04 (do this only once)
1. git clone https://gitlab.com/aigents/pygents.git (do this only once)
1. cd pygents
1. virtualenv env (do this only once)
1. . env/bin/activate
1. pip install jupyter (do this only once)
1. sudo iptables -A INPUT -p tcp --dport 8888 -j ACCEPT
1. jupyter notebook --no-browser

### Do this on your local machine:

1. `ssh -i <yourkey>.pem -N -f -L localhost:9999:localhost:8888 <yourusername>@<yourhost>` (do this in terminal)
1. http://localhost:9999/ (access this in the browser)

## References
https://hostadvice.com/how-to/how-to-install-jupyter-notebook-on-ubuntu-18-04-vps-or-dedicated-server/
https://www.datacamp.com/community/tutorials/installing-jupyter-notebook

