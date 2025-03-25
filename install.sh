sudo apt update -y
sudo apt-get update -y
sudo apt-get install tmux htop nginx zip -y
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

git clone git@github.com:FrankCCCCC/diffusers.git
git fetch -a
git checkout -b my remotes/origin/my
cd diffusers
pip install .
cd ..

sudo su - root -c "curl -fsSL https://code-server.dev/install.sh | sh"

tmux new-session -d -s train
tmux new-session -d -s monitor "nvidia-smi -l 1"
tmux new-session -d -s code-server "code-server --bind-addr=0.0.0.0:5001 --auth none --cert=/home/u2941379/ssl/nginx.crt --cert-key=/home/u2941379/ssl/nginx.key"
tmux new-session -d -s tfboard "tensorboard --logdir ."
nohup sudo -i -H -u u2941379 /run_jupyter.sh --port=8888 --notebook-dir=/work/u2941379 --config=/etc/jupyter/jupyter_notebook_config.py &

echo $HF_DATASETS_CACHE