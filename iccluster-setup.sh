#!/bin/bash

# Set exit script on error
set -o errexit

echo ""
echo "### 1. Install basics #######################################################################"
echo ""

# Install stuff
sudo apt-get update
apt-get install -y git screen vim htop gcc

# Config git
git config --global user.name "trouleau"
git config --global user.email "william.trouleau@iccluster.epfl.ch"

########## Add personal ssh public key to authorized_keys
echo ""
echo ""
echo "### Add personal ssh public key to authorized_keys"
echo "### ----------------------------------------------"
echo ""

mkdir ~/.ssh
echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDPqJjY1js7ubxTdPdk0PuGWjSGJyVnDx4c3nGOzwRfTSYp5bpWIAfHpD7DQKX8LCIwVQ4z3WoQrO7/BY3VQ328OvoL+BsTUL89G1HtgdAK0jG+8ICydFdbwc9Yj4taV3gbyTKur3W9sgnC5t0vGXwbUl0Q3Af30BKJKYNJzJ9elzUmIANl5TI4CeaBqqieB2sDu6WjHwmB+AUb/We1YIcUmPcRM3tmOOo+IQUSNUsaeW/2CDV7SH1S0IHUVyfMpaywLN8Vc41dYC4/z1TiuAo44tl8jpdapNTxelpARLfIx1dcpsfhPTP+zs9IBUY0Q/TSBV/tiACHEhh/etGoVKtJ trouleau@icsil1noteb194.epfl.ch' >> ~/.ssh/authorized_keys

########## Create a ssh key for the server
echo ""
echo ""
echo "### Create a ssh key for the server"
echo "### -------------------------------"
echo ""
ssh-keygen -t rsa -b 4096 -C "william.trouleau@iccluster.epfl.ch" -f ~/.ssh/id_rsa
echo "The ssh key of the server is:"
cat ~/.ssh/id_rsa.pub
echo ""
read -p "Add the key to git repo host. Then press enter to continue"

##########  Install python with miniconda
echo ""
echo ""
echo "### Install python with miniconda"
echo "### -----------------------------"
echo ""

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
echo ". /root/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
source /root/miniconda/etc/profile.d/conda.sh
rm ~/miniconda.sh
# Test install
echo "Check install:"
conda --version
echo "Install ok!"

########## Prepare project environment
echo ""
echo ""
echo ""
echo "### 2. Prepare project environment ##########################################################"
echo ""

# Clone project source repos
echo ""
echo ""
echo "### Clone project source repos"
echo "### --------------------------"
echo ""

cd /root/ && git clone git@github.com:trouleau/var-wold.git  # Add personal lib

# Make virtualenv
echo ""
echo ""
echo "### Make virtualenv `env`"
echo "### ---------------------"
echo ""

# Make virtualenv
conda create -y -n env python=3.7

# Install libs
cd /root/var-wold/ && pip install -e .  # Install internal lib
cd /root/var-wold/ && pip install -r requirements.txt
cd /root/var-wold/lib/granger-busca/ && pip install cython && pip install -e .  # Install gb

# Make output directory
mkdir /root/var-wold/output/

# Setup jupyter config
echo ""
echo ""
echo "### Make virtualenv `env`"
echo "### ---------------------"
echo ""

conda activate env \
    && jupyter notebook --generate-config \
    && echo "
c.NotebookApp.ip = '*'
c.NotebookApp.allow_root = True
c.NotebookApp.open_browser = False
c.NotebookApp.password = 'sha1:b66fca924c15:af4b00b51bc827689857e386d2304148029fa257'
c.NotebookApp.port = 2636" >> ~/.jupyter/jupyter_notebook_config.py

# Setup backup output to icsil1-access1
(crontab -l; echo "5,15,25,35,45,55 * * * * rsync -rav /root/var-wold/output trouleau@icsil1-access1.epfl.ch:/dfs/ephemeral/storage/trouleau/var-wold/ >> /root/cron.log 2>&1") | crontab -













# Clone project source repos
cd /root/workspace/ && git clone git@github.com:trouleau/var-wold.git  # Add personal lib

# Make virtualenv
conda create -n env python=3.7 && conda activate env

cd /root/workspace/var-wold/ && pip install -e .  # Install internal lib
cd /root/workspace/var-wold/ && pip install -r requirements.txt
cd /root/workspace/var-wold/lib/granger-busca/ && pip install cython && pip install -e .  # Install gb
