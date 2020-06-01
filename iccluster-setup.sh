#!/bin/bash

#################### Install basics ###############################################################

# Install stuff
sudo apt-get update
apt-get install -y git tmux screen vim htop gcc

# Config git
git config --global user.name "trouleau"
git config --global user.email "william.trouleau@iccluster.epfl.ch"

# Add personal ssh public key to authorized_keys
mkdir ~/.ssh
echo 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDPqJjY1js7ubxTdPdk0PuGWjSGJyVnDx4c3nGOzwRfTSYp5bpWIAfHpD7DQKX8LCIwVQ4z3WoQrO7/BY3VQ328OvoL+BsTUL89G1HtgdAK0jG+8ICydFdbwc9Yj4taV3gbyTKur3W9sgnC5t0vGXwbUl0Q3Af30BKJKYNJzJ9elzUmIANl5TI4CeaBqqieB2sDu6WjHwmB+AUb/We1YIcUmPcRM3tmOOo+IQUSNUsaeW/2CDV7SH1S0IHUVyfMpaywLN8Vc41dYC4/z1TiuAo44tl8jpdapNTxelpARLfIx1dcpsfhPTP+zs9IBUY0Q/TSBV/tiACHEhh/etGoVKtJ trouleau@icsil1noteb194.epfl.ch' >> ~/.ssh/authorized_keys

ssh-keygen -t rsa -b 4096 -C "william.trouleau@iccluster.epfl.ch" -f ~/.ssh/id_rsa
echo "The ssh key of the server is:"
cat ~/.ssh/id_rsa.pub
echo ""
read -p "Add the key to git repo host. Then press enter to continue"

# Install python with miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
echo ". /root/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
rm ~/miniconda.sh

#################### Prepare project environment ##################################################

# Make workspace dir
mkdir /root/workspace/

# Clone project source repos
cd /root/workspace/ && git clone git@github.com:trouleau/var-wold.git  # Add personal lib

# Make virtualenv
conda create -n env python=3.7 && conda activate env

cd /root/workspace/var-wold/ && pip install -e .  # Install internal lib
cd /root/workspace/var-wold/ && pip install -r requirements.txt
cd /root/workspace/var-wold/lib/granger-busca/ && pip install cython && pip install -e .  # Install gb
