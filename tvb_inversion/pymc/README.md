# Installation guide of PyMC and TVB backend

To install the latest version of the PyMC methods from the GitHub repository type the following in the terminal:

```shell
$ git clone https://github.com/the-virtual-brain/tvb-inversion.git
$ git checkout pymc-dev
```
Before executing the next steps, make sure to activate your python environment (we recommend to use anaconda and python 3.9):
```shell
$ cd tvb-inversion/
$ python -m pip install .
$ conda install -c conda-forge pymc pytensor
```
You could also use `pip` to install PyMC and PyTensor:
```shell
$ python -m pip install pymc
$ python -m pip install pytensor
```
To use the PyMC package with TVB, the TVB backend for pytensor is required. This can be installed via:
```shell
$ git clone https://github.com/the-virtual-brain/tvb-root.git
$ git checkout pytensor-backend
$ cd tvb-root/tvb_build/
$ sh install_full_tvb.sh
```

## Instructions for the CSCS cluster
To use the CSCS cluster, a project lead has to invite your institutional e-mail address to the project. Afterwards you get a link to create your CSCS account. To access the server you have to create a ssh key pair using:
```shell
$ ssh-keygen -t ed25519
```
The key pair will be saved on your local machine under `~/.ssh/`. Then you have to copy your ssh key to the CSCS front end service `ela.cscs.ch`:
```shell
$ ssh-copy-id username@ela.cscs.ch
$ ssh-copy-id -i ~/.ssh/id_ed25519.pub username@ela.cscs.ch
```
You can now ssh to the service via:
```shell
$ ssh username@ela.cscs.ch
```
The computing resources are provided by the computer cluster Piz Daint. You can access this service via ssh from the ela frontend. For this you have to copy the ssh key from `ela.cscs.ch` to `daint.cscs.ch`:
```shell
[username@ela1 ~]$ ssh-copy-id username@daint.cscs.ch
[username@ela1 ~]$ ssh-copy-id -i ~/.ssh/id_ed25519.pub username@daint.cscs.ch
```
You can now ssh to the Piz Daint service via:
```shell
[username@ela1 ~]$ ssh username@daint.cscs.ch
```
To avoid the double login process and shortcut the commands, you can edit your ssh config file on your local machine:
```shell
$ vim ~/.ssh/config
```
and the following lines:
```
Host daint
    HostName daint.cscs.ch
    User username
    ProxyJump username@ela.cscs.ch
    LocalForward 8118 localhost:8888
```
This will also add port forwarding to use running jupyter notebooks on the server in your local browser. The last line makes port 8118 of your local machine to listen to the remote port 8888. This means that a jupyter notebook, that is running on the remote server on `localhost:8888`, can be access on your local machine under `localhost:8118`. 

Information and instructions on how to use the slurm job manager on Piz Daint can be found here: https://user.cscs.ch/access/running/piz_daint/.
A template bash script to submit a slurm job can be found in this repo under `examples/pymc_bash.sh`.