# **Contributing**

We welcome contributions to the Holistic AI library. If you are interested in contributing, in this section we provide some guidelines to get you started.

> **Note:** All of these instructions have been tested on a fresh installation of Ubuntu 20.04.2 LTS. If you are using a different operating system, you may need to adapt some of the instructions below.

## **Prerequisites:**

- Ubuntu >= 20.04
  
Although the holistic ai package can be developed on any operating system, we highly recommend using a Linux distribution. This will allow you to use the same environment as the one used for the continuous integration pipeline.

If you are a Windows user, you can use the Windows Subsystem for Linux (WSL) to install a Linux distribution on your machine. You can find instructions on how to install WSL [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10).


## **Installation:**

First, you will need to install the `pyenv` package. This will allow you to install multiple versions of Python on your machine. You can install it by running the following instructions:

### Installing Pyenv

- Install the dependencies:
```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
curl https://pyenv.run | bash
```

- Then, you need to add the following lines to your .bashrc file:
```bash
export PATH="$HOME/.pyenv/bin:$PATH" && eval "$(pyenv init --path)" && echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
```

- And source the .bashrc file:
```bash
source ~/.bashrc
```


- Finally, you can install any Python version. We highly recommend the 3.9.8, you can install it with:
```bash
pyenv install 3.9.8
```

### Installing Poetry

Next, you will need to install Poetry. Poetry is a tool for dependency management and packaging in Python. You can install it by running the following instructions:

- Install the dependencies:
```bash
sudo apt update
sudo apt install python3-distutils
```

- Download the installer using curl:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

- And source the .bashrc file:
```bash
source ~/.bashrc
```

### **Configuring the virtual environment:**

Now that you have installed all the dependencies, you can clone the repository and configure the virtual environment:

- Clone the repository:

```bash
git clone git@github.com:holistic-ai/holisticai.git
```

- Go to the repository folder:

```bash
cd holisticai
```

- Set the Python version to 3.9.8:

```bash
pyenv local 3.9.8
```

- To avoid conflicts with the system packages, we will appoint Poetry to the Python version installed in the previous step:

```bash
which python
```

- This will return the path to the python version installed in the previous step. Copy this path and run the following command:

```bash
poetry env use <path_to_python>
```

- And then, install the dependencies:

```bash
poetry install
```

- Once you have finished the installation, you can observe the virtual environment created by poetry:

```bash
poetry env list
```

- The output should be similar to this:

```bash
holisticai-5faADsE3-py3.9 (Activated)
```

- Now, we need to activate the virtual environment, first we need to find the path to the virtual environment:

```bash
poetry env info --path
```

- This will return the path to the virtual environment. Copy this path and run the following command:

```bash
source <copied path>/bin/activate
```

- Finally, we can run the tests:

```bash
pytest
```
