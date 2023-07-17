# **Contributing**

We welcome contributions to the Holistic AI library. If you are interested in contributing, in this section we provide some guidelines to get you started.

> **Note:** All of these instructions have been tested on a fresh installation of Ubuntu 20.04.2 LTS. If you are using a different operating system, you may need to adapt some of the instructions below.

## **Prerequisites:**

- Ubuntu >= 20.04
  
Although the holistic ai package can be developed on any operating system, we highly recommend using a Linux distribution. This will allow you to use the same environment as the one used for the continuous integration pipeline.

If you are a Windows user, you can use the Windows Subsystem for Linux (WSL) to install a Linux distribution on your machine. You can find instructions on how to install WSL [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10).


## **Installation (Using Conda):**

First, you will need to install the `conda`. This will allow you to install multiple python environments on your machine. You can install it by running the following instructions:

Using Conda to create a new environment "holistic-ai"

```bash
conda create --name holistic-ai python==3.10
conda activate holistic-ai
conda install pip poetry

git clone git@github.com:holistic-ai/holisticai.git
cd holisticai
poetry install --all-extras
```

-to send a PR first you have to run the tests:

```bash
poetry run pytest
```

and format the code using pre-commit

```bash
pre-commit run --show-diff-on-failure --color=always --all-files
```
