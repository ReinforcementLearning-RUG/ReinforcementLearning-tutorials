<br />
<p align="center">
  <h1 align="center">Reinforcement Learning</h1>

  <p align="center">
  </p>
</p>

## About
Repository containing reinforcement learning code. 

## Getting started

You can run the notebooks through google colab. Alternatively you can clone the repository and set up a local virtual environment.

### Prerequisites

- [Poetry](https://python-poetry.org/).

## Running
<!--
-->

You can also setup a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment (after `Add new interpreter`). The interpreter that Pycharm should use is `./.venv/bin/python3.10`.

If you want to add dependencies to the project then you can simply do
```
poetry add <package_name>
```
