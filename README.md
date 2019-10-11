# 4042_NNDL

4042 Neural Networks and Deep Learning

## Requirements

1. [Python](https://www.python.org/)
2. [Pipenv](https://github.com/pypa/pipenv)
3. [NumPy](https://numpy.org/)
4. [Matplotlib](https://matplotlib.org/)
5. [Tensorflow](https://www.tensorflow.org/)
6. [tqdm](https://tqdm.github.io/)

## Setup

### Initial

1. Download and Install Python
2. Navigate to your Python installation directory and copy path
3. Open _cmd_ in administrator mode
4. Input `cd <paste copied path>`
5. Input `py -x.x -m pip install --upgrade pip`
   1. Replace `x.x` with Python version number (i.e. downloaded: **3.7**.4, x.x: **3.7**)
6. Input `py -x.x -m pip install --upgrade setuptools`
   1. Replace `x.x` with Python version number (i.e. downloaded: **3.7**.4, x.x: **3.7**)
7. Input `pip install --user pipenv`
   1. Copy path presented after successful installation: _looks like_ `C:\Users\<Username>\AppData\Roaming\Python<Version>\Scripts`
8. Input abovementioned path into user's path

### Continued

1. Clone/Download project to a desired directory
2. Copy path to cloned/downloaded project directory after successful clone/download
3. Input `cd <path to cloned/download project path>`
4. Input `pipenv install numpy`
5. Input `pipenv install matplotlib`
6. Input `pipenv install tensorflow==1.14`
7. Input `pipenv install tqdm`

## Execution

1. To run, input `pipenv run python <file>.py`
