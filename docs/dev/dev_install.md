# Developer Installation

Install autoplex from source, by cloning the repository via [github](https://github.com/JaGeo/autoplex.git)

```bash
git clone https://github.com/JaGeo/autoplex.git
cd autoplex
pip install -e .[docs,strict,dev]
```
This will install autoplex will all dependencies for tests, pre-commit and docs building. 
However, note that non-python programs like `buildcell` and `julia` needed for ACE potential fitting will not be installed with above command. One needs to install these 
seperately.

Alternatively, one can use the `devcontainer` provided to have your developer environment setup automatically in your IDE. It has been tested to work in [VSCode](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container) and [PyCharm](https://blog.jetbrains.com/pycharm/2023/06/2023-2-eap-4/). 
Only prerequisite is one has [docker](https://docs.docker.com/get-started/get-docker/) installed on the system as it uses the published docker images to create this developer env. 


## Running unit tests

Unit tests can be run from the source folder using `pytest`. 

```bash
pytest
```
This will run all the tests.

To get a detailed report of test coverage you can use following command
```bash
pytest --cov=autoplex --cov-report term-missing --cov-append
```

If you feel test execution takes too long locally, you can speedup the execution using [pytest-xdist](https://pypi.org/project/pytest-xdist/). Install this in library in your environment using

```bash
pip install pytest-xdist
```

Once installed, you can now use multiple processors to run your tests. For example, if you want to use 8 processors to run tests in parallel, run

```bash
pytest -n 8
```

We rely on pytest-split to run tests in parallel on github workflow, thus it is necessary to update the test-durations files in the repository, incase you add new tests. To generate this file, use

```bash
pytest --cov=autoplex --cov-append --splits 1 --group 1 --durations-path ./tests/test_data/.pytest-split-durations --store-durations
```

## Building the documentation locally

The autoplex documentation can be built using the sphinx package.

The docs can be built to the `_build` directory:

```bash
sphinx-build -W docs _build
```
