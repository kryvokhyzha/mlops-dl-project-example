# mlops-dl-project-example

This repository contains the example of the `MLOps` project for the training
`Deep Learning` model.

## Setup python environment

1. Clone the repository using `git clone` command.
2. Open the terminal and go to the project directory using `cd` command.
3. Create virtual environment using `python -m venv venv` or
   `conda create -n venv python=3.10` command. We have used `Python 3.10` during
   development.
4. Activate virtual environment using `source venv/bin/activate` or
   `conda activate venv` command.
5. Install poetry using instructions from
   [here](https://python-poetry.org/docs/#installation). Use
   `with the official installer` section.
6. Set the following option to disable new virtualenv creation:
   ```bash
   poetry config virtualenvs.create false
   ```
7. Install dependencies using `poetry install --no-root -E all` command. The
   `--no-root` flag is needed to avoid installing the package itself.
8. Setup `pre-commit` hooks using `pre-commit install` command. More information
   about `pre-commit` you can find [here](https://pre-commit.com/).
9. Run the test to check the correctness of the project work using following
   command:
   ```bash
   python -m unittest -b
   ```
10. After successful passing of the tests, you can work with the project!
11. If you want to add new dependencies, use `poetry add <package_name>`
    command. More information about `poetry` you can find
    [here](https://python-poetry.org/docs/basic-usage/).
12. If you want to add new tests, use `unittest` library. More information about
    `unittest` you can find
    [here](https://docs.python.org/3/library/unittest.html). All tests should be
    placed in the `tests` directory.
13. All commits should be checked by `pre-commit` hooks. If you want to skip
    this check, use `git commit --no-verify` command. But it is not recommended
    to do this.
14. Also, you can run `pre-commit` hooks manually using
    `pre-commit run --all-files` command.
15. More useful commands you can find in `Makefile`.

## Setup ClearML server

See the following [guide](docs/clearml/README.md) to setup ClearML server.

## Examples

### How to start?

1. TODO
