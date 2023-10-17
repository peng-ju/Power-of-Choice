# Power-of-Choice

reproduce the paper "Towards Understanding Biased Client Selection in Federated Learning"

## Instructions

1. Clone the repo.
2. (Recommended) Create and activate a [virtualenv](https://virtualenv.pypa.io/) under the `env/` directory. Git is already configured to ignore it.
3. Install the very minimal requirements using `pip install -r requirements.txt`
4. Run [Jupyter](https://jupyter.org/) in whatever way works for you. The simplest would be to run `pip install jupyter && jupyter notebook`.
5. All relevant code and instructions are in [`Example.ipynb`](/Example.ipynb).

## Explanation

This project structure is as an example of how to work with DVC from inside a Jupyter Notebook.

This workflow should enable you to enjoy the full benefits of working with Jupyter Notebooks, while getting most of the benefit out of DVC - 
namely, **reproducible and versioned data science**.

The project takes a toy problem as an example - the [California housing dataset](https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset), which comes packaged with scikit-learn.
You can just replace the relevant parts in the notebook with your own data and code.
Significantly different project structures might require deeper intervention.  

The idea is to leverage DVC in order to create immutable snapshots of your data and models as part of your git commits.
To enable this, we created the following DVC stages:
1. **Raw data** - kept in `data/raw/`, versioned in `data/raw.dvc` 
2. **Processed data** - kept in `data/processed/`, versioned in `process_data.dvc` 
3. **Trained models** - kept in `models/`, versioned in `models.dvc` 
4. **Metrics** - kept in `metrics/metrics.json`, versioned as part of the git commit and referenced in `models.dvc`

Unlike a typical DVC project, which requires you to refactor your code into modules which are runnable from the command line,
In this project the aim is to enable you to stay in your comfortable notebook home territory.

So, instead of using `dvc repro` or `dvc run` commands, **just run your code as you normally would in [`Example.ipynb`](/Example.ipynb)**. 
We prepared special cells (marked with green headers) inside this notebook that let you run `dvc commit` commands on the relevant
DVC stages defined above, immediately after you create the relevant data files from your notebook code.

[`dvc commit`](https://dvc.org/doc/commands-reference/commit) computes the hash of the versioned data and saves that hash
as text inside the relevant `.dvc` file. The data itself is ignored and not versioned by git, instead being versioned with DVC.
However, the `.dvc` files, being plain text files, ARE checked into git.

So, to summarize, this workflow should enable you to create a git commit which contains all relevant code, together with
*references* to the relevant data and the resulting models and metrics. **Painless reproducible data science!**

It's intended as a guideline - definitely feel free to play around with its structure to suit your own needs.

---

To create a project like this, just go to https://dagshub.com/repo/create and select the **Jupyter Notebook + DVC** project template.

Made with 🐶 by [DAGsHub](https://dagshub.com/).
