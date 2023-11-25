# Power-of-Choice

reproduce the paper "Towards Understanding Biased Client Selection in Federated Learning"

## Activate virtual environment with Python3.10

```
source venv/bin/activate 
deactivate
```

- python==3.10.13

## Set DagsHub as MLflow server

```
export MLFLOW_TRACKING_USERNAME=<token>
```

or 

```
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<password/token>
```

## Set the remote to our dvc remote. This will allow us to interact with DagsHub's DVC storage.

```
dvc remote add origin s3://dvc
dvc remote modify origin  endpointurl https://dagshub.com/<username>/<repo-name>.s3
dvc remote modify origin --local access_key_id <token>
dvc remote modify origin --local secret_access_key <token>
```

## Instructions

1. Make sure you have installed `python3.10.13` or do `sudo apt install Python3.10.13`.
1. Clone the repo.
2. (Recommended) Create and activate a [virtualenv](https://virtualenv.pypa.io/) under the `env/` directory. Git is already configured to ignore it.
3. Install the very minimal requirements using `pip install -r requirements.txt`
4. Run [Jupyter](https://jupyter.org/) in whatever way works for you. The simplest would be to run `pip install jupyter && jupyter notebook`.
5. Run Python script within your virtual environment.
<!-- 5. All relevant code and instructions are in [`Example.ipynb`](/Example.ipynb). -->

## Save and load model using MLflow

Please refer to `MLflow_guide.ipynb` for detailed information.
