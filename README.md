# Power-of-Choice
Reproducibility study of the paper "Towards Understanding Biased Client Selection in Federated Learning" by Yae Jee Cho, Jianyu Wang and Gauri Joshi.


## Directory Structure
```
.
├── quadratic_optimization                      # Experiment 1: Quadratic optimization
├── logistic_regression_synthetic_data          # Experiment 2: Logistic regression using synthetic data
├── dnn                                         # Experiment 3: Image classification using FMNIST, CIFAR10 data
├── MLP_sentiment_analysis_Twitter              # Experiment 4: MLP using Twitter data
├── data
├── MLflow_guide.ipynb                          # ML flow guide
├── jee-cho22a.pdf                              # Original paper for reproducibility
├── requirements.txt                            # PIP requirements file
├── environment.yml                             # CONDA environment file
├── .gitignore
├── LICENSE
└── README.md
```


## Dataset
All dataset files (except synthetic data) are automatically downloaded from their respective repositories. Synthetic data is included in our repository. No action is required for downloading/preprocessing any data.


## Getting Started
We highly recommending setting up the environment through following commands instead of using `environment.yml` or `requirements.txt` to avoid issues arising from different architectures/machines.  

To get started, install `conda` distribution for managing python packages. Create an environment:

Step 1: Create and activate conda environment
* `$ conda create -n myenv python=3.10 ipython ipykernel -y`
* `$ conda activate myenv`
* `$ python -m ipykernel install --user --name myenv --display-name "Python (myenv)"`

Step 2.1: Install **PyTorch 2.0.1**, [reference](https://pytorch.org/get-started/previous-versions/#v201)
* For Mac: `$ conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch -y`
* For Windows/Ubuntu (CPU): `$ conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch -y`
* Verify install:  
```bash
$ python -c "import torch; print(f'Torch: {torch.__version__}\nCUDA: {torch.version.cuda}\nCUDA devices: {torch.cuda.device_count()}')"
```

Step 2.2: Install common packages
* ML Toolkits: `$ conda install -c anaconda pandas numpy tqdm -y`
* Misc: `conda install -c conda-forge matplotlib jupyterlab -y`
* Verify install:
```bash
$ python -c "import pandas, numpy, tqdm, matplotlib; print ('Done')"
```

Note: you will need to activate the environment everytime you run our scripts. For jupyter notebook/lab, you need to select our custom kernel "Python (myenv)" created in Step 1.


## Instructions
To reproduce the results of the main paper, follow these steps:

* **Quadratic Experiments**: Follow the readme guide for [./quadratic_optimization](./quadratic_optimization)
* **Logistic Experiments on Synthetic Data**: Follow the readme guide for [./logistic_regression_synthetic_data](./logistic_regression_synthetic_data)
* **Image Classification Expts on FMNIST, CIFAR10 Data**: Follow the readme guide for [./dnn](./dnn)


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

## Save and load model using MLflow
Please refer to `MLflow_guide.ipynb` for detailed information.
