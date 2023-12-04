# Power-of-Choice
Reproducibility study of the paper ["Towards Understanding Biased Client Selection in Federated Learning"](./jee-cho22a.pdf) by Yae Jee Cho, Jianyu Wang and Gauri Joshi.

* [Report](./Report-ReScience.pdf)
* [Slides](./Presentation-Hackathon.pdf)


## Directory Structure
```
.
├── quadratic_optimization            # Experiment 1: Quadratic simulations
├── logistic_regression               # Experiment 2: Logistic regression using synthetic data
├── image_classification              # Experiment 3: Image classification using FMNIST, CIFAR10 data
├── sentiment_analysis                # Experiment 4: Sentiment analysis using Twitter data
├── data                              # synthetic data
├── MLflow_guide.ipynb                # ML flow guide
├── jee-cho22a.pdf                    # Original paper for reproducibility
├── Report-ReScience.pdf              # Our reproducibility report
├── Presentation-Hackathon.pdf        # Presentation talk for hackathon
├── requirements.txt                  # PIP requirements file
├── environment.yml                   # CONDA environment file
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
* For Windows/Ubuntu (GPU with Cuda 11.8): `$ conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia`
* Verify install:  
```bash
$ python -c "import torch; print(f'Torch: {torch.__version__}\nCUDA: {torch.version.cuda}\nIs CUDA available: {torch.cuda.is_available()}\nCUDA devices: {torch.cuda.device_count()}')"
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
To reproduce the experimental results of the main paper, follow these steps:

* **Quadratic simulations**: Follow the readme guide for [`./quadratic_optimization`](./quadratic_optimization)
* **Logistic Regression on Synthetic Data**: Follow the readme guide for [`./logistic_regression`](./logistic_regression)
* **Image Classification on FMNIST, CIFAR10 Data**: Follow the readme guide for [`./image_classification`](./image_classification)
* **Sentiment Analysis on Twitter Data**: Follow the readme guide for [`./sentiment_analysis`](./sentiment_analysis)


## Custom Experiment
To write your own custom federated learning experiment, you may reuse the codebase/pipeline for `image_classification`. Under this directory, add your custom dataset to `data_utils.py`, write your custom client selection algorithm in `FedAvg.py`, specify custom hyperparams in a new config file `custom.json` and you're ready to run your experiment by `$ python main.py -c configs/custom.json`


## Set DagsHub as MLflow server
```
export MLFLOW_TRACKING_USERNAME=<token>
```

or 

```
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<password/token>
```


## Save and load model using MLflow
Please refer to `MLflow_guide.ipynb` for detailed information.


## TODO
- [ ] difference in loss values for image classification experiments (but accuracy values matches approximately)
- [ ] remove default values in argparse, to be doubly sure that only the provided values are used
- [ ] confirm correctness of pipeline with another fedml code/paper
- [ ] distirbuted training setup using pytorch


## Contact 
[Peng Ju](mailto:ju26@purdue.edu), [Gautam Choudhary](mailto:gc.iitr@gmail.com)
