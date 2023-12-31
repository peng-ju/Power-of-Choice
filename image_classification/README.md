## Directory Structure
```
├── deprecated            # original code for reference
├── configs               # hyperparameter configs for diff experiments
├── logs                  # logs output directory
├── __init__.py
├── FedAvg.py             # our implementation
├── main.py               # main experiment runner script
├── models.py             # model definitions
├── plot.py               # script for generating plots
├── launch.sh             # launch script for running all expts
├── utils.py              # helper functions
├── job.sh                # slurm job script for Purdue's Lawson servers
├── scholarjob.sh         # slurm job script for Purdue's RCAC scholar cluster
└── README.md
```


## Instructions
To reproduce, for instance, Figure 4(a) from the main paper, follow these steps:
* Activate the env: `$ conda activate myenv`
* The hyperparameter configuration is present in `configs/fig4a.json`
* Obtains plots in logs directory: `$ python main.py --config configs/fig4a.json`
* Each configuration (except fig3a/fig3b) take around 1.5hrs to complete. Fig3a/3b take only a few mins.
