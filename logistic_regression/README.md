## Directory Structure

```
├── reference_implementation            # original code for reference
├── __init__.py
├── FedAvg.py                           # our implementation
├── main.py                             # main experiment runner script
├── setup_mlflow.py                     # helper script for mlflow setup
├── synthetic_m=1.pdf                   # sample Figure 3(a) obtained
├── synthetic_m=3.pdf                   # sample Figure 3(b) obtained
├── utils.py                            # helper functions
└── README.md
```

## Instructions
To reproduce Figure 3 from the main paper, follow these steps:
* Activate the env: `$ conda activate myenv`
* For Figure 3(a): run `$ python main.py 1` to obtain `synthetic_m=1.pdf`
* For Figure 3(b): run `$ python main.py 3` to obtain `synthetic_m=3.pdf`
* Each figure takes only a few mins to complete.
