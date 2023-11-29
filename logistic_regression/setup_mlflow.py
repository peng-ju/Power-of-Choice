import mlflow

# setup environment variables for remote tracking
# $ export MLFLOW_TRACKING_USERNAME=<username>
# $ export MLFLOW_TRACKING_PASSWORD=<token>

MLFLOW_TRACKING_URI = 'https://dagshub.com/peng-ju/Power-of-Choice.mlflow'

def get_experiment_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    return experiment_id

# print(get_experiment_id('tester'))