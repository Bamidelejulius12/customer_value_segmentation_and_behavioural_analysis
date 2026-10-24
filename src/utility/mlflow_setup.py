import dagshub
import mlflow
from dotenv import load_dotenv
import os
load_dotenv(override=True)

# local accessibility of mlflow dagshub
# def setup_mlflow():

#     dagshub.init(repo_owner='babatundejulius911', 
#                 repo_name='customer_value_segmentation_and_behavioural_analysis', 
#                 mlflow=True)
#     mlflow.set_experiment("vault_chain_customer_segmentation_project")


# remote accessibility of mlflow dagshub

def setup_mlflow():
    dagshub_token = os.getenv("MLFLOW_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("MLFLOW_TOKEN environment variable not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner='babatundejulius91' 
    repo_name='customer_value_segmentation_and_behavioural_analysis'

    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    mlflow.set_experiment("vault_chain_customer_segmentation_project")

