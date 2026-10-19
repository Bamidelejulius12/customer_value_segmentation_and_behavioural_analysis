import dagshub
import mlflow

def setup_mlflow():

    dagshub.init(repo_owner='babatundejulius911', 
                repo_name='customer_value_segmentation_and_behavioural_analysis', 
                mlflow=True)
    mlflow.set_experiment("vault_chain_customer_segmentation_project")