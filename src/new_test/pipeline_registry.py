"""Project pipelines."""
from typing import Dict


from kedro.pipeline import Pipeline
from new_test.pipelines import data_pull

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_pull_pipeline = data_pull.create_pipeline()


    return {
       
            "__default__": data_pull_pipeline,
            "dat_pull_only" : data_pull_pipeline 
           }   