"""Project pipelines."""
from typing import Dict


from kedro.pipeline import Pipeline
from new_test.pipelines import data_pull, feature_engineering, Ml_pre_processing

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_pull_pipeline = data_pull.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    pre_processing =Ml_pre_processing.create_pipeline()


    return {
       
            "__default__": feature_engineering_pipeline + pre_processing,

            # register individual pipelines:
            "data_pull_only" : data_pull_pipeline,
            "feature_engineering_only" : feature_engineering_pipeline,
            "pre_processing_only" : pre_processing

           }   