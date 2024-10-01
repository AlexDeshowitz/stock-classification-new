"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_models, extract_feature_importances, select_champion_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_models,
                inputs=["X_train", "X_test", "y_train", "y_test", "params:modeling_settings"],
                outputs=["detailed_model_results", "aggregated_model_results", "detailed_feature_importances","feature_importances"],
                name="model-training",
            ),

            node(
                func=select_champion_model,
                inputs=["detailed_model_results", "params:modeling_settings"],
                outputs="champion_model_details",
                name="champion-selection",
            )

        ]
    
    )
