"""
This is a boilerplate pipeline 'data_pull'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import pull_stock_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=pull_stock_data,
                inputs=["params:stock_pull_settings"],
                outputs="raw_combined_equity_data", # change this setting if using separate outputs for each equity
                name="pull-in-stock-data",
            ),

        ]
    )
