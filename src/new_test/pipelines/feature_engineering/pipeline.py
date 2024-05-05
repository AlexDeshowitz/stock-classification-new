"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.5
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import calculate_rolling_means, calculate_rolling_standard_deviations, \
                   create_above_below_indicator_fields, create_bollinger_bands, \
                   calculate_cumulative_days_above, create_classification_target_variable


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=calculate_rolling_means,
                inputs=["raw_combined_equity_data", "params:moving_average_settings"],
                outputs="combined_equity_data_moving_averages",
                name="Add-moving-averages",
            ),

            node(
                func=calculate_rolling_standard_deviations,
                inputs=["combined_equity_data_moving_averages", "params:moving_average_settings"],
                outputs="combined_equity_data_standard_deviations",
                name="Add-standard-deviations",
            ),

             node(
                func=create_above_below_indicator_fields,
                inputs=["combined_equity_data_standard_deviations", "params:moving_average_settings"],
                outputs="combined_equity_data_above_below_indicators",
                name="Add-indicator-fields",
            ),

            node(
                func=calculate_cumulative_days_above,
                inputs=["combined_equity_data_above_below_indicators", "params:moving_average_settings"],
                outputs="combined_equity_data_above_below_ind_cum",
                name="Add-cumulative-indicator-fields",
            ),

            node(
                func=create_bollinger_bands,
                inputs=["combined_equity_data_above_below_ind_cum",
                        "params:moving_average_settings",
                        "params:bollinger_band_settings"],
                outputs="combined_equity_data_w_bollinger_bands",
                name="Add-bollinger-fields",
            ),

            node(
                func=create_classification_target_variable,
                inputs=["combined_equity_data_w_bollinger_bands", "params:moving_average_settings", "params:target_classifier_settings"],
                outputs="combined_modeling_input",
                name="Add-classification-target",
            ),
        ]
    )