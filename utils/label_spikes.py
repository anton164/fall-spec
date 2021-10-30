from merlion.utils import TimeSeries
from merlion.models.anomaly.forecast_based.prophet import ProphetDetector, ProphetDetectorConfig
from merlion.plot import plot_anoms_plotly
from merlion.post_process.threshold import Threshold
import pandas as pd

def convert_to_merlion(df, column):
    df = df.copy()
    df = df[column]
    df.index = df.index.tz_convert(None)
    return TimeSeries.from_pd(df)

def timeseries_from_query_counts(query_counts):
    df_timeseries = pd.DataFrame([
        {"time_end": timestep["end"], "tweet_count": timestep["tweet_count"]}
        for timestep in query_counts["data"]
    ]).sort_values("time_end", ascending=True)

    df_timeseries["time_end"] = pd.to_datetime(df_timeseries["time_end"])
    return df_timeseries.set_index("time_end")

def detect_anomalies(
    df_timeseries, 
    column,
    model=ProphetDetector(ProphetDetectorConfig(
        threshold=Threshold(alm_threshold=0.5),
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        add_seasonality=False,
        uncertainty_samples=1000
    )),
    include_plot=True
):
    train_data = convert_to_merlion(df_timeseries, column)
    anomaly_score = model.train(train_data=train_data, anomaly_labels=None)
    scores = model.get_anomaly_score(train_data)
    df_scores = scores.to_pd()
    labels_train = model.get_anomaly_label(train_data)
    df_labels = labels_train.to_pd()

    if (include_plot):
        fig = model.plot_anomaly_plotly(
            time_series=train_data,
            plot_forecast=True,
            plot_forecast_uncertainty=True
        )
        plot_anoms_plotly(fig, anomaly_labels=labels_train)
        return df_scores, df_labels, fig
    
    else:
        return df_scores, df_labels