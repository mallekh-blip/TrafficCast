## Project overview

TrafficCast predicts five-minute freeway vehicle counts at 15-, 30-, and
60-minute horizons using the San Diego subset of LargeST.

The main evaluation compares LightGBM and XGBoost against persistence,
same time last week, and simple exponential smoothing on 100 sensors.
The interactive dashboard uses a separate 30-sensor diagnostic sample
and will be added in a later commit.

## Repository scope

This repository contains the forecasting and evaluation code.
The datasets, fitted models, and evaluation outputs are stored separately
and are required to reproduce the full study.
