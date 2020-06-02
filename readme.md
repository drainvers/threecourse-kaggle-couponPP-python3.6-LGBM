## Dependency
OS : Windows 10 version 2004
```
Python 3.6.8
LightGBM v2.3.2
```
used python packages are below:
```
ipython 7.14.0
pandas 1.0.3
numpy 1.18.4  
scipy 1.4.1
scikit-learn 0.23.1
joblib 0.15.1
```

## Data
Add Data from Kaggle into "input" folder.
(except prefecture_locations.csv. BOM-removed file is already there.)

## Run
Run by `ipython batch.py` in the src folder.
It took 1 hour, 2 minutes and 49 seconds to run on a machine with the following specs:
- Processor: i5-6200U
- RAM: 12GB
- Threads: 2
To adjust the number of threads to use, change the `threads` variable in `d00_create_lgbdata.py`, line 33

## Notes
The [original code](https://github.com/threecourse/kaggle-coupon-purchase-prediction) by threecourse blended the results of XGBoost and Vowpal Wabbit models. I modified the code to use LightGBM only and also run with newer versions of libraries.

## Latest Results (MAP@10)
Private: 0.00932
Public: 0.01157
