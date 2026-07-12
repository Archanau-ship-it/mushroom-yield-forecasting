
# Random Forest Cross Validation

## Method

TimeSeriesSplit with 5 folds was used.

## Cross Validation Results

Mean CV MAE : 0.157

Std CV MAE : 0.029

## Overfitting Check

Training MAE : 0.052

Testing MAE : 0.188

Observation:

Training error is expected to be lower than testing error.
If the difference is small, the model generalizes well.
A very large difference indicates overfitting.

## Conclusion

Random Forest was compared against Linear Regression using identical train-test splits and evaluation metrics.
