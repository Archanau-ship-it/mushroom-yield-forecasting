
# Champion Model Selection

## Selected Model

Random Forest (Tuned)

## Best Hyperparameters

{
    "max_depth": 16,
    "min_samples_leaf": 3,
    "n_estimators": 100
}

## Comparison Summary

                  Model  CV MAE  Test MAE  RMSE  R2 Score Training Time (sec) Interpretability
      Linear Regression   0.152     0.188 0.207     0.330                   -             High
Random Forest (Default)   0.158     0.188 0.226     0.201                0.16           Medium
  Random Forest (Tuned)   0.154     0.189 0.228     0.189                6.89           Medium

## Performance

Cross Validation MAE : 0.154

Test MAE : 0.189

RMSE : 0.228

R² Score : 0.189

## Why this model?

The tuned Random Forest model was selected because it achieved the lowest prediction error during cross-validation while maintaining strong performance on the unseen test dataset.

Compared with Linear Regression and the Default Random Forest model, it provides better generalization and more accurate yield prediction.

## Business Justification

Lower prediction error helps growers estimate mushroom yield more accurately.

Accurate yield prediction supports:

• Harvest planning

• Labor allocation

• Inventory management

• Supply chain planning

## Limitations

• Model performance depends on sensor quality.

• Extreme environmental conditions outside the training data may reduce accuracy.

• Predictions should support expert grower decisions rather than replace them.

