# ML_project_expedia
Code must run in an order as the number given to the files.
For example 1st run 0_data_split.py then 1_leakage_solution.py.

## File descriptions:
0_data_split.py : This file divides training data to train_new.csv and test_new.csv. These files will be used for validation.
1_leakage_solution.py : This file tackles with leakage data that happened after the start of the kaggle competition.
2_random_forest.py : Random forest model.
3_xgboost.py : XgBoost model.
4_adaboost.py : Adaboost model.
5_merge models.py : Combines all the three models.
6_final.py : Combines merge models with leakage solution.
