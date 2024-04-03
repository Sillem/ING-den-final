from data_preparation.data_preparation import transform_data
import pandas as pd

train_data = pd.read_csv('./datasets/in_time.csv')

print(transform_data(train_data))