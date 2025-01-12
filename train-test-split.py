import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Save the datasets to CSV
train_data = pd.DataFrame(X_train)
train_data['Survival_Rate'] = y_train
train_data.to_csv('train_data.csv', index=False)

val_data = pd.DataFrame(X_val)
val_data['Survival_Rate'] = y_val
val_data.to_csv('validation_data.csv', index=False)

test_data = pd.DataFrame(X_test)
test_data['Survival_Rate'] = y_test
test_data.to_csv('test_data.csv', index=False)

print("Data saved successfully!")
