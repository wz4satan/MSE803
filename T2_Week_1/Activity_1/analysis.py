from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset
iris = fetch_ucirepo(id=53)

# Features and target
X = iris.data.features
y = iris.data.targets

# Combine features and target into one dataframe
df = pd.concat([X, y], axis=1)

# Number of features
number_of_features = X.shape[1]

# Number of classes
number_of_classes = y["class"].nunique()

# Class names
class_names = y["class"].unique()

# Number of duplicate records
number_of_duplicates = df.duplicated().sum()

# Number of records per class
records_per_class = y["class"].value_counts()

print("Number of records:", df.shape[0])
print("Number of features:", number_of_features)
print("Number of classes:", number_of_classes)
print("Class names:", class_names)
print("Number of records per class:")
print(records_per_class)
print("Number of duplicate records:", number_of_duplicates)
