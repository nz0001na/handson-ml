import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.base import BaseEstimator , TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# root_path = 'https://github.com/ageron/handson-ml/tree/master/'
# root_path = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
src_path = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
dst_path = 'datasets/example-2-1/'

# fetch data vis urllib
def fetch_housing_data(src_path, dst_path):
    if os.path.exists(dst_path) is False:
        os.makedirs(dst_path)

    tgz_file,_ = urllib.request.urlretrieve(src_path)
    tgz_data = tarfile.open(tgz_file, mode='r')
    tgz_data.extractall(path=dst_path)
    tgz_data.close()

# load data from csv file
def load_housing_data(file_path):
    return pd.read_csv(file_path)

def split_train_test(data, test_ratio):
    # np.random.seed(42)
    test_size = int(test_ratio*len(data))
    shuffled_index = np.random.permutation(len(data))
    test_index = shuffled_index[:test_size]
    train_index = shuffled_index[test_size:]
    return data.iloc[train_index], data.iloc[test_index]


class DataFrameSelector(BaseEstimator,TransformerMixin):
  def __init__(self, attribute_names):
      self.attribute_names = attribute_names

  def fit(self, X, y=None):
      return self

  def transform(self, X):
      return X[self.attribute_names].values



# fetch_housing_data(src_path=src_path, dst_path=dst_path)
file_path = dst_path + 'housing.csv'
data = load_housing_data(file_path=file_path)
# get the first five records
top5 = data.head()
# get all values
info = data.info
info2 = data["ocean_proximity"].value_counts()
info3 = data.describe()

# plot
data.hist(bins=50, figsize=(20,15))
plt.show()
data.plot(kind='scatter', x='longitude', y='latitude', colorbar=True, alpha=0.4, label='population', s=data['population']/100, c='median_house_value', cmap=plt.get_cmap('jet'))
plt.show()
# get correaltion
corr_matrix = data.corr()
attribute=['median_house_value','median_income']
scatter_matrix(data[attribute], figsize=(12,8))
plt.show()

data.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
plt.show()

train_data, test_data = split_train_test(data, 0.2)
print('train len=' + str(len(train_data)))
print('test_len=' + str(len(test_data)))
# data cleaning
# data.dropna(subset=['total_bedrooms'])
# data.drop('total_bedrooms',axis=1)
median = data['total_bedrooms'].median()
data['total_bedrooms'].fillna(median)

imputer = Imputer(strategy='median')
house_num = data.drop('ocean_proximity',axis=1)
imputer.fit(house_num)

encoder = LabelEncoder()
housing_cat_encoded = encoder.fit_transform(data['ocean_proximity'])
print(encoder.classes_)

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(data.values)

num_attribs = list(house_num)
cat_attribs = ['ocean_proximity']

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', OneHotEncoder(sparse=False)),
])

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
    ('std_scaler', StandardScaler()),
])


full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])


housing_prepared = full_pipeline.fit_transform(data)

print('done')