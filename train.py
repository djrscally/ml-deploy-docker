import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cloudpickle

#%%

df = pd.read_csv(
    './data/adult.data',
    names=[
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'income'
    ])

#%%

df.info()

#%%

NUMERIC_FEATURES = [
    'age',
    'fnlwgt',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]

CATEGORICAL_FEATURES = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country'
]
#%%
numeric_pipeline = Pipeline(steps=[
    ('Imputer', SimpleImputer(strategy='constant', fill_value=0.))
])

categorical_pipeline = Pipeline(steps=[
    ('OneHotEncoders', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('NumericFeatures', numeric_pipeline, NUMERIC_FEATURES),
    ('CategoricalFeatures', categorical_pipeline, CATEGORICAL_FEATURES)
], remainder='drop')

#%%
estimator = RandomForestClassifier(n_jobs=4, n_estimators=50, max_depth=20)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('estimator', estimator)
])

#%%
x_train, x_test, y_train, y_test = train_test_split(df.drop('income', axis=1), df.income.astype('category'))
#%%
print("Training...")
pipeline.fit(x_train, y_train)
print("Training complete. Validation results: ")
#%%
print(classification_report(y_test, pipeline.predict(x_test)))
#%%

cloudpickle.dump(pipeline, open('artifacts/model.pkl', 'wb'))

print("Training script complete")
