from datetime import datetime
import dill
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def clean_data(data):
    data_clean = data.copy()

    data_clean.device_brand = data_clean.device_brand.apply(lambda x:
                                                            'other' if x == '(not set)' else x)

    data_clean['device_browser'] = data_clean['device_browser'].apply(lambda x:
                                                                      'other' if x == '(not set)' else x)

    data_clean['utm_medium'] = data_clean['utm_medium'].apply(lambda x:
                                                              'other' if (x == '(none)') | (x == '(not set)') else x)

    data_clean['device_screen_resolution'] = data_clean['device_screen_resolution'].apply(lambda x:
                                                                                          data_clean.device_screen_resolution.mode()[
                                                                                              0] if x == '(not set)' else x)

    return data_clean


def featuring(df):
    data_clean = df.copy()

    data_clean.visit_date = pd.to_datetime(data_clean.visit_date, utc=True)
    data_clean['day_of_week'] = data_clean.visit_date.dt.weekday
    data_clean['visit_hour'] = data_clean['visit_time'].apply(lambda x: x.split(':')[0])
    data_clean['visit_hour'] = pd.to_numeric(data_clean.visit_hour)
    data_clean['is_night'] = data_clean['visit_hour'].apply(lambda x: 1 if 6 < x < 22 else 0)

    data_clean['screen_resolution_X'] = data_clean['device_screen_resolution'].apply(lambda x:
                                                                                     x.split('x')[0])
    data_clean['screen_resolution_Y'] = data_clean['device_screen_resolution'].apply(lambda x:
                                                                                     x.split('x')[1])

    data_clean.screen_resolution_X = data_clean.screen_resolution_X.astype(int)
    data_clean.screen_resolution_Y = data_clean.screen_resolution_Y.astype(int)

    data_clean['screen_resolution_sqr'] = data_clean.screen_resolution_X * data_clean.screen_resolution_Y
    data_clean = data_clean.drop(columns=['screen_resolution_X', 'screen_resolution_Y'])

    data_clean['geo_russia'] = data_clean['geo_country'].apply(lambda x: 1 if x == 'Russia' else 0)

    data_clean['geo_city_1'] = data_clean.geo_city.apply(
        lambda x: 1 if x == ('Moscow' or 'Saint Petersburg') else 0)

    data_clean['geo_cat'] = data_clean.geo_city_1 + data_clean.geo_russia

    columns_for_drop = ['visit_date', 'visit_time', 'device_screen_resolution',
                        'geo_country', 'geo_city', 'geo_russia', 'geo_city_1', 'visit_hour']

    return data_clean.drop(columns=columns_for_drop)


def main():

    df_sessions = pd.read_csv('data/ga_sessions.csv', low_memory=False).drop(columns=['device_model',
                                                                                      'utm_keyword', 'device_os',
                                                                                      'client_id'])
    df_hits = pd.read_csv('data/ga_hits.csv').drop(columns=['event_value', 'hit_time', 'hit_referer',
                                                            'event_label', 'hit_date', 'hit_number',
                                                            'hit_type', 'hit_page_path', 'event_category'])

    target_list = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                   'sub_open_dialog_click', 'sub_custom_question_submit_click',
                   'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
                   'sub_car_request_submit_click']
    df_hits['target'] = df_hits['event_action'].apply(lambda x: 1 if x in target_list else 0)
    df_hits = df_hits.drop(columns=['event_action'])
    group = df_hits.groupby('session_id').max()

    df = pd.merge(left=group, right=df_sessions, on='session_id', how='inner')

    print('SberAuto Prediction Pipeline')

    x = df.drop('target', axis=1)
    y = df['target']

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor_transform = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(clean_data)),
        ('featuring', FunctionTransformer(featuring)),
        ('transformer', preprocessor_transform),
    ])

    logreg = LogisticRegression(solver='liblinear', max_iter=200, C=3.0)

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', logreg)
    ])

    score = cross_val_score(pipe, x, y, cv=4, scoring='roc_auc')
    print(score)

    pipe.fit(x, y)

    with open('sber_auto_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                "name": "Sber auto model",
                "author": "Kokidko",
                "version": 1,
                "date": datetime.now(),
                "type": type(pipe.named_steps["classifier"]).__name__,
                "accuracy": score
            }
        }, file, recurse=True)


if __name__ == '__main__':
    main()
