import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def print_uniqueCounts(df):
    for name in df.columns:
        print(f'{name} : {df[name].nunique()}')


def to_categorical(df, col_name):
    df[col_name] = df[col_name].astype('category')


def to_str(df, col_name):
    df[col_name] = df[col_name].astype('str')


def fprint(comment, val):
    print(f'{comment}:{val}')


def transform_data(tfidf, df, col):
    features = tfidf.transform(df[col])
    return pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())


def featurize_str(df: pd.DataFrame, col: str, inplace: bool = False):
    stemmer = SnowballStemmer("english")
    stemmed_col = f'name_{col}'
    df[stemmed_col] = df[col].map(lambda x: ' '.join(
        [stemmer.stem(y) for y in x.split(' ')]))
    tvec = TfidfVectorizer(min_df=.0025, max_df=.1,
                           stop_words='english', ngram_range=(1, 2))
    tvec.fit(df[stemmed_col].dropna())
    features = transform_data(tvec, df, stemmed_col)
    df.drop(columns=stemmed_col, inplace=True)
    if inplace:
        index_name = df.index.name
        features[index_name] = df.index.values
        features.set_index(index_name, inplace=True)
        df.drop(columns=col, inplace=True)
        features = pd.merge(df, features, left_index=True, right_index=True)
    return features


def num_encode(df):
    cat_cols = df.select_dtypes(['category']).columns
    df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)


def to_numeric(df):
    cat_cols = df.select_dtypes(['int8', 'int16', 'int64']).columns
    df[cat_cols] = df[cat_cols].apply(lambda x: x.astype('float64'))


Categories = ['Unknown', 'Free', 'Cheap', 'Average', 'Expensive', 'Luxury']


def price_category(price):
    cat = 0
    if price == 0.0:
        cat = 1
    elif price > 0.0 and price < 69.000000:
        cat = 2
    elif price > 69.000000 and price < 106.000000:
        cat = 3
    elif price > 106.000000 and price < 175.000000:
        cat = 4
    elif price > 175.000000:
        cat = 5
    return cat


def prepare_data() -> pd.DataFrame:
    # Load data
    df = pd.read_csv('./data/AB_NYC_2019.csv')
    df.set_index('id', inplace=True)

    # Data prep
    df['last_review'] = pd.to_datetime(df['last_review'], format='%Y-%m-%d')
    df['last_review'].fillna(df['last_review'].mode()[0].date())
    df['last_review_year'] = df['last_review'].dt.year
    df['last_review_month'] = df['last_review'].dt.month
    df['last_review_day'] = df['last_review'].dt.day

    to_str(df, 'name')
    to_str(df, 'host_name')

    # Set label
    df['price_category'] = df['price'].apply(lambda x: price_category(x))

    to_categorical(df, 'neighbourhood_group')
    to_categorical(df, 'minimum_nights')
    to_categorical(df, 'room_type')
    to_categorical(df, 'neighbourhood')
    to_categorical(df, 'calculated_host_listings_count')
    to_categorical(df, 'last_review_year')
    to_categorical(df, 'last_review_month')
    to_categorical(df, 'last_review_day')

    # Fill-up null values
    df['reviews_per_month'].fillna(
        df['reviews_per_month'].mean(), inplace=True)

    # Drop unneeded columns
    df.drop(columns=['last_review', 'host_id', 'price'], inplace=True)

    # numerical encoding of categorical columns
    num_encode(df)

    # Standardize numeric types
    to_numeric(df)

    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    # Text Feature engineering
    df = featurize_str(df, 'name', inplace=True)
    df = featurize_str(df, 'host_name', inplace=True)
    return df


def train_model(df: pd.DataFrame):
    # Train / test split
    y = df.pop('price_category')
    X = df
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3)

    # Resampling
    smote = SMOTE('minority')
    X_sm, y_sm = smote.fit_sample(X_train, y_train)
    X_train, y_train = X_sm, y_sm

    # Train model
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclassova',
        'num_leaves': 100,
        'num_class': 6,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=100,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    # save model to file
    gbm.save_model('model.txt')
    return gbm, X_train, X_test, y_train, y_test


def predict(X_test: pd.DataFrame, y_test, gbm: lgb.Booster):
    # predict
    pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred = []

    for x in pred:
        y_pred.append(np.argmax(x))

    # Print the precision and recall, among other metrics
    print(metrics.classification_report(
        y_test, y_pred, target_names=Categories))


if __name__ == '__main__':
    df = prepare_data()
    df = add_text_features(df)
    gbm, X_train, X_test, y_train, y_test = train_model(df)
    predict(X_test, y_test, gbm)
