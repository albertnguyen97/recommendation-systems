import pandas as pd
import numpy as np
from ast import literal_eval

df = pd.read_csv('datasets/movies_metadata.csv')
# ----------------BASIC---------------------------------
# df.head()
#
# m = df['vote_count'].quantile(0.70)
# q_movies = df[(df['runtime'] >= 60) & (df['runtime'] <= 300)]
# q_movies = q_movies[q_movies['vote_count'] >= m]
# print(q_movies.shape)
# c = df['vote_average'].mean()
# print(c)  # 5.6/10
#
#
# def weighted_rating(x, m=m, c=c):
#     v = x['vote_count']
#     R = x['vote_average']
#     return (v / (v + m) * R) + (m / (m + v) * c)
#
#
# q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
#
# q_movies = q_movies.set_index('title')
# q_movies = q_movies.sort_values('score', ascending=False)
# -------------------------------------------------

df = df[['title', 'genres', 'release_date', 'runtime', 'vote_average', 'vote_count']]

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

df['year'] = df['release_date'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)


def convert_int(x):
    try:
        return int(x)
    except:
        return 0


df['year'] = df['year'].apply(convert_int)
df = df.drop('release_date', axis=1)

print(df.head())
print(df.columns)

df['genres'] = df['genres'].fillna('[]')
df['genres'] = df['genres'].apply(literal_eval)

df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

s = df.apply(lambda x: pd.Series(x['genres']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'

gen_df = df.drop('genres', axis=1).join(s)
gen_df.head()


def build_chart(gen_df, percentile=0.8):
    print("input preferred genre")
    genre = input()
    print("input shortest duration")
    low_time = int(input())
    print("Input longest duration")
    high_time = int(input())
    print("Input earliest year")
    low_year = int(input())
    print("Input latest year")
    high_year = int(input())
    movies = gen_df.copy()
    movies = movies[(movies['genre'] == genre) &
                    (movies['runtime'] >= low_time) &
                    (movies['runtime'] <= high_time) &
                    (movies['year'] >= low_year) &
                    (movies['year'] <= high_year)
                    ]
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(percentile)

    q_movies = movies.copy().loc[movies['vote_count'] >= m]
    q_movies['score'] = q_movies.apply(lambda x: (x['vote_count'] / (x['vote_count'] + m) * x['vote_average'])
                                                 + (m / (m + x['vote_count']) * C)
                                       , axis=1)
    q_movies = q_movies.sort_values('score', ascending=False)

    return q_movies


# build_chart(gen_df).head()
df.to_csv('datasets/metadata_clean.csv', index=False)

