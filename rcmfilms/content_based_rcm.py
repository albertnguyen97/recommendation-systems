import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval

df = pd.read_csv('datasets/metadata_clean.csv')
cred_df = pd.read_csv('datasets/credits.csv')
key_df = pd.read_csv('datasets/keywords.csv')
cred_df.head()

og_df = pd.read_csv('datasets/movies_metadata.csv', low_memory=False)

df['overview'], df['id'] = og_df['overview'], og_df['id']
tfidf = TfidfVectorizer(stop_words='english')

df['overview'] = df['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df['overview'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()


def content_recommender(title, cosine_sim=cosine_sim, df=df, indices=indices):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices]


content_recommender('The Lion King')

df['id'] = df['id'].astype('int')


def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan


df['id'] = df['id'].apply(clean_ids)
df = df[df['id'].notnull()]
df['id'] = df['id'].astype('int')

df = df.merge(cred_df, on='id')
df = df.merge(key_df, on='id')
df.head()

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df[feature] = df[feature].apply(literal_eval)

df.iloc[0]['crew'][0]


def get_director(x):
    for crew_member in x:
        if crew_member['job'] == 'Director':
            return crew_member['name']
    return np.nan

df['director'] = df['crew'].apply(get_director)

df['director'].head()

def generate_list(x):
    if isinstance(x, list):
        names = [ele['name'] for ele in x]
        #Check if more than 3 elements exist. If yes, return only first three.
        #If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []