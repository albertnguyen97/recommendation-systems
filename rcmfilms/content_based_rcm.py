import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


df = pd.read_csv('datasets/metadata_clean.csv')


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
