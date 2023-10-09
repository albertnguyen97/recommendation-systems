import numpy as np
import pandas as pd

df = pd.read_csv('datasets/movies_metadata.csv')
df.head()
type(df)

df.shape  # 45k films, 24 features

df.columns

second = df.iloc[1]
df = df.set_index('title')
jump = df.loc['Jumanji']
print(jump)
df = df.reset_index()  # reset index
small_df = df[['title', 'release_date', 'budget', 'revenue', 'runtime', 'genres']]

small_df.head()
small_df.head(15)
small_df.info()


def to_float(x):
    try:
        x = float(x)
    except:
        x = np.nan
    return x


small_df['budget'] = small_df['budget'].apply(to_float)
small_df['budget'] = small_df['budget'].astype('float')
small_df.info()


small_df['release_date'] = pd.to_datetime(small_df['release_date'], errors='coerce')

small_df['year'] = small_df['release_date'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

print(small_df.head(10))
small_df = small_df.sort_values('year')
print(small_df.head(10))
small_df = small_df.sort_values('revenue', ascending=False)
print(small_df.head(10))
new = small_df[small_df['revenue'] > 1e9]
print(new.head(5))
new2 = small_df[(small_df['revenue'] > 1e9) & (small_df['budget'] < 1.5e8)]
print(new2.head(5))
runtime = small_df['runtime']
print(runtime.max(), runtime.min())

budget = small_df['budget']
print(budget.mean(), budget.median())

revenue = small_df['revenue']
print(revenue.quantile(0.90))
small_df['year'].value_counts()