# SAVE DATAFRAME ONCE FOR TRAINING SCRIPT

# standard imports
import os

# third party libraries
import matplotlib.pyplot as plt
import pandas as pd

# custom libraries
from src.data import df_utils


DATA_DIR = '../data'


# load the raw data info df
print('Loading raw data dataframe...')
img_dir = os.path.join(DATA_DIR, 'images')
# broken jpgs
broken_jpgs = ['/kaggle/input/best-artworks-of-all-time/images/images/Edgar_Degas/Edgar_Degas_216.jpg']
df = df_utils.create_dataframe(img_dir, skip_files=broken_jpgs)
df_utils.check_ascii_conformity(df)

# load the artist info dataframe
print('Loading artists.csv...')
csv_dir = os.path.join(DATA_DIR, 'artists.csv')
artist_df = pd.read_csv(csv_dir)
# there is a problem with one of the artists with an umlaut...
artist_df.loc[artist_df['name'].str.contains('Albrecht'), 'name'] = 'Albrecht Durer'

# check if all artists are present in the artist_df
print(f'All artists in raw df present in artist df:'
      f' {set(df["artist"].unique()).issubset(set(artist_df["name"].unique()))}')

# inspect number of paintings per artist
artist_df['paintings'].plot.hist(bins=range(0, 1000, 50))
plt.xlabel('paintings')
plt.ylabel('artists')
plt.title('Paintings distribution')
plt.show()
# there's a bit of imbalance.

# many artists without a lot of paintings... let's filter them out
keep_artists = artist_df.query('paintings >= 200')['name'].unique().tolist()

# associate each artist with an integer
df = df[df['artist'].isin(keep_artists)].copy()
integer_mapping = {x: i for i, x in enumerate(keep_artists)}
df['id'] = df['artist'].apply(lambda x: integer_mapping[x])

print(f'Number of unique artists: {keep_artists}')
print(f'Dataframe:\n{df.head()}')

df.to_csv(os.path.join(DATA_DIR, 'training_df.csv'), index=False)
