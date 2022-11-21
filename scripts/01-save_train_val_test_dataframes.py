# SAVE DATAFRAME ONCE FOR TRAINING SCRIPT

# standard imports
import logging
import os

# third party libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# custom libraries
from src.data import df_utils

# set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s[%(name)s][%(levelname)s]: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

DATA_DIR: str = '../data'


# load the raw data info df
print('Loading raw data dataframe...')
img_dir = os.path.join(DATA_DIR, 'images')
# broken jpgs
broken_jpgs = ['Edgar_Degas_216.jpg']
df = df_utils.create_dataframe(img_dir, skip_files=broken_jpgs)
df_utils.check_ascii_conformity(df)
# there is a problem with one of the artists with an umlaut...
df.loc[df['artist'].str.contains('Albrecht'), 'artist'] = 'Albrecht Durer'

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
artist_to_id = {x: i for i, x in enumerate(keep_artists)}
df['id'] = df['artist'].apply(lambda x: artist_to_id[x])

print(f'Number of unique artists: {len(keep_artists)}')
print(f'Dataframe:\n{df.head()}')

# partition into train, validation and test, stratify by artist
train_df, val_test_df = train_test_split(df,
                                         test_size=0.2,
                                         random_state=42,
                                         stratify=df['artist'])
validation_df, test_df = train_test_split(val_test_df,
                                          test_size=0.5,
                                          random_state=42,
                                          stratify=val_test_df['artist'])

train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
validation_df.to_csv(os.path.join(DATA_DIR, 'validation.csv'), index=False)
test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)
