# FUNCTIONS AND UTILITIES FOR DATAFRAME HANDLING

# standard libraries
import glob
import logging
import os

# third party libraries
import pandas as pd

# set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s[%(name)s][%(levelname)s]: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def check_ascii_conformity(df: pd.DataFrame):

    def check_one_col(column: pd.Series):
        # check for weird characters in artist name
        weird_entries = set()
        for entry in column:
            if any([not 0 <= ord(c) <= 127 for c in entry]):
                weird_entries.add(entry)
        return weird_entries

    output = True

    for col in df.columns:
        weird_names = check_one_col(df[col])
        if weird_names:
            logger.warning(f'Dataframe entries {weird_names} in column \'{col}\' contain some non-ascii characters.')
            output = False

    return output


def create_dataframe(data_dir: str, skip_files: list[str] = None):

    # basic data fetch
    image_filenames = sorted(glob.glob(os.path.join(data_dir, '**/*.jpg'),
                                       recursive=True))

    # for broken jpgs
    if skip_files:
        image_filenames = [filename for filename in image_filenames
                           if not any([skip_file in filename for skip_file in skip_files])]

    # create a bare data info df
    df = pd.DataFrame({'filepath': image_filenames})
    df['artist'] = df['filepath'].apply(lambda x: x.split(os.path.sep)[-2].replace('_', ' '))

    logger.info(f'Created dataframe ({len(df)} rows) from data found in {data_dir}.')

    return df


if __name__ == '__main__':
    DATA_DIR = '../../data'
    img_dir = os.path.join(DATA_DIR, 'images')
    # broken jpgs
    broken_jpgs = ['Edgar_Degas_216.jpg']
    df = create_dataframe(img_dir, skip_files=broken_jpgs)
    check_ascii_conformity(df)

    print(df.head())
