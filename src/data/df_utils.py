# FUNCTIONS AND UTILITIES FOR DATAFRAME HANDLING

# standard libraries
import glob
import logging
import os

# third party libraries
import pandas as pd

# set logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s[%(name)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)


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
            logger.warning(f'Entries {weird_names} in column \'{col}\' contain some non-ascii characters.')
            output = False

    return output


def create_dataframe(data_dir: str):

    # basic data fetch
    image_filenames = glob.glob(os.path.join(data_dir, '**/*.jpg'),
                                recursive=True)
    # create a bare data info df
    df = pd.DataFrame({'filepath': image_filenames})
    df['artist'] = df['filepath'].apply(lambda x: x.split(os.path.sep)[-2].replace('_', ' '))

    logger.info(f'Created dataframe ({len(df)} rows) from data found in {data_dir}.')

    return df


if __name__ == '__main__':
    DATA_DIR = '../../data/images'
    df = create_dataframe(DATA_DIR)
    check_ascii_conformity(df)

    print(df.head())
