import configparser
import sys
import json
import pandas as pd

sys.path.append('../../WACs/WAC_Utils')
from wac_utils import filter_X_by_filelist, filter_refdf_by_filelist
from wac_utils import filter_relational_expr


def get_refexp_to_translate(preproc_path, split=None):
    refdf = pd.read_json(preproc_path + 'PreprocOut/saiapr_refdf.json.gz', typ='frame',
                         orient='split', compression='gzip')

    if split:
        with open(preproc_path + 'PreprocOut/saiapr_90-10_splits.json', 'r') as f:
            s_splits = json.load(f)

        refdf_split = filter_refdf_by_filelist(refdf, s_splits[split])
        refdf = filter_relational_expr(refdf_split)

    refdf.to_csv(preproc_path + 'Translate/' + split + 'set_to_translate.csv',
                 sep='\t', columns=['refexp'])


config_file = '../../Config/default.cfg'
config = configparser.ConfigParser()
with open(config_file, 'r', encoding='utf-8') as f:
    config.read_file(f)

dsgv_home = config.get('DSGV-PATHS', 'dsgv_home')
preproc_path = dsgv_home + '/Preproc/'

# Select split: 'train'/'test'/None
split = 'train'

get_refexp_to_translate(preproc_path, split)
