# coding: utf-8

"""
Run the models on the data sets.
"""

# TODO: Need to write out which region was selected, so that I can do
#  error analysis later!

from __future__ import division
import sys
import argparse
import configparser
import gzip
import pickle
import json
import h5py as h5
import os
from datetime import datetime

from apply_model import *
sys.path.append('../Utils')
from utils import print_timestamped_message
from data_utils import load_dfs
from apply_utils import apply_wac_set_matrix, logreg
sys.path.append('../WACs/WAC_Utils')
from wac_utils import filter_refdf_by_filelist, is_relational, create_word2den, filter_relational_expr
#from augment_model import compute_confidences


def get_en_refexp(path):
    with open(path + 'saiapr_90-10_splits.json', 'r') as f:
        ssplit90 = json.load(f)

    srefdf = pd.read_json(path + 'saiapr_refdf.json.gz',
                          typ='frame', orient='split', compression='gzip')
    split_df = filter_refdf_by_filelist(srefdf, ssplit90['test'])
    split_df = filter_relational_expr(split_df, lang)
    return split_df


def get_fr_refexp(path, split='val'):
    with open(path + 'fr_splits.json', 'r') as f:
        splits = json.load(f)

    refdf = pd.read_pickle(path + 'FR_small_dataset.pkl')
    split_df = filter_refdf_by_filelist(refdf, splits[split])
    return split_df


def main(config, model, lang, description, split=None):
    outfilename = './EvalOut/' + model + '_results'
    if os.path.isfile(outfilename + '.pklz'):
        print('Outfile (%s) exists. Better check before I overwrite anything!' % (outfilename + '.pklz'))
        exit()

    dsgv_home = config.get('DSGV-PATHS', 'dsgv_home')
    model_path = dsgv_home + '/WACs/ModelsOut/'
    preproc_path = dsgv_home + '/Preproc/PreprocOut/'
    feats_path = dsgv_home + '/ExtractFeats/ExtractOut/'

    # ------ DATA ------
    print_timestamped_message('Loading up data.')

    if lang == 'EN':
        evaldf = get_en_refexp(preproc_path)
    elif lang == 'FR':
        evaldf = get_fr_refexp(preproc_path, split)
    else:
        print('Sorry, language {} is not yet supported.'.format(lang))
        exit()
    print('Evaluation dataset:\n', evaldf)

    # Image features
    with h5.File(feats_path + 'saiapr_bbdf_rsn50-max.hdf5') as rf:
        X = np.array(rf["img_feats"])
    print('First image features X[0]:\n', X[0])

    # Bounding boxes
    s_bbdf = pd.read_json(preproc_path + 'saiapr_bbdf.json.gz',
                          typ='frame', orient='split', compression='gzip')
    print('First bounding box s_bbdf[0]:\n', s_bbdf.loc[0])

    # ------ MODEL ------
    print_timestamped_message('Loading model ' + model)

    with gzip.open(model_path + model + '.pklz', 'r') as rf:
        wacs = pickle.load(rf)

    # ------ EVALUATION ------
    """
    Given an image I together with bounding boxes of regions (bb1; : : : ; bbn) within it, and a referring
    expression e, predict which of these regions contains the referent of the expression.
    """
    print_timestamped_message('Evaluating...')
    results = eval_testdf(evaldf, wacs, X)
    print('Results:\n', results)

    summary = summarise_eval(results)
    print('Summary:\n', summary)

    # Save
    if split == 'val':
        with open('val_summaries.txt', 'a') as sf:
            sf.write('\n[{}] {} - {}\n{}\n'.format(datetime.now(), model, description, summary))

    else:
        with gzip.open(outfilename + '.pklz', 'w') as rf:
            pickle.dump(results, rf)

        with open('results_summaries.txt', 'a') as sf:
            sf.write('\n[{}] {} - {}\n{}\n'.format(datetime.now(), model, description, summary))

    print_timestamped_message('Done!')


#
#
#
# ======== MAIN =========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test WAC model(s)')
    parser.add_argument('-m', '--model',
                        help='Model to evaluate.')
    parser.add_argument('-l', '--lang',
                        help='Language. Languages currently supported: EN, FR',
                        choices=['EN', 'FR'])
    parser.add_argument('-s', '--split',
                        help='FR dataset split to test on: val or test',
                        choices=['val', 'test', None])
    args = parser.parse_args()

    if not args.model:
        model = input('specify model:')
    else:
        model = args.model

    if not args.lang:
        lang = input('specify language:')
    else:
        lang = args.lang

    if lang == 'FR':
        if not args.split:
            split = 'val'
        else:
            split = args.lang
    else:
        split = None

    description = input('enter description for summary log:')

    config = configparser.ConfigParser()
    with open('../Config/default.cfg', 'r', encoding='utf-8') as f:
        config.read_file(f)

    main(config, model, lang, description, split)
