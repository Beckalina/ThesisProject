# coding: utf-8
"""
Train a (set of) WAC models. Modified template for a specific parameter
setting.
"""

from __future__ import division
import sys
import argparse
import configparser
import json
import h5py as h5
import pickle
import gzip
from os.path import isfile
from joblib import Parallel, delayed
from datetime import date
import numpy as np
import pandas as pd
from sklearn import linear_model

sys.path.append('../../Utils')
from utils import print_timestamped_message
sys.path.append('../WAC_Utils')
from wac_utils import filter_X_by_filelist, filter_refdf_by_filelist
from wac_utils import filter_relational_expr
from wac_utils import create_word2den, make_X_id_index, make_mask_matrix
from wac_utils import train_this_word, get_X_for_word

# The first features in the image feature Xs encode the region ID
ID_FEATS = 3

N_JOBS = 2  # how many threads to run in parallel during training


def main(config, modelname):
    #basename = os.path.splitext(os.path.basename(__file__))[0]
    print_timestamped_message('Starting to train model %s'
                              % (modelname))

    outfile_base = config.get('runtime', 'out_dir') + '/' + modelname
    if isfile(outfile_base + '.npz'):
        print('%s exists. Will not overwrite. ABORTING.' % (outfile_base + '.npz'))
        return

    dsgv_home = config.get('DSGV-PATHS', 'dsgv_home')
    preproc_path = dsgv_home + '/Preproc/PreprocOut/'
    feats_path = dsgv_home + '/ExtractFeats/ExtractOut/'

    # Define classifier
    classifier = linear_model.LogisticRegression
    classf_params = {
        'penalty': 'l2',
        'warm_start': True,
        'solver': 'lbfgs',
        'max_iter': 500
    }

    # Model description:
    model = {
        'rcorp': 'referit',        # ref corpus
        'cnn': 'rsn50-max',        # CNN used for vision feats
        'rel':   'excl',           # exclude relational expressions
        'wrdl':  'min',            # wordlist: minimal n occurrences...
        'wprm':  40,               # ... 40 times
        'clsf':  'logreg-l1',      # logistic regression, l1 regularized
        'params': classf_params,
        'scaled': True,
        'nneg':  2000,            # maximum neg instances
        'nsrc':  'randmax',        # ... randomly selected
        'notes': ''
    }

    # ========================= DATA =================================
    print_timestamped_message('loading up data.', indent=4)

    with open(preproc_path + 'fr_splits.json', 'r') as f:
        splits = json.load(f)

    # Image features
    with h5.File(feats_path + 'saiapr_bbdf_rsn50-max.hdf5') as f:
        X = np.array(f["img_feats"])
    X_tr = filter_X_by_filelist(X, splits['train'])
    print('X_tr shape:', X_tr.shape)

    refdf = pd.read_pickle(preproc_path + 'FR_small_dataset.pkl')
    print('refdf shape:', refdf.shape)

    refdf_tr = filter_refdf_by_filelist(refdf, splits['train'])
    print('Training dataset:\n', refdf_tr)

    # ======================= Intermediate ==============================
    print_timestamped_message('creating intermediate data structures', indent=4)

    word2den = create_word2den(refdf_tr)
    X_idx = make_X_id_index(X_tr)
    mask_matrix = make_mask_matrix(X_tr, X_idx, word2den, word2den.keys())

    # ======================= Wordlist ==============================
    print_timestamped_message('selecting words to train models for', indent=4)

    min_freq = model['wprm']
    counts = mask_matrix.sum(axis=1)
    wordlist = np.array(list(word2den.keys()))[counts > min_freq]

    # ======================= TRAIN ==============================
    print_timestamped_message('and training the %d WACs!' % (len(wordlist)),
                              indent=4)

    wacs = Parallel(n_jobs=N_JOBS, require='sharedmem', prefer='threads')\
                   (delayed(train_this_word)(X_tr, word2den, mask_matrix,
                                             model['nneg'],
                                             classifier, classf_params,
                                             this_word)
                    for this_word in wordlist)
    print('\nTraining complete!')

    # ======================= SAVE ==============================
    print('')
    print_timestamped_message('writing to disk', indent=4)

    clsf = {}
    for wd, npos, n, wac in wacs:
        clsf[wd] = {'npos': npos,
                    'n': n,
                    'clsf': wac}

    with gzip.open(outfile_base + '.pklz', 'w') as f:
        pickle.dump(clsf, f)

    print_timestamped_message('DONE!')


#
#
#
# ======== MAIN =========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train French WAC model(s)')
    parser.add_argument('-m', '--modelname',
                        help='name of model.',
                        default='wac_FR_' + str(date.today()))
    args = parser.parse_args()

    config = configparser.ConfigParser()
    with open('../../Config/default.cfg', 'r', encoding='utf-8') as f:
        config.read_file(f)

    if config.has_option('DSGV-PATHS', 'train_out_dir'):
        out_dir = config.get('DSGV-PATHS', 'train_out_dir')
    else:
        out_dir = '../ModelsOut'

    config.add_section('runtime')
    config.set('runtime', 'out_dir', out_dir)

    main(config, args.modelname)
