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
    print_timestamped_message('Starting to train model %s' % modelname)

    dsgv_home = config.get('DSGV-PATHS', 'dsgv_home')
    model_path = dsgv_home + '/WACs/ModelsOut/'
    preproc_path = dsgv_home + '/Preproc/PreprocOut/'
    feats_path = dsgv_home + '/ExtractFeats/ExtractOut/'

    out_path = model_path + modelname
    if isfile(out_path + '.npz'):
        print('%s exists. Will not overwrite. ABORTING.' % (out_path + '.npz'))
        return

    # Define classifier
    classifier = linear_model.LogisticRegression
    classf_params = {
        'penalty': 'l2',
        'C': 0.9,
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
        'wprm':  10,               # ... wprm times
        'clsf':  'logreg-l2',      # logistic regression, l2 regularized
        'params': classf_params,
        'scaled': True,
        'nneg':  2000,              # maximum neg instances
        'nsrc':  'randmax',         # ... randomly selected
        'l1_probs': 'high',         # 'all', 'high'
        'notes': ''
    }

    # ========================= DATA =================================
    print_timestamped_message('loading up data.', indent=4)

    with open(preproc_path + 'fr_splits.json', 'r') as f:
        splits = json.load(f)

    # Features
    with h5.File(feats_path + 'saiapr_bbdf_rsn50-max.hdf5') as f:
        img_fts = np.array(f["img_feats"])
    with h5.File(feats_path + 'L1_wac_EN_3.hdf5') as f:
        l1_fts = np.array(f["img_feats"])[:, ID_FEATS:]

    if model['l1_probs'] == 'high':
        high_probs = np.zeros_like(l1_fts)

        it = np.nditer(l1_fts, flags=['multi_index'])
        for p in it:
            if p > 0.75:
                idx = it.multi_index
                high_probs[idx[0]][idx[1]] = p
        l1_fts = high_probs

    X = np.concatenate((img_fts, l1_fts), axis=1)
    X_tr = filter_X_by_filelist(X, splits['train'])
    print('X_tr shape:', X_tr.shape)

    # Ref Exps
    refdf = pd.read_pickle(preproc_path + 'FR_small_dataset.pkl')
    refdf_tr = filter_refdf_by_filelist(refdf, splits['train'])
    print('refdf_tr shape:', refdf_tr.shape)
    print(refdf_tr)

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

    with gzip.open(out_path + '.pklz', 'w') as f:
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
                        default='L2wac_' + str(date.today()))
    args = parser.parse_args()

    config = configparser.ConfigParser()
    with open('../../Config/default.cfg', 'r', encoding='utf-8') as f:
        config.read_file(f)

    main(config, args.modelname)
