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
from wac_utils import filter_refdf_by_filelist, is_relational, create_word2den
#from augment_model import compute_confidences


def main(config, model):
    outfilename = config.get('runtime', 'out_dir') + '/' + model + '_results'
    if os.path.isfile(outfilename + '.pklz'):
        print('Outfile (%s) exists. Better check before I overwrite anything!' % (outfilename + '.pklz'))
        exit()

    dsgv_home = config.get('DSGV-PATHS', 'dsgv_home')
    model_path = dsgv_home + '/WACs/ModelsOut/'
    preproc_path = dsgv_home + '/Preproc/PreprocOut/'
    feats_path = dsgv_home + '/ExtractFeats/ExtractOut/'

    results = []

    # ------ DATA ------
    print_timestamped_message('Loading up data. This may take some time.')

    # Image features
    with h5.File(feats_path + 'saiapr_bbdf_rsn50-max.hdf5') as rf:
        X = np.array(rf["img_feats"])
    print('First image features X[0]:\n', X[0])

    # Bounding boxes
    s_bbdf = pd.read_json(preproc_path + 'saiapr_bbdf.json.gz',
                          typ='frame', orient='split', compression='gzip')
    print('First bounding box s_bbdf[0]:\n', s_bbdf.loc[0])

    # Ref Exp test split
    with open(preproc_path + 'saiapr_90-10_splits.json', 'r') as rf:
        ssplit90 = json.load(rf)

    srefdf = pd.read_json(preproc_path + 'saiapr_refdf.json.gz',
                          typ='frame', orient='split', compression='gzip')
    testdf = filter_refdf_by_filelist(srefdf, ssplit90['test'])
    print('Test set length: ', len(testdf))
    print('First test entry testdf[0]:\n', testdf.loc[0])

    # ------ MODEL ------
    print_timestamped_message('Loading model ' + model)

    with gzip.open(model_path + model + '.pklz', 'r') as rf:
        wacs = pickle.load(rf)

    """wac_weights = np.load(model_path + model + '.npz')['arr_0']
    print('WAC weights matrix shape:', wac_weights.shape)

    wac_clsfs = []
    for ws in wac_weights:
        classifier = linear_model.LogisticRegression
        classifier.coef_ = ws[:-1]
        classifier.intercept_ = ws[-1]
        wac_clsfs.append(classifier)

    with open(model_path + model + '.json', 'r') as f:
        wac_words = json.load(f)[1]
    #wac_words = pd.DataFrame(wac_words, columns=['word', 'npos', 'n'])

    wacs = {}
    for w, clsf in zip(wac_words, wac_clsfs):  # wac_weights):
        wacs[w[0]] = {'npos': w[1], 'n': w[2], 'clsf': clsf}
    print('WAC info & weights for word \'hat\':', wacs['hat'])"""

    # ------ EVALUATION ------
    """
    Given an image I together with bounding boxes of regions (bb1; : : : ; bbn) within it, and a referring
    expression e, predict which of these regions contains the referent of the expression.
    """
    print_timestamped_message('Evaluating...')
    results = eval_testdf(testdf, wacs, X)
    print('Results:\n', results)

    summary = summarise_eval(results)
    print('Summary:\n', summary)

    # Save
    with gzip.open(outfilename + '.pklz', 'w') as rf:
        pickle.dump(results, rf)

    with open('results_summaries.txt', 'a') as sf:
        sf.write('[{}]\n{}\n'.format(datetime.now(), summary))

    print_timestamped_message('Done!')


#
#
#
# ======== MAIN =========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test WAC model(s)')
    parser.add_argument('-c', '--config_file',
                        help='''
                        path to config file specifying data paths.
                        default: '../Config/default.cfg' ''',
                        default='../Config/default.cfg')
    parser.add_argument('-m', '--model',
                        help='model to evaluate.')
    parser.add_argument('-n', '--name',
                        help='''
                        name to give results file.
                        default: <model name> + '_results' ''')
    parser.add_argument('-o', '--out_dir',
                        help='''
                        where to put the resulting files.
                        default: './EvalOut' ''',
                        default='./EvalOut')
    args = parser.parse_args()

    config = configparser.ConfigParser()

    try:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config.read_file(f)
    except IOError:
        print('no config file found at %s' % (args.config_file))
        sys.exit(1)

    if not args.model:
        print('please specify a model to evaluate, e.g. python run_evals.py -m my_model')
        sys.exit(1)
    else:
        model = args.model

    config.add_section('runtime')
    config.set('runtime', 'out_dir', args.out_dir)

    main(config, model)