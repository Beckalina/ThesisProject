from __future__ import division
import sys
import configparser
import argparse
import h5py as h5
import gzip
import pickle
import numpy as np
sys.path.append('../Utils')
from utils import print_timestamped_message


def main(config, model):
    dsgv_home = config.get('DSGV-PATHS', 'dsgv_home')
    preproc_path = dsgv_home + '/Preproc/PreprocOut/'
    feats_path = dsgv_home + '/ExtractFeats/ExtractOut/'
    model_path = dsgv_home + '/WACs/ModelsOut/'

    ID_FEATS = 3

    # Image features
    with h5.File(feats_path + 'saiapr_bbdf_rsn50-max.hdf5') as f:
        X = np.array(f["img_feats"])
    print('X shape:', X.shape)

    # EN WACs
    with gzip.open(model_path + model + '.pklz', 'r') as rf:
        en_wacs = pickle.load(rf)
    print('EN WACs length:', len(en_wacs))

    # Get activations
    print_timestamped_message('Getting L1 activations for feature vectors', indent=4)
    X_fts = X[:, ID_FEATS:]
    activs = np.zeros((len(en_wacs), len(X_fts)))

    for i, w in enumerate(en_wacs.keys()):
        prob = np.array(en_wacs[w]['clsf'].predict_proba(X=X_fts)[:, 1])
        activs[i] = prob
        print('.', end='')

    # Add ID feats to start of arrays
    l1 = np.concatenate((X[:, :ID_FEATS], activs.T), axis=1, dtype='<f8')
    print('l1 shape:', l1.shape)

    print_timestamped_message('Saving L1 activation vectors', indent=4)
    outfile = feats_path + 'L1_' + model + ".hdf5"
    with h5.File(outfile, 'w') as f:
        f.create_dataset('l1_feats', data=l1)


#
#
#
# ======== MAIN =========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get L1 activations for image features')
    parser.add_argument('-m', '--model',
                        help='Which L1 model to use.',
                        default='wac_EN_3')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    with open('../Config/default.cfg', 'r', encoding='utf-8') as f:
        config.read_file(f)

    main(config, args.model)