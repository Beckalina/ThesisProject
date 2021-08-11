# coding: utf-8

from __future__ import division
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats
from scipy.special import expit
import sys

sys.path.append('../WACs/WAC_Utils')
from wac_utils import is_relational, intersectbb, intoveru


# number of ID features at the beginning of X so that [POSTIDINDEX:] indexes the true features
POSTIDINDEX = 3
STOPWORDS = []

'''Some functions for evaluating on region proposals. Adding the bounding 
boxes to the refdf, pre-computing intersection over union, etc.'''


def get_gold_bbs(refdf, bbdf):
    gbbs = []
    for n, row in refdf.iterrows():
        this_i_corp = row['i_corpus']
        this_image_id = row['image_id']
        this_region_id = row['region_id']
        gbb = bbdf[(bbdf['i_corpus'] == this_i_corp) &\
                   (bbdf['image_id'] == this_image_id) &\
                   (bbdf['region_id'] == this_region_id)]['bb'].tolist()[0]
        gbbs.append(gbb)
    return gbbs


def get_rprp_bbs(filelist, bbdf):
    outrows = []
    for this_image_id in filelist:
        all_bbs = bbdf[bbdf['image_id'] == this_image_id]['bb'].tolist()
        outrows.append((this_image_id, all_bbs))
    return pd.DataFrame(outrows, columns='image_id bbs'.split())


def apply_iou_to_refdf_row(row, rprp_df):
    this_bblist = rprp_df[rprp_df['image_id'] == row['image_id']]['bbs'].values[0]
    this_gbb = row['gbb']
    #print this_bblist
    return map(lambda x:intoveru(this_gbb, x), this_bblist)


'''The following work uniformly for gold regions and region proposals.'''


def make_test(X, region_spec):
    icorp, image_id, region_id = region_spec
    X_tst = X[np.logical_and(X[:,0] == icorp,
                             X[:,1] == image_id)]
    if X_tst.shape[0] == 0:
        return None, None

    correct_index = np.nan
    if region_id is not None:
        correct_index = np.where(X_tst[:,2] == region_id)[0]
        if len(correct_index) > 0:
            correct_index = correct_index[0]

    X_tst = X_tst[:, POSTIDINDEX:]  # actual features, w/o the ids

    return X_tst, correct_index


def reduce_refexp(wac, refexp_toks):
    refexp_toks_stop = [w for w in refexp_toks if w not in STOPWORDS]
    refexp_toks_wac = [w for w in refexp_toks_stop if w in wac.keys()]

    if len(refexp_toks_stop) == 0:
        coverage = 0
    else:
        coverage = len(refexp_toks_wac) / len(refexp_toks_stop)
    return refexp_toks_wac, coverage


# util function to measure classifier certainty
def p(X, w):
  return expit(np.dot(X, w))


def apply_words(X_tst, wac, refexp_toks_wac, conf_threshold=None):
    response_vectors = []
    for word in refexp_toks_wac:
        response_vector = np.array(wac[word]['clsf'].predict_proba(X=X_tst)[:,1])
        if 'conf' in wac[word]:  # classifier confidences scores - if there, use them
            # conf score thrown in logistic function:
            # response_vector *= expit(wac[word]['conf'])
            # flatten out the distribution in proportion to (in)confidence
            # response_vector += 1/wac[word]['conf']
            # response_vector /= response_vector.sum()
            if conf_threshold is not None:
                if wac[word]['conf'] < conf_threshold:
                    response_vector = np.ones(response_vector.shape[0])
        response_vectors.append(response_vector)
    response_matrix = np.array(response_vectors).T   # n_objs x n_words
    return response_matrix


def apply_refexp_to_image(row, wac, X, compfunc=lambda x:np.sum(x, axis=1),
                          conf_thresh=None, restrict_pos=None):
    icorp = row['i_corpus']
    image_id = row['image_id']
    region_id = row['region_id']
    refexp = row['refexp']
    
    refexp_toks = refexp.split()  # highly sophisticated tokenization...

    if restrict_pos is not None:
        refexp_toks = [w for (w,pos) in row['tagged'] if pos in restrict_pos]

    refexp_toks_wac, coverage = reduce_refexp(wac, refexp_toks)
    if coverage == 0:
        return 0, False, np.nan, np.nan

    if 'lexrel' in wac[refexp_toks_wac[0]]:  # Extend expression with related words, if field exists
        out = []
        for this_token in refexp_toks_wac:
            out.append(this_token)
            out.extend(wac[this_token]['lexrel'])
        refexp_toks_wac = out

    # A bit of a hack (again)... If we're evaluating on region  proposals, then the X icorp and the
    # refdf icorp are different. To not confuse make_test, we set icorp to that of X[0] here.
    if 'ious' in row.index:
        icorp = X[0,0]
    
    X_tst, correct_ix = make_test(X, (icorp, image_id, region_id))
    if X_tst is None:
        return np.nan, np.nan, np.nan, np.nan
    
    response_matrix = apply_words(X_tst, wac, refexp_toks_wac, conf_thresh)

    composed_vector = compfunc(response_matrix)  # compfunc = sum/multiply/harmonic mean

    # are we evaluating on region proposals here?
    if 'ious' in row.index:
        #ious = np.array(row['ious'])
        #rank = ious[np.argsort(composed_vector)[::-1]]
        rank = np.argsort(composed_vector)[::-1]
        success = False
    else:
        success = np.argmax(composed_vector) == correct_ix
        rank = np.where(np.argsort(composed_vector)[::-1] == correct_ix)[0][0] + 1
    return coverage, success, rank, len(composed_vector)


def multiply_apply(row, wac, X, **kwargs):
    return apply_refexp_to_image(row, wac, X,
                                 compfunc=lambda x:np.multiply.reduce(x, axis=1), **kwargs)


def hmean_apply(row, wac, X, **kwargs):
    return apply_refexp_to_image(row, wac, X,
                                 compfunc=lambda x:scipy.stats.hmean(x, axis=1), **kwargs)


def apply_areabaseline_to_image(row, wac, X, compfunc=lambda x:np.sum(x, axis=1)):
    '''Baseline: Just pick biggest objects (largest area)'''
    icorp = row['i_corpus']
    image_id = row['image_id']
    region_id = row['region_id']
        
    X_tst, correct_ix = make_test(X, (icorp, image_id, region_id))
    if X_tst is None:
        return np.nan, np.nan, np.nan, np.nan

    area_vector = X_tst[:,-3]
    biggest = np.argmax(area_vector)
    
    success = biggest == correct_ix
    rank = np.where(np.argsort(area_vector)[::-1] == correct_ix)[0][0] + 1
    #return correct_ix, coverage, response_matrix, composed_vector
    return 1.0, success, rank, len(area_vector)


# 'main' evaluation function
def eval_testdf(testdf, wac, X, applyfunc=apply_refexp_to_image, **kwargs):
    results = []
    for n, row in tqdm(testdf.iterrows(), total=len(testdf)):
        results.append(applyfunc(row, wac, X, **kwargs))  # usually apply_refexp_to_image
    outdf = pd.concat([testdf, pd.DataFrame(results, columns='cov suc rnk nob'.split())], axis=1)
    outdf['is_rel'] = outdf['refexp'].apply(is_relational)
    # if outdf.isnull().values.any():
    #     print 'Contains NaNs. Consider dropping those before summarising!'
    return outdf


def mrr_f(series):
    return np.mean(series.apply(lambda x:(1/x)))


def acc_f(series):
    return np.count_nonzero(series) / len(series)


def summarise_eval(edf, asdf=True, col=None):
    acc = acc_f(edf['suc'])
    mrr = mrr_f(edf['rnk'])
    acv = edf['cov'].mean()
    rnd = mrr_f(edf['nob'])
    if asdf:
        return pd.DataFrame([acc, mrr, acv, rnd],
                            index='acc mrr acv rnd'.split(),
                            columns=col).T
    return acc, mrr, acv, rnd


