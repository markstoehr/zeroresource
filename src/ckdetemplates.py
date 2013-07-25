#!/usr/bin/python

import numpy as np
import argparse,itertools,sys, os
from template_speech_rec import configParserWrapper
from scipy.spatial.distance import cdist,pdist,squareform
import matplotlib.pyplot as plt



def kde_bernoulli_probs(E,sigma):
    """
      Infer a smoothed probability estimate from a single template 
      and a single parameter sigma that determines how much smoothing happens
    """
    # assume there is a single example which is E
    # time coordinates
    times = (np.arange(E.size)  / E.shape[-1])[E.ravel()==1]
    freqs = (np.arange(E.size) % E.shape[-1])[E.ravel()==1]
    edge_locs = np.vstack((times,freqs)).T

    all_locs = np.vstack((
            np.arange(E.size) / E.shape[-1],
            np.arange(E.size) % E.shape[-1])).T
    edge_dists = np.exp(- (cdist(all_locs,edge_locs)/sigma)**2).sum(-1)
    base_dists = np.exp(- (squareform(pdist(all_locs))/sigma)**2).sum(-1)  

    return (edge_dists/ base_dists).reshape(E.shape)



def main(args):
    config_d = configParserWrapper.load_settings(open(args.c,'r'))

    # load in the data
    X = np.load("%s_E.npy" % args.f)
    M = np.array([ tuple(int(e) for e in l.split()) for l in open("%s_meta.txt" % args.f,'r')])
    bgd = np.clip(np.load(args.b),
                  config_d['TEMPLATE_TRAINING']['bgd_min_prob'],
                  config_d['TEMPLATE_TRAINING']['bgd_max_prob']).ravel()


    # prepare the background portion of the log likelihood ratio filter
    bgd_inv_log = np.log(1-bgd)
    bgd_log = np.log(bgd)
    bgd_norm_term = np.sum(bgd_inv_log)
    bgd_log_odds = bgd_log - bgd_inv_log
   
    # construct the model portion of the log likelihood ratio filter
    models = np.zeros(X.shape)
    
    for example_id, example_stats in enumerate(M):
        start,end,word_id  = example_stats
        for edge_id in xrange(X.shape[-1]):
            models[start:end,:,edge_id] = kde_bernoulli_probs(
                X[start:end,:,edge_id],
                config_d['TEMPLATE_TRAINING']['bandwidth'])
    
    models = np.clip(models,
                         config_d['TEMPLATE_TRAINING']['min_probability'],
                         1-config_d['TEMPLATE_TRAINING']['min_probability'])

    models = models.reshape(*((len(models),np.prod(models.shape[1:]))))
    models_inv_log = np.log(1-models)
    model_norm_terms = models_inv_log.sum(1)
    model_log_odds = np.log(models) - models_inv_log
    log_ratio_filter = model_log_odds - bgd_log_odds
    log_ratio_const = model_norm_terms - bgd_norm_term

    bgd_response = (log_ratio_filter * bgd).sum(1) + log_ratio_const
    np.save('%s_filter.npy' % args.o,log_ratio_filter)
    np.save('%s_const.npy' % args.o,log_ratio_const)
    np.save('%s_bgd_response.npy' % args.o,bgd_response)
    np.save('%s_meta.npy' %args.o,M)        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""
Template Estimation function, takes as input extracted features
and outputs models, basic function is to get the templates via the KDE
smoothing of single examples.
We also construct templates for computing
""")
    parser.add_argument('-b',type=str,help="path to background file")
    parser.add_argument('-f',type=str,help="path to the input utterance file")
    parser.add_argument('-c',type=str,help='path to the config file')
    parser.add_argument('-o',type=str,help='path to the output file')
    main(parser.parse_args())

