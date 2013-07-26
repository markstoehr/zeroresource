#!/usr/bin/python

import numpy as np
import argparse,itertools,sys, os
from template_speech_rec import configParserWrapper

def get_utterance_speaker(s):
    """
    retrieves the speaker information form a string
    the strings are of the form
    
    models/bfeat_2717B_1084_filter.npy
    """
    broken_up_file_name = s.split('/')[-1].split('_')
    return broken_up_file_name[1],int(broken_up_file_name[2])

def main(args):
    """
    """
    log_ratio_filter = np.load('%s_filter.npy' % args.m)
    log_ratio_const = np.load('%s_const.npy' % args.m)
    bgd_response = np.load('%s_bgd_response.npy' % args.m)
    model_utterance_tag,model_speaker_id = get_utterance_speaker(args.m)
    M = np.load('%s_meta.npy' %args.m).astype(int)

    fls = open(args.d,'r').read().strip().split('\n')

    for fl_id, fl in enumerate(fls):
        meta = np.loadtxt("%s_meta.txt" % fl).astype(int)

        X = np.load("%s_E.npy" % fl).reshape(*((meta[-1][1],) + 
                                               log_ratio_filter.shape[1:]))

        fl_utterance_tag,fl_speaker_id = get_utterance_speaker(fl)
        
        
        word_scores = ()
        score_meta = np.zeros((meta.shape[0],M.shape[0],6))

        for cur_model_id, cur_model_terms in enumerate(M):
            start,end,model_word_id = cur_model_terms
            model_length = end-start

            cur_log_ratio_filter = log_ratio_filter[start:end]
            cur_log_ratio_const = log_ratio_const[start:end]
            cur_bgd_response = np.tile(bgd_response[start:end],
                                      model_length).reshape((model_length,
                                                             model_length))

            model_product = np.dot(X,cur_log_ratio_filter.T)
            W = np.zeros((np.max(meta[:,1]-meta[:,0]) + 2*np.max(M[:,1]-M[:,0]),
                          cur_bgd_response.shape[1]))
            W[:model_length] = cur_bgd_response

            for fl_entry_id, target_word_meta in enumerate(meta):
                target_start,target_end,target_word_id = target_word_meta
                target_length = target_end-target_start
                W[model_length:model_length+target_length] = model_product[target_start:target_end]
                W[model_length+target_length:
                          2*model_length+target_length] = cur_bgd_response

                # get the window for detection -- say 1/3
                # of the length
                window_radius = min(target_length,model_length)/3
                max_value = -np.inf
                max_idx = - window_radius
                for sum_idx in xrange(1-window_radius,window_radius):
                    value = np.sum(np.diagonal(W,sum_idx-model_length) + cur_log_ratio_const)
                    if value > max_value:
                        max_idx = sum_idx
                        max_value = value
                
                score_meta[fl_entry_id,cur_model_id,0] = target_word_id
                score_meta[fl_entry_id,cur_model_id,1] = fl_speaker_id
                score_meta[fl_entry_id,cur_model_id,2] = model_speaker_id
                score_meta[fl_entry_id,cur_model_id,3] = model_word_id
                score_meta[fl_entry_id,cur_model_id,4] = max_idx
                score_meta[fl_entry_id,cur_model_id,5] = max_value


        np.save('%s/%s_score_meta_model_utt_%s.npy' % (args.o,fl_utterance_tag,model_utterance_tag),
                 score_meta)
            
            

if __name__=="__main__":
    parser = argparse.ArgumentParser("""
        Takes in a template model and a list of locations for where
data files can be found and then 
""")
    parser.add_argument('-m',type=str,help="path to background file")
    parser.add_argument('-d',type=str,help="path to the input utterance file")
    
    parser.add_argument('-c',type=str,help='path to the config file')
    parser.add_argument('-o',type=str,help='path to the output file')
    main(parser.parse_args())

