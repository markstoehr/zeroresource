#!/usr/bin/python

import numpy as np
import argparse, collections,itertools
import template_speech_rec.edge_signal_proc as esp
from template_speech_rec import configParserWrapper
from scipy.io import wavfile


def reorg_part_for_fast_filtering(part,feature_types=8):
    """
    Assumes the patch for different edge types have been vertically stacked
    and that there are eight edge types
    dimensions are features by time
    want time by feature by edge type
    """
    H = part.shape[0]/feature_types
    return np.array([
            part[i*H:(i+1)*H].T
            for i in xrange(feature_types)]).swapaxes(0,1).swapaxes(1,2)


class AverageBackground:
    def __init__(self):
        self.num_frames = 0
        self.processed_frames = False
    # Method to add frames
    def add_frames(self,E,edge_feature_row_breaks=None,
                   edge_orientations=None,abst_threshold=None,
                   time_axis=0):
        new_E = E.copy()
        if abst_threshold is not None:
            esp._edge_map_threshold_segments(new_E,
                                             40,
                                             1,
                                             threshold=.3,
                                             edge_orientations = edge_orientations,
                                             edge_feature_row_breaks = edge_feature_row_breaks)
        if not self.processed_frames:
            self.E = np.mean(new_E,axis=time_axis)
            self.processed_frames = True
        else:
            self.E = (self.E * self.num_frames + np.sum(new_E,axis=time_axis))/(self.num_frames+new_E.shape[time_axis])
        self.num_frames += new_E.shape[time_axis]

def compute_edge_features(x,config_d):
    S, sample_mapping, sample_to_frames =  esp.get_spectrogram_features(x.astype(float)/(2**15-1),
                                                                        config_d['SPECTROGRAM']['sample_rate'],
                                                                        config_d['SPECTROGRAM']['num_window_samples'],
                                                                        config_d['SPECTROGRAM']['num_window_step_samples'],
                                                                        config_d['SPECTROGRAM']['fft_length'],
                                                                        config_d['SPECTROGRAM']['freq_cutoff'],
                                                                        config_d['SPECTROGRAM']['kernel_length'],
                                                                        preemph=config_d['SPECTROGRAM']['preemphasis'],
                                                                        no_use_dpss=config_d['SPECTROGRAM']['no_use_dpss'],
                                                                        do_freq_smoothing=config_d['SPECTROGRAM']['do_freq_smoothing'],
                                                                        return_sample_mapping=True
                                 )

    E, edge_feature_row_breaks,\
        edge_orientations = esp._edge_map_no_threshold(S.T)
    esp._edge_map_threshold_segments(E,
                                     config_d['EDGES']['block_length'],
                                     config_d['EDGES']['spread_length'],
                                     threshold=config_d['EDGES']['threshold'],
                                     edge_orientations = edge_orientations,
                                     edge_feature_row_breaks = edge_feature_row_breaks,
                                         abst_threshold=config_d['EDGES']['abst_threshold'],
                                     verbose=False)
    return reorg_part_for_fast_filtering(E).astype(np.uint8)

def main(args):
    """
    """
    config_d = configParserWrapper.load_settings(open(args.c,'r'))
    fls = open(args.f,'r').read().strip().split('\n')
    avg_bgd = AverageBackground()
    for fl in fls:
        sr, x = wavfile.read(fl)
        X = compute_edge_features(x,config_d)
        avg_bgd.add_frames(X)
        
    np.save(args.o,avg_bgd.E)


if __name__=="__main__":
    parser = argparse.ArgumentParser("""
    Extracts background model
    """)
    parser.add_argument('-c',type=str,help="path to configuration file")
    parser.add_argument('-f',type=str,help="path to .wav file containing the audio recording")
    parser.add_argument('-o',type=str,help="file path to save the background model to")
    main(parser.parse_args())

