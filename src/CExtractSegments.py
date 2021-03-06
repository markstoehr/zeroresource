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


def main(args):
    """
    """
    config_d = configParserWrapper.load_settings(open(args.c,'r'))
    sr, x = wavfile.read(args.f)
        
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
    E = reorg_part_for_fast_filtering(E).astype(np.uint8)

    start_times = np.array(args.t[::3],dtype=int)
    end_times = np.array(args.t[1::3],dtype=int)
    word_identities = np.array(args.t[2::3],dtype=int)

    max_length = np.max(end_times-start_times)

    E_shape = E.shape[1:]
    E = np.vstack((E,np.zeros((max_length,) + E_shape,dtype=np.uint8)))

    
    num_frames = np.sum(end_times - start_times)
    examples = np.zeros((num_frames,) + E.shape[1:],dtype=np.uint8)
    example_frames = np.zeros(len(start_times),dtype=int)



    if args.do_wave_output:
        wave_padding = 40*config_d['SPECTROGRAM']['num_window_samples']
        start_wave_times = (start_times.astype(float)/100*config_d['SPECTROGRAM']['sample_rate']  ).astype(int)
        end_wave_times = (end_times.astype(float)/100*config_d['SPECTROGRAM']['sample_rate']+wave_padding/5).astype(int)
        num_samples = np.sum(end_wave_times-start_wave_times) + wave_padding*len(end_wave_times)*2
        
        wave_examples = np.zeros(num_samples,dtype=x.dtype)
        cur_idx = 0
        y = np.random.permutation(x)
        y_front = y[:wave_padding]/8
        # pad the signal x just to make sure that we get
        # the full signal
        x = np.hstack((x,x[:wave_padding]))
        for example_id, start_end_time in enumerate(itertools.izip(start_wave_times,end_wave_times)):
            start_time, end_time = start_end_time
            ex_length = end_time-start_time
            wave_examples[cur_idx:cur_idx+wave_padding] = y_front
            cur_idx += wave_padding
            try:
                wave_examples[cur_idx:cur_idx+ex_length] = x[start_time:end_time]
            except:
                import pdb; pdb.set_trace()
            cur_idx += ex_length
            wave_examples[cur_idx:cur_idx+wave_padding] = y_front
            cur_idx += wave_padding

    # print out the meta-data-file
    fhandle = open("%s_meta.txt" % args.o,'w')
    cur_idx = 0
    for example_id, start_end_time in enumerate(itertools.izip(start_times,end_times,word_identities)):
        start_time, end_time,word_identity = start_end_time
        time_length = end_time - start_time
        example_frames[example_id] = cur_idx
        try:
            examples[cur_idx:cur_idx+time_length] = E[start_time:end_time]
        except: import pdb; pdb.set_trace()
        out_str = str(cur_idx)
        cur_idx += time_length
        out_str += " %d %d\n" % (cur_idx,word_identity)
        fhandle.write(out_str)
        
    fhandle.close()
    np.save("%s_E.npy" % args.o,examples)
    np.save("%s_frames.npy" % args.o,example_frames)
    if args.do_wave_output:
        wavfile.write("%s_padded.wav" % args.o,
                      config_d['SPECTROGRAM']['sample_rate'],
                      wave_examples)


    

    


if __name__=="__main__":
    parser = argparse.ArgumentParser("""
    Extracts segments using features computed with a config file based on time locations fed into 
    the feature extractor
    """)
    parser.add_argument('-c',type=str,help="path to configuration file")
    parser.add_argument('-f',type=str,help="path to .wav file containing the audio recording")
    parser.add_argument('-s',type=str,nargs='+',help="speaker identifier")
    parser.add_argument('-t',type=int,nargs='+',help="start and end times for the audio recording and identity of the word from the dictionary")
    parser.add_argument('-o',type=str,help="file prefix to save the guide and the other work")
    parser.add_argument('-m',type=int,help="maximum section length")
    #parser.add_argument('-u',type=float,default=10,help="number of milliseconds for each time unit that the -s argument was given in, default is 10")
    parser.add_argument('--do_wave_output',action='store_true',
                        help="output the wave files to listen to what was extracted")
    main(parser.parse_args())
