#!/usr/bin/python

import argparse, collections

def main(args):
    fls_dict = dict([(l.split()[0][3:],l.split()[-1]) for l in open(args.f,'r').read().strip().split('\n')])

    wd_list = [l.split() for l in open(args.d,'r').read().strip().split('\n')]
    just_words = [l[-1] for l in wd_list]
    just_words.sort()
    unique_words= []
    prev_word = ''
    for w in just_words:
        if w != prev_word:
            unique_words.append(w)
        prev_word = w[:]

    word_to_dict_idx = dict((v,str(k)) for k,v in enumerate(unique_words))

    open(args.out_dict,'w').write('\n'.join(["%s %s" % (str(k),v) for k,v in enumerate(unique_words)]))

    speaker_dict = {}
    for l in wd_list:
        if l[0] not in speaker_dict.keys():
            speaker_dict[l[0]] = l[1]
    

    segments_dict = collections.defaultdict(list)
    max_length = 0
    for l in wd_list:
        max_length = max(max_length,int(l[3]) - int(l[2]))
        segments_dict[l[0]].extend([l[2],l[3],word_to_dict_idx[l[4]]])

    
    for l in segments_dict.keys():
        print "CExtractSegments.py -f %s -c %s -s %s -t %s -o %s -m %d" % (
            fls_dict[l],
            args.c,
            speaker_dict[l],
            ' '.join(segments_dict[l]),
            '%s_%s_%s' % (args.o,
                          l,
                              speaker_dict[l]),
            max_length)

if __name__=='__main__':
    parser = argparse.ArgumentParser("""
    Take a list of sorted words with file information from switchboard
    and the file paths to where the data is kept, where the
    features should be stored and where the program to 
    compute the features is stored
    and produce a shell script that extracts all the features
    and words that we care about from the switchboard data.
""")
    parser.add_argument('-f',type=str,help="input wav file list")
    parser.add_argument('-d',type=str,help="evaluation word list")
    parser.add_argument('-c',type=str,help="config file path")
    parser.add_argument('-o',type=str,help="out path")
    parser.add_argument('--out_dict',type=str,help="dictionary for words")
    main(parser.parse_args())
    
