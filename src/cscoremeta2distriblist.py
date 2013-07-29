#!/usr/bin/python


import argparse
import numpy as np

MAXNUM=np.exp(6.5)

def main(args):
    """
    """
    print args
    print "running over all files in list" 
    for fl_id,fl in enumerate(open(args.s,'r')):
        if args.v and fl_id % 100 == 0:
            print fl_id, fl
            
        fl = fl.strip()

        if len(fl) < len('.npy') or fl[-len('.npy'):] !='.npy':
            continue
        sm = np.load(fl)


        for fl_entry_id, fl_entry in enumerate(sm):
            for model_entry_id, model_entry in enumerate(fl_entry):
                print "%f %d %d" % (
                    np.exp(min(MAXNUM,-model_entry[5])),
                    1 if int(model_entry[3]) == int(model_entry[0]) else 0,
                    1 if int(model_entry[2]) == int(model_entry[1]) else 0)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""
    Take a list of files containing metadata on the scoring
    and compute the list of outputs that can be fed
    to the compute_distrib function for the zero
    resource scoring mechanism and prints those outputs out
    """)
    parser.add_argument('-s',type=str,help="list of score_meta files to load")
    parser.add_argument('-v',action="store_true",help="whether to print out progress")
    main(parser.parse_args())
