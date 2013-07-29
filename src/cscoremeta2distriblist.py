#!/usr/bin/python


import argparse

def main(args):
    """
    """
    out_str = ""
    for fl in open(args.s,'r'):
        if len(fl) < len('.npy') or fl[-len('.npy'):] !='.npy':
            continue
        sm = np.load(fl)
        for fl_entry_id, fl_entry in enumerate(sm):
            for model_entry_id, model_entry in enumerate(fl_entry):
                print "%f %d %d" % (
                    -model_entry[5],
                    1 if int(model_entry[3]) == int(model_entry[0]) else 0,
                    1 if int(model_entry[2]) == int(model_entry[1]))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("""
    Take a list of files containing metadata on the scoring
    and compute the list of outputs that can be fed
    to the compute_distrib function for the zero
    resource scoring mechanism and prints those outputs out
    """)
    parser.add_argument('-s',type=path,help="list of score_meta files to load")
    main(parser.parse_args())
