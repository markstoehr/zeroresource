#!/usr/bin/python

import sys

new_lines = ()
for line in sys.stdin:
    cols = line.split()
    new_lines+= ((int(cols[2][:-1]), #file id without the channel identifier
                  cols[2][-1], # channel identifier for the utterance
                  int(cols[3]),
                  int(cols[4]),
                  int(cols[5]),
                  cols[0]),
              )

new_lines = sorted(new_lines,key=lambda l: l[3])
new_lines = sorted(new_lines,key=lambda l: l[1])
new_lines = sorted(new_lines,key=lambda l: l[0])
for line in new_lines:
    print "%d%s %d %d %d %s" % line

