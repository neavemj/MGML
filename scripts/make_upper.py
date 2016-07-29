#!/usr/bin/env python

# make nucleotides uppercase
# Matthew J. Neave 28.07.16

import sys

nuc_file = open(sys.argv[1])
output = open(sys.argv[2], "w")

for line in nuc_file:
    if line.startswith(">"):
        output.write(line)    
    else:
        line = line.upper()
        output.write(line)
