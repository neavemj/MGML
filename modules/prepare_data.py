#!/usr/bin/env python

# make nucleotides uppercase
# Matthew J. Neave 28.07.16

import sys

def make_upper(genome_file): 
    with open(genome_file) as f:
        for line in genome_file:
            if line.startswith(">"):
                output.write(line)    
            else:
                line = line.upper()
                output.write(line)
