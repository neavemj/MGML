#!/usr/bin/env python

# need to shuffle data for machine learning
# Matthew J. Neave 28.07.16

import sys
import numpy as np

data_file = [line.strip() for line in open(sys.argv[1])]
output = open(sys.argv[2], "w")

init_range = np.arange(len(data_file))
np.random.shuffle(init_range)
print init_range

for new_row in init_range:
    output.write(data_file[new_row] + "\n")
