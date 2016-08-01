#!/usr/bin/env python
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2016 Matthew J. Neave
#
# symBIN wrapper script to coordinate data collection and binning


from modules import prepare_data 
from modules.calc_gc import calc_gc
from modules.calc_kmer import calc_kmer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import argparse

# import data as numpy arrays

def parse_options():
    parser = argparse.ArgumentParser("symBIN Symbiotic Genome Binning with Machine Learning")

    parser.add_argument("-f", "--genome_files", type=str, required=True,
            nargs="+", help="genomes in fasta format to train data with")

    args = parser.parse_args()
    return args

def generate_data(fl, genome_num, chunk_size):
    """
    go through genome in fasta format and extract gc content, kmer freqs
    cuts genome into "chunk_size" for these calculations
    returns a numpy array with this information
    """
    fl_list = []
    fl_chunk_generator =  prepare_data.parse_clean_fasta(fl, chunk_size)
    for fl_chunk_tuples in fl_chunk_generator:
        for fl_chunk in fl_chunk_tuples:
            seq_name = fl_chunk[0]
            seq = fl_chunk[1]
            gc_content = calc_gc(seq) 
            fl_list.append([seq_name, genome_num, gc_content])
    return np.array(fl_list)


def symBIN():
    args = parse_options()
   
    # loop through each genome and calculate required data
    genome_num = 0
    for genome in args.genome_files:
        print  generate_data(genome, genome_num, 1000)
        genome_num += 1

if __name__ == "__main__":
    symBIN()

#train_array = create_numpy_array(train_file)[0]
#test_array, test_answers = create_numpy_array(test_file)
#
## create random forest object
#
#forest = RandomForestClassifier(n_estimators = 100)
#
## fit training data
#
##print test_array[0::,3:4]
#
#forest = forest.fit(train_array[0::,100::],train_array[0::,1])
#
#output = forest.predict(test_array[0::,100::])
##output = forest.predict_proba(test_array[0::,100::])
#
## check result against known answers
#
#correct = 0
#incorrect = 0
#for result, answer in zip(output, test_answers):
##    if result == "1" or answer == "1": 
#        if answer == result:
#            correct += 1
#        else:
#            incorrect += 1
#
#print "test accuracy:", correct, incorrect, (float(correct) / float(correct + incorrect))
#    
#    
    
    
    
