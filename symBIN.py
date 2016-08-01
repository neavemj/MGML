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
from itertools import product
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import argparse


def parse_options():
    parser = argparse.ArgumentParser("symBIN Symbiotic Genome Binning with Machine Learning")

    parser.add_argument("-r", "--reference_genomes", type=str, required=True,
            nargs="+", help="genomes in fasta format to train data with")
    parser.add_argument("-c", "--chunk_size", type=int, required=False,
            default=1000, help="split genome into n-size chunks for learning")

    args = parser.parse_args()
    return args

def generate_data(fl, genome_num, chunk_size=1000, kmer_size=4):
    """
    go through genome in fasta format and extract gc content, kmer freqs
    cuts genome into "chunk_size" for these calculations
    returns a numpy array with this information
    """

    # first calc all possible kmers for a given kmer_size 
    # important to do this here so that the list order stays the same
    possible_kmers = ["".join(kmer) for kmer in product(("G", "C", "T", "A"), repeat=kmer_size)] 
    fl_list = []
    fl_chunk_generator =  prepare_data.parse_clean_fasta(fl, chunk_size)

    for fl_chunk_tuples in fl_chunk_generator:
        for fl_chunk in fl_chunk_tuples:
            seq_name = fl_chunk[0]
            seq = fl_chunk[1]
            gc_content = calc_gc(seq) 
            kmer_list = calc_kmer(seq, possible_kmers, kmer_size)
            fl_list.append([seq_name, genome_num, gc_content] + kmer_list)
    return np.array(fl_list)


def symBIN():
    args = parse_options()
   
    # loop through each genome and calculate required data
    genome_num = 0
    array_list = []
    for genome in args.reference_genomes:
        tmp_array = generate_data(genome, genome_num, chunk_size=1000, kmer_size=4)
        genome_num += 1
        array_list.append(tmp_array)
    genomes_array = np.concatenate(array_list, axis=0)
    print genomes_array
    print genomes_array.shape

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
    
    
    
