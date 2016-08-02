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
from modules.random_forest import random_forest
from modules.check_accuracy import check_accuracy
from itertools import product
from sklearn import cross_validation
import numpy as np
import argparse


def parse_options():
    parser = argparse.ArgumentParser("symBIN Symbiotic Genome Binning with Machine Learning")

    parser.add_argument("-r", "--reference_genomes", type=str, required=True,
            nargs="+", help="genomes in fasta format to train data with")
    parser.add_argument("-c", "--chunk_size", type=int, required=False,
            default=1000, help="split genome into n-size chunks for learning.\
                    Default=1000")
    parser.add_argument("-k", "--kmer_size", type=int, required=False,
            default=4, help="kmer size for creating distribution table. Default=4")
    parser.add_argument("-t", "--test_size", type=float, required=False,
            default=0.4, help="fraction of data to use for model testing (as decimal)\
                    Default=0.4")
    parser.add_argument("-n", "--n_est", type=int, required=False,
            default=100, help="number of estimators for Random Forest model\
                    Default=100")

    args = parser.parse_args()
    return args

def generate_data(fl, genome_num, chunk_size, kmer_size):
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
        tmp_array = generate_data(genome, genome_num, args.chunk_size, args.kmer_size)
        genome_num += 1
        array_list.append(tmp_array)

    # concatentate all reference genome data into single array
    reference_array = np.concatenate(array_list, axis=0)

    # split reference array into train / test datasets
    ref_train, ref_test = cross_validation.train_test_split(reference_array, \
            test_size=args.test_size, random_state = 0)
    
    print "train_set\n", ref_train, "\ntest_set\n", ref_test 
    print ref_train.shape, ref_test.shape
    
    # train Random Forest classifier
    rf_output = random_forest(ref_train, ref_test, args.n_est)
    
    # check accuracy of training
    check_accuracy(ref_test, rf_output)


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
    
    
    
