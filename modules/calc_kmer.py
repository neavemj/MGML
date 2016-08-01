# calculate frequencies for givem kmer length
# Matthew J. Neave

def calc_kmer(sequence, kmer_order, kmer_size):
    """
    calculate kmer distribution for a given kmer size
    returns list of kmer abundances following the order given in kmer_order
    this ensures that order is maintained for different sequences
    """
    from collections import defaultdict
    window = 0
    kmer_dict = defaultdict(int)    # use defaultdict for easy kmer counting
    
    # slide window through sequence counting up the kmers
    while True:
        current_kmer = sequence[window: window + kmer_size]
        kmer_dict[current_kmer] += 1
        window += 1
        if (window + kmer_size) > len(sequence):
            break
    
    # now output list of kmer frequences in same order as "kmer_order"
    kmer_list = []
    for kmer in kmer_order:
        if kmer in kmer_dict:
            kmer_list.append(kmer_dict[kmer])
        else:
            kmer_list.append(0)
    return kmer_list
