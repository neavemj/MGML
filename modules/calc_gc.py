# calculate gc content of DNA sequence
# Matthew J. Neave 01.08.16

def calc_gc(sequence):
    """
    calculates gc content and returns percentage
    can be either upper or lower case
    >>> calc_gc("ACTGGNACCTAC")
    50.0
    """
    gc_num = [sequence.count(n) for n in ["G", "C", "g", "c"]]
    gc_sum = sum(gc_num)
    try:
        return gc_sum * 100.0 / len(sequence) 
    except ZeroDivisionError:
        return 0
