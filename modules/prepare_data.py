# format fasta file for use with kmer, gc content modules
# Matthew J. Neave 28.07.16

import os

def parse_clean_fasta(genome_file_handle, chunk_length):
    
    genome_name = os.path.basename(genome_file_handle).split(".")[0] 
    
    with open(genome_file_handle) as f:
        # ignore comments, blank lines, etc.
        while True:
            line = f.readline()
            if line == "":
                return  # file is empty or empty line
            if line.startswith(">"): # must be header line
                break
        
        # get info from sequence
        while True:
            header = line.strip().split(" ")[0].replace("|", "_") # must be header because of startswith(">") break above
            line = f.readline()
            whole_seq = ""
            while line: # will stop running if no line present
                if line.startswith(">"): # again must be header line
                    break
                whole_seq += line.strip()
                line = f.readline()
            
            if not line:    # finished reading fasta file
                return
            
            if len(whole_seq) > chunk_length:   # break large sequences into chunks
                window = 0
                count = 0
                chunk_list = []
                while True:
                    seq_chunk = whole_seq[window: window + chunk_length]
                    new_name = header + "_" + genome_name +  "_chunk_" + str(count)
                    chunk_list.append((new_name, seq_chunk))
                    window += chunk_length
                    count += 1
                    if (window + chunk_length) > len(whole_seq):
                        yield chunk_list
                        break
