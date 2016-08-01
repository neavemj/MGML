# format fasta file for use with kmer, gc content modules
# Matthew J. Neave 28.07.16


def parse_clean_fasta(genome_file_handle, chunk_length):
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
            header = line.strip()
            seq = []
            line = f.readline()
            while True:
                if line.startswith(">"):
                    break
                seq.append(line.strip())
                line = f.readline()
            
            whole_seq = "".join(seq)
            
            if len(whole_seq) > chunk_length:
                window = 0
                count = 0
                chunk_list = []
                while True:
                    seq_chunk = whole_seq[window: window + chunk_length]
                    new_name = header + "_chunk_" + str(count)
                    chunk_list.append((new_name, seq_chunk))
                    window += chunk_length
                    count += 1
                    if (window + chunk_length) > len(whole_seq):
                        yield chunk_list
            
            else:           # seq too small to be included in calculations
                return
            
            if not line:    # finished reading fasta file
                return
