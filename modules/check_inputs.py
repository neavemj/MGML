# check inputs from argparse
# Matthew J. Neave 02.08.16

def check_inputs(args, parser):
    
    import sys

    # have to specify an algorithm for the learning
    algorithm_list = ["rf"]
    if args.algorithm[0] not in algorithm_list:
        parser.print_help() 
        print "error: algorithm must be one of:", ", ".join(algorithm_list)
        sys.exit()

    # amount of data used for testing should be a decimal
    if args.test_size < 0 or args.test_size > 1:
        parser.print_help()
        print "error: test_size is given as decimal. Needs to be between 0 and 1."
        sys.exit()
