# use itertools
# based a whole lot on testbench_but_python.py



# generation function
# ex: we have 3 lanes and each has the following number of commands 
# we will remove a lane from the pool once that lane has run out of compressed hex
# 1: 3 rows, 2: 1 row, 3: 2 rows
# [1, 2, 3, 1, 3, 1] RR order
# [3, 2, 1, 1, 1, 3] random order
# ...... and it goes on and on and on

# main
# load in the sample image
# based on how many compressed hex beats there are per row, 
# generate all schdule permutations, if not supplied
# else, load a generated txt schedule
# should have an arg that says which line num to pick up from as well
# initiate a var to keep track of the best cycle count so far, and the rax hexes that were sent to DUT

# for each schedule
    # for each lane picked by schedule
        # if that lane is not ready
            # let clk tick
            # note:
        # feed that lane
    # record the raw hex, clk cycle and compare



