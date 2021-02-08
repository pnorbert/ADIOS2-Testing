
Number of writers and readers must be set up in the settings file and match the number of processes on writer/reader side. This is necessary to pre-calculate the communication pattern between readers and writers in every process on its own. 

Assumptions in the data blocks:

  1D: the blocks are enumerated monotonically
  1D: the producer ranks are monotonically increasing
  3D: each block has the same size but the sizes can be different in x-y-z dimensions
  3D: the producer ranks are monotonically increasing


Small test case for debugging:
```
test.json
input/decomp_test_8blocks_1D.in
input/decomp_test_8blocks_3D.in
```
* 4 producers generated 8 blocks (both in 3D and 1D), therefore
* test can run with max writers = 4, max readers = 4
* set up for 4 writers, 2 readers
* in 1D, only rank 0 and 2 generated the 8 1D blocks, so only two writers will write them

