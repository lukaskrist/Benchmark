# Benchmark

This is a file containing quite a bit of the code which was used for my thesis, using an Augmented Random Search to search for optimal solutions of quantum control. 

In Test0 file the basic algorithm is shown, using a random starting value of rows, and then seeing how long it takes to converge, spitting out the minimum amount of time it takes to converge in terms of quantum time.

In Test1 file the spin chain is tried to see if a roll_out is actually working. 

In Test2 the spin chain minimum QSL time is found. 

In Test_Data we try the version which includes a data driven Augmented Random Search (Own algorithm). 

And finally T-SNE-DATA-test shows how the Augmented random search is spread in results unlike other algorithms.
