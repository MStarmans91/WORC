============ ===================================================================================================================================
Subkey       Description                                                                                                                        
============ ===================================================================================================================================
Type         If performing a cross-validationm, type of cross-validation used. Currently random-splitting and leave-one-out (LOO) are supported.
N_iterations Number of times the data is split in training and test in the outer cross-validation when using random-splitting.                  
test_size    The percentage of data to be used for testing when using random-splitting.                                                         
fixed_seed   If True, use a fixed seed for the cross-validation splits when using random-splitting.                                             
============ ===================================================================================================================================