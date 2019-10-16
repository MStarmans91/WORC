============== ======================================================================================================================================
Subkey         Description                                                                                                                           
============== ======================================================================================================================================
scoring_method Specify the optimization metric for your hyperparameter search.                                                                       
test_size      Size of test set in the hyperoptimization cross validation, given as a percentage of the whole dataset.                               
n_splits       Number of iterations in train-validation cross-validation used for model optimization.                                                
N_iterations   Number of iterations used in the hyperparameter optimization. This corresponds to the number of samples drawn from the parameter grid.
n_jobspercore  Number of jobs assigned to a single core. Only used if fastr is set to true in the classfication.                                     
maxlen         Number of estimators for which the fitted outcomes and parameters are saved. Increasing this number will increase the memory usage.   
ranking_score  Score used for ranking the performance of the evaluated workflows.                                                                    
============== ======================================================================================================================================