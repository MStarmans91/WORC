============ ==============================================================================================================================================
Subkey       Description                                                                                                                                   
============ ==============================================================================================================================================
use          If True, use SMAC as the optimization strategy.                                                                                               
n_smac_cores Number of independent, parallel SMAC instances to use.                                                                                        
budget_type  Type of budget to use for the SMAC optimization, either an evaluation limit or a time limit.                                                  
budget       Size of the budget, which depends on the type of budget. Number of evaluations for an evaluation limit, or wallclock seconds for a time limit.
init_method  Initialization method of SMAC. Supported are a random initialization or a sobol sequence.                                                     
init_budget  Number of evaluations used for the initialization. Always an evaluation limit, regardless of the budget type choice of the optimization.      
============ ==============================================================================================================================================