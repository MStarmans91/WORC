=================== ==================================================================================================================================================
Subkey              Description                                                                                                                                       
=================== ==================================================================================================================================================
fastr               Use fastr for the optimization gridsearch (recommended on clusters, default) or if set to False , joblib (recommended for PCs but not on Windows).
fastr_plugin        Name of execution plugin to be used. Default use the same as the self.fastr_plugin for the WORC object.                                           
classifiers         Select the estimator(s) to use. Most are implemented using `sklearn <https://scikit-learn.org/stable/>`_. For abbreviations, see above.           
max_iter            Maximum number of iterations to use in training an estimator. Only for specific estimators, see `sklearn <https://scikit-learn.org/stable/>`_.    
SVMKernel           When using a SVM, specify the kernel type.                                                                                                        
SVMC                Range of the SVM slack parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b).                  
SVMdegree           Range of the SVM polynomial degree when using a polynomial kernel. We sample on a uniform scale: the parameters specify the range (a, a + b).     
SVMcoef0            Range of SVM homogeneity parameter. We sample on a uniform scale: the parameters specify the range (a, a + b).                                    
SVMgamma            Range of the SVM gamma parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b)                   
RFn_estimators      Range of number of trees in a RF. We sample on a uniform scale: the parameters specify the range (a, a + b).                                      
RFmin_samples_split Range of minimum number of samples required to split a branch in a RF. We sample on a uniform scale: the parameters specify the range (a, a + b). 
RFmax_depth         Range of maximum depth of a RF. We sample on a uniform scale: the parameters specify the range (a, a + b).                                        
LRpenalty           Penalty term used in LR.                                                                                                                          
LRC                 Range of regularization strength in LR. We sample on a uniform scale: the parameters specify the range (a, a + b).                                
LDA_solver          Solver used in LDA.                                                                                                                               
LDA_shrinkage       Range of the LDA shrinkage parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b).              
QDA_reg_param       Range of the QDA regularization parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b).         
ElasticNet_alpha    Range of the ElasticNet penalty parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b).         
ElasticNet_l1_ratio Range of l1 ratio in LR. We sample on a uniform scale: the parameters specify the range (a, a + b).                                               
SGD_alpha           Range of the SGD penalty parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b).                
SGD_l1_ratio        Range of l1 ratio in SGD. We sample on a uniform scale: the parameters specify the range (a, a + b).                                              
SGD_loss            hinge, Loss function of SG                                                                                                                        
SGD_penalty         Penalty term in SGD.                                                                                                                              
CNB_alpha           Regularization strenght in ComplementNB. We sample on a uniform scale: the parameters specify the range (a, a + b)                                
=================== ==================================================================================================================================================