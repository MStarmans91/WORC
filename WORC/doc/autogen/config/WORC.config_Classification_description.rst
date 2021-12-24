====================== =======================================================================================================================================================================
Subkey                 Description                                                                                                                                                            
====================== =======================================================================================================================================================================
fastr                  Use fastr for the optimization gridsearch (recommended on clusters, default) or if set to False , joblib (recommended for PCs but not on Windows).                     
fastr_plugin           Name of execution plugin to be used. Default use the same as the self.fastr_plugin for the WORC object.                                                                
classifiers            Select the estimator(s) to use. Most are implemented using `sklearn <https://scikit-learn.org/stable/>`_. For abbreviations, see the options: LR = logistic regression.
max_iter               Maximum number of iterations to use in training an estimator. Only for specific estimators, see `sklearn <https://scikit-learn.org/stable/>`_.                         
SVMKernel              When using a SVM, specify the kernel type.                                                                                                                             
SVMC                   Range of the SVM slack parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale).                               
SVMdegree              Range of the SVM polynomial degree when using a polynomial kernel. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).                  
SVMcoef0               Range of SVM homogeneity parameter. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).                                                 
SVMgamma               Range of the SVM gamma parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale)                                
RFn_estimators         Range of number of trees in a RF. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).                                                   
RFmin_samples_split    Range of minimum number of samples required to split a branch in a RF. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).              
RFmax_depth            Range of maximum depth of a RF. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).                                                     
LRpenalty              Penalty term used in LR.                                                                                                                                               
LRC                    Range of regularization strength in LR. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).                                             
LR_solver              Solver used in LR.                                                                                                                                                     
LR_l1_ratio            Ratio between l1 and l2 penalty when using elasticnet penalty, see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.     
LDA_solver             Solver used in LDA.                                                                                                                                                    
LDA_shrinkage          Range of the LDA shrinkage parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale).                           
QDA_reg_param          Range of the QDA regularization parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale).                      
ElasticNet_alpha       Range of the ElasticNet penalty parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale).                      
ElasticNet_l1_ratio    Range of l1 ratio in LR. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).                                                            
SGD_alpha              Range of the SGD penalty parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale).                             
SGD_l1_ratio           Range of l1 ratio in SGD. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).                                                           
SGD_loss               Loss function of SGD.                                                                                                                                                  
SGD_penalty            Penalty term in SGD.                                                                                                                                                   
CNB_alpha              Regularization strenght in ComplementNB. We sample on a uniform scale: the parameters specify the range (loc, loc + scale)                                             
AdaBoost_n_estimators  Number of estimators used in AdaBoost. Default is equal to config['Classification']['RFn_estimators'].                                                                 
AdaBoost_learning_rate Learning rate in AdaBoost.                                                                                                                                             
XGB_boosting_rounds    Number of estimators / boosting rounds used in XGB. Default is equal to config['Classification']['RFn_estimators'].                                                    
XGB_max_depth          Maximum depth of XGB.                                                                                                                                                  
XGB_learning_rate      Learning rate in AdaBoost. Default is equal to config['Classification']['AdaBoost_learning_rate'].                                                                     
XGB_gamma              Gamma of XGB.                                                                                                                                                          
XGB_min_child_weight   Minimum child weights in XGB.                                                                                                                                          
XGB_colsample_bytree   Col sample by tree in XGB.                                                                                                                                             
====================== =======================================================================================================================================================================