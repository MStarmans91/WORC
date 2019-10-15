================== ====================================================================================================================================================================
Subkey             Description                                                                                                                                                         
================== ====================================================================================================================================================================
cross_validation   Determine whether a cross validation will be performed or not. Obsolete, will be removed.                                                                           
Segmentix          Determine whether to use Segmentix tool for segmentation preprocessing.                                                                                             
FeatureCalculator  Specifies which feature calculation tool should be used.                                                                                                            
Preprocessing      Specifies which tool will be used for image preprocessing.                                                                                                          
RegistrationNode   Specifies which tool will be used for image registration.                                                                                                           
TransformationNode Specifies which tool will be used for applying image transformations.                                                                                               
Joblib_ncores      Number of cores to be used by joblib for multicore processing.                                                                                                      
Joblib_backend     Type of backend to be used by joblib for multicore processing.                                                                                                      
tempsave           Determines whether after every cross validation iteration the result will be saved, in addition to the result after all iterations. Especially useful for debugging.
================== ====================================================================================================================================================================