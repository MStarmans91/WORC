============================== ============================ =====================================================
Subkey                         Default                      Options                                              
============================== ============================ =====================================================
cross_validation               True                         True, False                                          
Segmentix                      False                        True, False                                          
FeatureCalculators             [predict/CalcFeatures:1.0]   [predict/CalcFeatures:1.0]                           
Preprocessing                  worc/PreProcess:1.0          worc/PreProcess:1.0, your own tool reference         
RegistrationNode               'elastix4.8/Elastix:4.8'     'elastix4.8/Elastix:4.8', your own tool reference    
TransformationNode             'elastix4.8/Transformix:4.8' 'elastix4.8/Transformix:4.8', your own tool reference
Joblib_ncores                  1                            Integer > 0                                          
Joblib_backend                 threading                    multiprocessing, threading                           
tempsave                       False                        True, False                                          
AssumeSameImageAndMaskMetadata False                        False                                                
============================== ============================ =====================================================