==================== ==================== ====================================================
Subkey               Default              Options                                             
==================== ==================== ====================================================
shape                True                 True, False                                         
histogram            True                 True, False                                         
orientation          True                 True, False                                         
texture_Gabor        True                 True, False                                         
texture_LBP          True                 True, False                                         
texture_GLCM         True                 True, False                                         
texture_GLCMMS       True                 True, False                                         
texture_GLRLM        False                True, False                                         
texture_GLSZM        False                True, False                                         
texture_NGTDM        False                True, False                                         
coliage              False                True, False                                         
vessel               True                 True, False                                         
log                  True                 True, False                                         
phase                True                 True, False                                         
image_type           CT                   CT                                                  
gabor_frequencies    0.05, 0.2, 0.5       Float(s)                                            
gabor_angles         0, 45, 90, 135       Integer(s)                                          
GLCM_angles          0, 0.79, 1.57, 2.36  Float(s)                                            
GLCM_levels          16                   Integer > 0                                         
GLCM_distances       1, 3                 Integer(s) > 0                                      
LBP_radius           3, 8, 15             Integer(s) > 0                                      
LBP_npoints          12, 24, 36           Integer(s) > 0                                      
phase_minwavelength  3                    Integer > 0                                         
phase_nscale         5                    Integer > 0                                         
log_sigma            1, 5, 10             Integer(s)                                          
vessel_scale_range   1, 10                Two integers: min and max.                          
vessel_scale_step    2                    Integer > 0                                         
vessel_radius        5                    Integer > 0                                         
dicom_feature_tags   0010 1010, 0010 0040 DICOM tag keys, e.g. 0010 0010, separated by comma's
dicom_feature_labels age, sex             List of strings                                     
==================== ==================== ====================================================