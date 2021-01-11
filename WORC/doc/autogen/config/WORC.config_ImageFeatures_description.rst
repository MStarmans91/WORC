==================== ========================================================================================================================================================
Subkey               Description                                                                                                                                             
==================== ========================================================================================================================================================
shape                Determine whether orientation features are computed or not.                                                                                             
histogram            Determine whether histogram features are computed or not.                                                                                               
orientation          Determine whether orientation features are computed or not.                                                                                             
texture_Gabor        Determine whether Gabor texture features are computed or not.                                                                                           
texture_LBP          Determine whether LBP texture features are computed or not.                                                                                             
texture_GLCM         Determine whether GLCM texture features are computed or not.                                                                                            
texture_GLCMMS       Determine whether GLCM Multislice texture features are computed or not.                                                                                 
texture_GLRLM        Determine whether GLRLM texture features are computed or not.                                                                                           
texture_GLSZM        Determine whether GLSZM texture features are computed or not.                                                                                           
texture_NGTDM        Determine whether NGTDM texture features are computed or not.                                                                                           
coliage              Determine whether coliage features are computed or not.                                                                                                 
vessel               Determine whether vessel features are computed or not.                                                                                                  
log                  Determine whether LoG features are computed or not.                                                                                                     
phase                Determine whether local phase features are computed or not.                                                                                             
image_type           Modality of images supplied. Determines how the image is loaded.                                                                                        
gabor_frequencies    Frequencies of Gabor filters used: can be a single float or a list.                                                                                     
gabor_angles         Angles of Gabor filters in degrees: can be a single integer or a list.                                                                                  
GLCM_angles          Angles used in GLCM computation in radians: can be a single float or a list.                                                                            
GLCM_levels          Number of grayscale levels used in discretization before GLCM computation.                                                                              
GLCM_distances       Distance(s) used in GLCM computation in pixels: can be a single integer or a list.                                                                      
LBP_radius           Radii used for LBP computation: can be a single integer or a list.                                                                                      
LBP_npoints          Number(s) of points used in LBP computation: can be a single integer or a list.                                                                         
phase_minwavelength  Minimal wavelength in pixels used for phase features.                                                                                                   
phase_nscale         Number of scales used in phase feature computation.                                                                                                     
log_sigma            Standard deviation(s) in pixels used in log feature computation: can be a single integer or a list.                                                     
vessel_scale_range   Scale in pixels used for Frangi vessel filter. Given as a minimum and a maximum.                                                                        
vessel_scale_step    Step size used to go from minimum to maximum scale on Frangi vessel filter.                                                                             
vessel_radius        Radius to determine boundary of between inner part and edge in Frangi vessel filter.                                                                    
dicom_feature_tags   DICOM tags to be extracted as features. See https://worc.readthedocs.io/en/latest/static/features.html.                                                 
dicom_feature_labels For each of the DICOM tag values extracted, name that should be assigned to the feature. See https://worc.readthedocs.io/en/latest/static/features.html.
==================== ========================================================================================================================================================