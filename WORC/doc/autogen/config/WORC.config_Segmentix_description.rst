==================== ==================================================================================================================================================
Subkey               Description                                                                                                                                       
==================== ==================================================================================================================================================
mask                 If a mask is supplied, should the mask be subtracted from the contour or multiplied.                                                              
segtype              If Ring, then a ring around the segmentation will be used as contour. If Dilate, the segmentation will be dilated per 2-D axial slice with a disc.
segradius            Define the radius of the ring or disc used if segtype is Ring or Dilate, respectively.                                                            
N_blobs              How many of the largest blobs are extracted from the segmentation. If None, no blob extraction is used.                                           
fillholes            Determines whether hole filling will be used.                                                                                                     
remove_small_objects Determines whether small objects will be removed.                                                                                                 
min_object_size      Minimum of objects in voxels to not be removed if small objects are removed                                                                       
==================== ==================================================================================================================================================