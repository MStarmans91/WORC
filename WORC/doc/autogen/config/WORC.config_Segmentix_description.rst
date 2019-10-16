========= =======================================================================================================
Subkey    Description                                                                                            
========= =======================================================================================================
mask      If a mask is supplied, should the mask be subtracted from the contour or multiplied.                   
segtype   If Ring, then a ring around the segmentation will be used as contour.                                  
segradius Define the radius of the ring used if segtype is Ring.                                                 
N_blobs   How many of the largest blobs are extracted from the segmentation. If None, no blob extraction is used.
fillholes Determines whether hole filling will be used.                                                          
========= =======================================================================================================