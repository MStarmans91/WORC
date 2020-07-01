..  features-chapter:

Features
===========

WORC is not a feature extraction toolbox, but a workflow management and foremost workflow optimization method / toolbox.
However, feature extraction is generally part of the workflow. Users can add their own feature toolbox, but the default
used feature toolboxes are `PREDICT <https://github.com/Svdvoort/PREDICTFastr/>`_ and
`PyRadiomics <https://github.com/Svdvoort/https://github.com/Radiomics/pyradiomics/>`_ . The options for feature extraction
using these toolboxes within WORC and their defaults are described in this chapter, organized per
feature group. For details on the settings for the feature extraction, please see the :ref:`Config chapter <config-chapter>`.

For all features, the feature labels reflect the descriptions named here. When parameters have to be set,
the values of these parameters are included in the feature label.

Furthermore, we refer the user to the following literature:

- More information on PyRadiomics: `Van Griethuysen, Joost JM, et al. "Computational radiomics system to decode the radiographic phenotype." Cancer research 77.21 (2017): e104-e107. <https://cancerres.aacrjournals.org/content/77/21/e104?utm_source=170339&utm_medium=convergence&utm_campaign=sections>`_
- More detailed description of many of the used features:  `Parekh, Vishwa, and Michael A. Jacobs. "Radiomics: a new application from established techniques." Expert review of precision medicine and drug development 1.2 (2016): 207-226. <https://www.tandfonline.com/doi/abs/10.1080/23808993.2016.1164013>`_
- Overview of often used radiomics features: `Zwanenburg, Alex, et al. "The image biomarker standardization initiative: standardized quantitative radiomics for high-throughput image-based phenotyping." Radiology 295.2 (2020): 328-338. <https://pubs.rsna.org/doi/full/10.1148/radiol.2020191145>`_

Histogram features
-------------------
Histogram features are based on the image intensities themselves. Usually, a histogram of the intensities is made, after
which several first order statistics are extracted. Therefore, these features are commonly also referred to as
first order or intensity features.

Both PREDICT and PyRadiomics include similar first order features. We have therefore chosen to only use PREDICT
by default to avoid redundant features. PREDICT extracts the following features using a histogram with 50 bins:

1. Minimum
2. Maximum
3. Range
4. Interquartile range
5. Standard deviation
6. Skewness
7. Kurtosis
8. Peak
9. Energy
10. Entropy
11. Mean
12. Median


.. note:: The minimum, maximum, range and interquartile range are extracted from the raw data, as histogram creation may
          may result in a loss of needed information.

Shape features
--------------
Shape features describe morphological properties of the region of interest and are therefore solely based on the
segmentation, not the image. As PREDICT and PyRadiomics offer complementary shape descriptors, both packages are used
by default.

In PREDICT, these descriptors are by default extracted per 2-D slice and aggregated over all slices,
as in our experience the slice thickness is often too large to create sensible 3-D shape descriptors. For each
aggregated descriptor, PREDICT extracts the mean and standard deviation.
Most of the shape features are based on the following papers:


    `Xu, Jiajing, et al. "A comprehensive descriptor of shape: method and application to content-based retrieval of similar appearing lesions in medical images." Journal of digital imaging 25.1 (2012): 121-128. <https://link.springer.com/content/pdf/10.1007/s10278-011-9388-8.pdf>`_

    `Peura, Markus, and Jukka Iivarinen. "Efficiency of simple shape descriptors." Aspects of visual form (1997): 443-451. <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.30.9018&rep=rep1&type=pdf>`_

The mean and standard deviation of following shape features are extracted:

1. Compactness
2. Radial distance
3. Roughness
4. Convexity
5. Circular variance
6. PRAX
7. Elliptical variance
8. Solidity
9. Area

Additional, if pixel spacing is included in the image or metadata, the volume is computed for a total of 19 shape
features.

In PyRadiomics, the following shape features according to the defaults are extracted:

1. Elongation
2. Flatness
3. Least Axis Length
4. Major Axis Length
5. Maximum 2D diameter for columns
6. Maximum 2D diameter for rows
7. Maximum 2D diameter for slices
8. Maximum 3D diameter
9. Mesh Volume
10. Minor Axis Length
11. Sphericity
12. Surface Area
13. Surface Volume Ratio
14. Voxel Volume

Hence, the total number of shape features is 33.

Texture features
-----------------
The last group is the largest and basically contains all features not within the other groups, as a feature
quantifying a form of texture is a broad definition. Within the texture features, there are several sub-groups.
If groupwise feature selection is used, each of these subgroups has an on/off hyperparameter.

Note that we have decided to split several groups from the texture features. Within the texture features,
we have included more commonly used texture features, as these are indeed commonly grouped under texture features.
The less well-known features are described later on in this chapter.

Gray-Level Co-occurence Matrix (GLCM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The GLCM and other gray-level based matrix features are based on a discretized version of the image, i.e.
the gray-level matrix. The ``config['ImageFeatures']['GLCM_levels']`` parameter determines the number of
levels for the discretization. As default, WORC uses 16 levels, as this works in smaller regions of
interests (ROI) containing fewer regions but does not throw away to much information in larger regions.

The GLCM counts the co-occurences of neighbouring pixels of each gray level value using two parameters:
the distance between pixels, and the angle in which co-occurences are counted. As generally beforehand it
is not known which of these settings may lead to relevant features, the GLCM at multiple values is extracted:

    config['ImageFeatures']['GLCM_angles'] = '0, 0.79, 1.57, 2.36'
    config['ImageFeatures']['GLCM_distances'] = '1, 3'

Boht PREDICT and PyRadiomics can extract GCLM features. Again, we would like to extract the GLCM per 2D slice, similar
to the shape fetures, As a default, we use therefore PREDICT, as PREDICT provides two ways to do so: compute
the GLCM and it's features per slice and aggregate, or aggregate the GLCM's of all slices and once compute features,
which PREDICT calls GLCM Multi Slice (GLCMMS) features.

PREDICT extracts both for the GLCM and GLCMMS for all combinations of angles and distances the following features:

1. Contrast
2. Dissimilarity
3. Homogeneity
4. Angular Second Momentum (ASM)
5. Energy
6. Correlation

The settings for the parameters are included in the feature label. For example, tf_GLCM_contrastd1.0A1.0 is
the contrast 






