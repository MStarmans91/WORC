..  features-chapter:

Radiomics Features
==================

WORC is not a feature extraction toolbox, but a workflow management and foremost workflow optimization method / toolbox.
However, feature extraction is generally part of the workflow. Users can add their own feature toolbox, but the default
used feature toolboxes are `PREDICT <https://github.com/Svdvoort/PREDICTFastr/>`_ and
`PyRadiomics <https://github.com/Svdvoort/https://github.com/Radiomics/pyradiomics/>`_ . The options for feature extraction
using these toolboxes within WORC and their defaults are described in this chapter, organized per
feature group.

Here, we provide an overview of all features and an explantion of what they
quantify. For a comprehensive overview of all functions and parameters, please look at
:ref:`the config chapter <config-chapter>`.

For all features, the feature labels reflect the descriptions named here. When parameters have to be set,
the values of these parameters are included in the feature label.

For all the features, you can determine whether PREDICT or PyRadiomics exctract these by changing the
related parameters in ``config['PyRadiomics']`` and ``config['ImageFeatures']`` for PREDICT.

Furthermore, we refer the user to the following literature:

- More information on PyRadiomics: `Van Griethuysen, Joost JM, et al. "Computational radiomics system to decode the radiographic phenotype." Cancer research 77.21 (2017): e104-e107. <https://cancerres.aacrjournals.org/content/77/21/e104?utm_source=170339&utm_medium=convergence&utm_campaign=sections>`_
- More detailed description of many of the used features:  `Parekh, Vishwa, and Michael A. Jacobs. "Radiomics: a new application from established techniques." Expert review of precision medicine and drug development 1.2 (2016): 207-226. <https://www.tandfonline.com/doi/abs/10.1080/23808993.2016.1164013>`_
- Overview of often used radiomics features: `Zwanenburg, Alex, et al. "The image biomarker standardization initiative: standardized quantitative radiomics for high-throughput image-based phenotyping." Radiology 295.2 (2020): 328-338. <https://pubs.rsna.org/doi/full/10.1148/radiol.2020191145>`_

In total, the defaults of WORC result in the following amount of features:

============================================ ===================================================
Type                                          Number
============================================ ===================================================
:ref:`Histogram <features-histogram>`         13
:ref:`Shape <features-shape>`                 35
:ref:`Orientation <features-orientation>`     9
:ref:`GLCM(MS) <features-GLCM>`               144
:ref:`GLRLM <features-GLRLM>`                 16
:ref:`GLSZM <features-GLSZM>`                 16
:ref:`NGTDM <features-NGTDM>`                 5
:ref:`GLDM <features-GLDM>`                   14
:ref:`Gabor filter <features-Gabor>`          156
:ref:`LoG filter <features-LoG>`              39
:ref:`Vessel filter <features-vessel>`        39
:ref:`Local Binary Patterns <features-lbp>`   39
:ref:`Local phase <features-phase>`           39
-------------------------------------------- ---------------------------------------------------
**Total**                                     **564**
============================================ ===================================================


.. note:: The settings for the parameters are included in the feature label. For example, tf_GLCM_contrastd1.0A1.57 is
          the contrast of the GLCM computed at a distance of 1 pixel and and angle of 1.57 radians ~ 90 degrees.

.. _features-histogram:

Histogram features
-------------------
Histogram features are based on the image intensities themselves. Usually, a histogram of the intensities is made, after
which several first order statistics are extracted. Therefore, these features are commonly also referred to as
first order or intensity features.

Both PREDICT and PyRadiomics include similar first order features. We have therefore chosen to only use PREDICT
by default to avoid redundant features. PREDICT extracts the following features using a histogram with 50 bins:

1. Minimum (defined as the 2nd percentile for robustness)
2. Maximum (defined as the 98nd percentile for robustness)
3. Range
4. Interquartile range
5. Standard deviation
6. Skewness
7. Kurtosis
8. Peak value
9. Peak position
10. Energy
11. Entropy
12. Mean
13. Median


.. note:: The minimum, maximum, range and interquartile range are extracted from the raw data, as histogram creation may
          may result in a loss of needed information.

.. _features-shape:

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
6. Principal axis ratio (PRAX)
7. Elliptical variance
8. Solidity
9. Area

Additional, the min and max area and, if pixel spacing is included in the image or metadata, the volume is computed for a total of 21 shape
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

Hence, the total number of shape features is 35.

.. _features-orientation:

Orientation features
--------------------
Orientation features describe the orientation and location of the ROI. While these on itself
may not be relevant for the prediction, these may serve as moderation features for orientation dependent features.
As PREDICT and PyRadiomics again provide complementary features, by default WORC uses both toolboxes for
orientation feature extraction

The following orientation features are extracted from PREDICT:

1. X-angle
2. Y-angle
3. Z-angle

The angles are extracted by fitting a 3D ellips to the ROI and using the orientations fo the three major axes.

The following orientation features are extracted from PyRadiomics using the Center Of Mass (COM):

1. COM index x
2. COM index y
3. COM index z
4. COM x
5. COM y
6. COM z

.. _features-texture:

Texture features
-----------------
The last group is the largest and basically contains all features not within the other groups, as a feature
quantifying a form of texture is a broad definition. Within the texture features, there are several sub-groups.
If groupwise feature selection is used, each of these subgroups has an on/off hyperparameter.

Note that we have decided to split several groups from the texture features. Within the texture features,
we have included more commonly used texture features, as these are indeed commonly grouped under texture features.
The less well-known features are described later on in this chapter.

.. _features-GLCM:

Gray-Level Co-occurence Matrix (GLCM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The GLCM and other gray-level based matrix features are based on a discretized version of the image, i.e.
the gray-level matrix. The ``config['ImageFeatures']['GLCM_levels']`` parameter determines the number of
levels for the discretization. As default, WORC uses 16 levels, as this works in smaller ROIs containing
fewer regions but does not throw away to much information in larger regions.

The GLCM counts the co-occurences of neighbouring pixels of each gray level value using two parameters:
the distance between pixels, and the angle in which co-occurences are counted. As generally beforehand it
is not known which of these settings may lead to relevant features, the GLCM at multiple values is extracted:

.. code-block:: python

    config['ImageFeatures']['GLCM_angles'] = '0, 0.79, 1.57, 2.36'
    config['ImageFeatures']['GLCM_distances'] = '1, 3'

Boht PREDICT and PyRadiomics can extract GCLM features. Again, we would like to extract the GLCM per 2D slice, similar
to the shape fetures, As a default, we use therefore PREDICT, as PREDICT provides two ways to do so: compute
the GLCM and it's features per slice and aggregate, or aggregate the GLCM's of all slices and once compute features,
which PREDICT calls GLCM Multi Slice (GLCMMS) features.
re
PREDICT extracts both for the GLCM and GLCMMS for all combinations of angles and distances the following features:

1. Contrast
2. Dissimilarity
3. Homogeneity
4. Angular Second Momentum (ASM)
5. Energy
6. Correlation

In total, computing these six features for both the GCLM and GLCMMS for all combinations of angles and degrees
results in a total of 144 features.

.. _features-GLRLM:

Gray-Level Run Length Matrix (GLRLM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The GRLM counts how many lines of a certain gray level and length occur, in a specific direction. The only
parameter of the GRLM is thus the direction, for which we use the PyRadiomics default. The GRLM is in PREDICT
extracted using PyRadiomics, so WORC relies on directly using PyRadiomics.

The following GRLM features are by default extracted:

1. Gray level non-uniformity
2. Gray level non-uniformity normalized
3. Gray level variance
4. High gray level run emphasis
5. Long run emphasis
6. Long run high gray level emphasis
7. Long run low gray level emphasis
8. Low gray level run emphasis
9. Run entropy
10. Run length non-uniformity
11. Run length non-uniformity normalized
12. Run percentage
13. Run variance
14. Short run emphasis
15. Short run high gray level emphasis
16. Short run low gray level emphasis

.. _features-GLSZM:

Gray-Level Size Zone Matrix (GLSZM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The GLSZM counts how many areas of a certain gray level and size occur. It therefore has no parameters.
The GLSZM is in PREDICT extracted using PyRadiomics, so WORC relies on directly using PyRadiomics.

The following GLSZM features are by default extracted:

1. Gray level non-uniformity
2. Gray level non-uniformity normalized
3. Gray level variance
4. High gray level zone emphasis
5. Large area emphasis
6. Large area high gray level emphasis
7. Large area low gray level emphasis
8. Low gray level zone emphasis
9. zone entropy
10. Size zone non-uniformity
11. Size zone non-uniformity normalized
12. Zone percentage
13. Zone variance
14. Small area emphasis
15. Small area high gray level emphasis
16. Small area low gray level emphasis

.. _features-GLDM:

Gray Level Dependence Matrix (GLDM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The GLDM determines how much voxels in a neighborhood depend (e.g. are similar) to the centre
voxel. Parameters include the distance to define the neighborhood and the similarity threshold.
The GLDM is also extracted using PyRadiomics, and it's default therefore used.

The following GLDM features are used:

1. Dependence Entropy
2. Dependence Non-Uniformity
3. Dependence Non-Uniformity Normalized
4. Dependence Variance
5. Gray Level Non-Uniformity
6. Gray Level Variance
7. High Gray Level Emphasis
8. Large Dependence Emphasis
9. Large Dependence High Gray Level Emphasis
10. Large Dependence Low Gray Level Emphasis
11. Low Gray Level Emphasis
12. Small Dependence Emphasis
13. Small Dependence High Gray Level Emphasis
14. Small Dependence Low Gray Level Emphasis

.. _features-NGTDM:

Neighborhood Gray Tone Difference Matrix (NGTDM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The NGTDM looks at the difference between a pixel's gray value and that of it's neighborhood within a distance,
which is the only parameter. The NGTDM is also extracted using PyRadiomics, and it's default therefore used.

The following NGTDM features are extracted:

1. Busyness
2. Coarseness
3. Complexity
4. Contrast
5. Strength

.. _features-Gabor:

Gabor filter features
^^^^^^^^^^^^^^^^^^^^^^
These features are extracted through PREDICT by first applying a set of Gabor filters to the image with the following
parameters:

.. code-block:: python

        config['ImageFeatures']['gabor_frequencies'] = '0.05, 0.2, 0.5'
        config['ImageFeatures']['gabor_angles'] = '0, 45, 90, 135'

The angles are equal to the GLCM angles, but are given in degrees. For each unique combination of angle and frequency,
the image is filtered per 2-D axial slice, after which the PREDICT histogram features
as :ref:`discussed earlier <features-histogram>` are extracted from the filtered images.

.. _features-LoG:

Laplacian of Gaussian (LoG) filter features
-------------------------------------------
Similar to the Gabor features, these features are extracted after the filtering the image, now with a LoG filter.
WORC includes the width of the Gaussian part of the filter as parameter:

.. code-block:: python

        config['ImageFeatures']['log_sigma'] = '1, 5, 10'

Again, for all sigma's, the images are filtered per 2-D slice after which the PREDICT histogram features
as :ref:`discussed earlier <features-histogram>` are extracted from the filtered images.

.. _features-Vessel:

Vessel filter features
----------------------
Similar to the Gabor features, these features are extracted after the filtering the image, now using a so called
vessel filter from the following paper:

    `Frangi, Alejandro F., et al. "Multiscale vessel enhancement filtering." International conference on medical image computing and computer-assisted intervention. Springer, Berlin, Heidelberg, 1998. <https://link.springer.com/chapter/10.1007/bfb0056195/>`_

As the filter triggers on tubular structeres, these filter may be used to not only detect vessels but any tube like
structure. The following parameters are used, see also the paper:

.. code-block:: python

        config['ImageFeatures']['vessel_scale_range'] = '1, 10'
        config['ImageFeatures']['vessel_scale_step'] = '2'
        config['ImageFeatures']['vessel_radius'] = '5'

As in several applications we were interested in vessel structures in the core of the ROI, WORC splits
the ROI in an inner and outer part using the vessel_radius parameter.

Again, for all parameter combinations, the images are filtered per 2-D slice and the PREDICT histogram features
as :ref:`discussed earlier <features-histogram>` are extracted from the filtered images. This is done for
the full ROI, the inner region, and the outer region.

.. _features-LBP:

Local Binary Patterns (LBP)
----------------------------
We recommend the following article for information about LBPs:

    `Ojala, Timo, Matti Pietikainen, and Topi Maenpaa. "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns." IEEE Transactions on pattern analysis and machine intelligence 24.7 (2002): 971-987. <https://ieeexplore.ieee.org/abstract/document/1017623/>`_

Again, a range of parameters is used to compute the LBP:

.. code-block:: python

        config['ImageFeatures']['LBP_radius'] = '3, 8, 15'
        config['ImageFeatures']['LBP_npoints'] = '12, 24, 36'

For all parameter combinations, as each npoints corresponds to a radius setting, the images are "filtered" (the LBP produces an image with the same
dimensions as the original, similar to a filtering operation) per 2-D slice and the PREDICT histogram features
as :ref:`discussed earlier <features-histogram>` are extracted from the filtered images, both for the inner and outer
region.

.. _features-phase:

Local phase features
--------------------
In many imaging modalities, e.g. MRI, the intensity scale varies a lot per image. Therefore, using intensity
information may not be relevant: changes in contrast in local regions may be more relevant. Therefore, PREDICT
includes features based on local phase, which transforms the image to an intensity invariant phase by
looking at fluctuations or the phase of the intensity in a local region. On these local phase images,
measures based on congruency or symmetry of phase may result in relevant features. For more information,
please see the work of `Peter Kovesi <https://www.peterkovesi.com/matlabfns/index.html/>`_.

Local phase computations serves as a filter, with the following parameters:

.. code-block:: python

        config['ImageFeatures']['phase_minwavelength'] = '3'
        config['ImageFeatures']['phase_nscale'] = '5'

Again, for all parameter combinations, the images are filtered per 2-D slice and the PREDICT histogram features
as :ref:`discussed earlier <features-histogram>` are extracted from the filtered images. This is done for
the local phase, phase congruency, and phase symmetry.

.. _features-dicom:

DICOM features
----------------
In PREDICT, several features may be extracted from DICOM headers, which can be provided in the metadata source.
By default, these include:

- ``(0010, 1010)``: Patient age
- ``(0010, 0040)``: Patient sex

You can define which tags you want to extract and how to name these features
by altering the following in the config:

.. code-block:: python

  config['ImageFeatures']['dicom_feature_tags'] = '0010 1010, 0010 0040'
  config['ImageFeatures']['dicom_feature_labels'] = 'age, sex'

Note that the value will be converted to a float. If that's not possible, or
the tag is not present, ``numpy.NaN`` will be used instead.

Other features may you want to include:

- ``(0008, 0070)``: Scanner manufacturer
- ``(0018, 0022)``: Scan options, see below
- ``(0018, 0050)``: Slice thickness
- ``(0018, 0080)``: Repetition time (MRI)
- ``(0018, 0081)``: Echo time (MRI)
- ``(0018, 0087)``: Magnetic field strength (MRI)
- ``(0018, 1314)``: Flip angle (MRI)
- ``(0028, 0030)``: Pixel spacing

Several routines for converting values to floats has been defined for the
following features:

- ``(0008, 0070)`` (Scanner manufacturer): 0 = Siemens, 1 = Philips,
  2 = General Electric, 3 = Toshiba. If not one of these, ``numpy.NaN`` is used.
- ``(0018, 0022)`` (Scan options): if name is 'FatSat', determine whether a
  a scan has been made with fat saturation or not from the scan options.
- ``(0010, 0040)`` (Patient Sex): M = 0, F = 1
- ``(0018, 0087)`` (Magnetic field strength): 5000 = 0.5, 10000 = 1.0,
  15000 = 1.5, 30000 = 3.0. If not convertible to float, use ``numpy.NaN``
- ``(0028, 0030)`` (Pixel spacing): Use first value and convert to float

.. _features-semantic:

Semantic features
-----------------
WORC allows the user to provide non-computational features, which are called semantic features. These
can be give to WORC as an Excel file, in which each column represents a feature. See the
:ref:`User manual chapter <usermanual-chapter>` for more details on providing these features


Other extraction choices
-----------------------------------

Filtering on ROI or full image.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For all filter based features, the images are first filtered using the full image, after which the features
are extracted from the region of interests (ROI). Only filtering the ROI with the filters would result in
edge artefacts. A drawback could be that now the ROI surroundings influence the feature, but this
can also be a benefit as a comparison between the ROI and it's surrounding could give relevant information.

Feature extraction parameter selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many of the extracted features have parameters to be set. For each application, the most suitable set of
parameters may vary. Therefore, in WORC, by default many features are extracted at a range of parameters.
We hypothesize that in the next steps, e.g. feature selection and classification, the most relevant features
will be automatically used.

Wavelet features
^^^^^^^^^^^^^^^^^
PyRadiomics supports the extraction of so-called wavelet features by first applying a set of filters
to the image before extracting the above mentioned features. The amount of features therefore quickly expands
when using wavelet features, while we have not noticed improvements in our experiments. Hence, to save
computation time, we have decided to only include original features in WORC. Usage of wavelet features
is however supported, both in feature extraction and selection, see the :ref:`Config chapter <config-chapter>`.

Fixed bin width vs fixed bin size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For all gray level matrix based features, WORC by default uses a fixed bin-width, while
`PyRadiomics argues to use a fixed bin-size <https://pyradiomics.readthedocs.io/en/latest/faq.html#what-about-gray-value-discretization-fixed-bin-width-fixed-bin-count/>`_
The reason for that is that we want the WORC default settings to work in a wide variety of applications,
including those with images in arbitrary scales, which often happens when using MRI. In these cases,
using a fixed bin-width may lead to odd features values and even errors.
