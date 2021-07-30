#!/usr/bin/env python
import os

import WORC
from fastr.helpers.rest_generation import create_rest_table


def generate_config():
    field_key = []
    field_subkey = []
    field_default = []
    field_description = []
    field_option = []

    a = WORC.WORC()

    config_defaults = a.defaultconfig()
    config_options = generate_config_options()
    config_descriptions = generate_config_descriptions()

    for key in config_defaults.keys():
        for num, subkey in enumerate(config_defaults[key].keys()):
            # print(f'[generate_config.py] Documenting field {key}: {subkey}')
            field_key.append(key)
            field_subkey.append(subkey)
            field_default.append(config_defaults[key][subkey])

            try:
                field_description.append(config_descriptions[key][subkey])
            except KeyError:
                print(f'[WARNING] No description for {key}: {subkey}')
                field_description.append('WIP')

            try:
                field_option.append(config_options[key][subkey])
            except KeyError:
                print(f'[WARNING] No options for {key}: {subkey}')
                field_option.append(config_defaults[key][subkey])

    data = [field_key, field_subkey, field_description, field_default, field_option]
    headers = ['Key', 'Subkey', 'Description', 'Default', 'Options',]

    return data, headers


def generate_config_doc():
    print('[generate_config.py] Generating config reference...')
    data, headers = generate_config()
    unique_keys = list(set(data[0]))
    if type(unique_keys) is not list:
        # Single number, convert to list
        unique_keys = [unique_keys]

    unique_keys.sort()

    # Per main section, create relevant tables
    for key in unique_keys:
        indices = [i for i, x in enumerate(data[0]) if x == key]
        subkeys = [data[1][i] for i in indices]
        descriptions = [data[2][i] for i in indices]
        defaults = [data[3][i] for i in indices]
        options = [data[4][i] for i in indices]

        # Create description table
        filename = os.path.join(os.path.dirname(__file__),
                                'autogen',
                                'config',
                                f'WORC.config_{key}_description.rst')
        headers_temp = ['Subkey', 'Description']
        data_temp = [subkeys, descriptions]
        table = create_rest_table(data_temp, headers_temp)

        with open(filename, 'w') as fh_out:
            fh_out.write(table)

        # Create defaults and options table
        filename = os.path.join(os.path.dirname(__file__),
                                'autogen',
                                'config',
                                f'WORC.config_{key}_defopts.rst')
        headers_temp = ['Subkey', 'Default', 'Options']
        data_temp = [subkeys, defaults, options]
        table = create_rest_table(data_temp, headers_temp)

        with open(filename, 'w') as fh_out:
            fh_out.write(table)

    # Create main table
    headers = ['Key', 'Reference']
    data = [unique_keys, [f':ref:`{h} <config-{h}>`' for h in unique_keys]]
    table = create_rest_table(data, headers)

    filename = os.path.join(os.path.dirname(__file__),
                            'autogen',
                            f'WORC.config.rst')
    with open(filename, 'w') as fh_out:
        fh_out.write(table)

    print(f'[generate_config.py] Config references saved!')


def generate_config_options():
    config = dict()

    # General configuration of WORC
    config['General'] = dict()
    config['General']['cross_validation'] = 'True, False'
    config['General']['Segmentix'] = 'True, False'
    config['General']['FeatureCalculators'] = 'predict/CalcFeatures:1.0, pyradiomics/Pyradiomics:1.0, pyradiomics/CF_pyradiomics:1.0, your own tool reference'
    config['General']['Preprocessing'] = 'worc/PreProcess:1.0, your own tool reference'
    config['General']['RegistrationNode'] = "'elastix4.8/Elastix:4.8', your own tool reference"
    config['General']['TransformationNode'] = "'elastix4.8/Transformix:4.8', your own tool reference"
    config['General']['Joblib_ncores'] = 'Integer > 0'
    config['General']['Joblib_backend'] = 'multiprocessing, threading'
    config['General']['tempsave'] = 'True, False'
    config['General']['AssumeSameImageAndMaskMetadata'] = 'True, False'
    config['General']['ComBat'] = 'True, False'

    # Segmentix
    config['Segmentix'] = dict()
    config['Segmentix']['mask'] = 'subtract, multiply'
    config['Segmentix']['segtype'] = 'None, Ring, Dilate'
    config['Segmentix']['segradius'] = 'Integer > 0'
    config['Segmentix']['N_blobs'] = 'Integer > 0'
    config['Segmentix']['fillholes'] = 'True, False'
    config['Segmentix']['remove_small_objects'] = 'True, False'
    config['Segmentix']['min_object_size'] = 'Integer > 0'

    # Preprocessing
    config['Preprocessing'] = dict()
    config['Preprocessing']['CheckSpacing'] = 'True, False'
    config['Preprocessing']['Clipping'] = 'True, False'
    config['Preprocessing']['Clipping_Range'] = 'Float, Float'
    config['Preprocessing']['Normalize'] = 'True, False'
    config['Preprocessing']['Normalize_ROI'] = 'True, False, Full'
    config['Preprocessing']['ROIDetermine'] = 'Provided, Otsu'
    config['Preprocessing']['ROIdilate'] = 'True, False'
    config['Preprocessing']['ROIdilateradius'] = 'Integer > 0'
    config['Preprocessing']['Method'] = 'z_score, minmed'
    config['Preprocessing']['Resampling'] = 'True, False'
    config['Preprocessing']['Resampling_spacing'] = 'Float, Float, Float'
    config['Preprocessing']['BiasCorrection'] = 'True, False'
    config['Preprocessing']['BiasCorrection_Mask'] = 'Float, Float, Float'
    config['Preprocessing']['CheckOrientation'] = 'True, False'
    config['Preprocessing']['OrientationPrimaryAxis'] = 'axial'

    # PREDICT - Feature calculation
    # Determine which features are calculated
    config['ImageFeatures'] = dict()
    config['ImageFeatures']['shape'] = 'True, False'
    config['ImageFeatures']['histogram'] = 'True, False'
    config['ImageFeatures']['orientation'] = 'True, False'
    config['ImageFeatures']['texture_Gabor'] = 'True, False'
    config['ImageFeatures']['texture_LBP'] = 'True, False'
    config['ImageFeatures']['texture_GLCM'] = 'True, False'
    config['ImageFeatures']['texture_GLCMMS'] = 'True, False'
    config['ImageFeatures']['texture_GLRLM'] = 'True, False'
    config['ImageFeatures']['texture_GLSZM'] = 'True, False'
    config['ImageFeatures']['texture_NGTDM'] = 'True, False'
    config['ImageFeatures']['coliage'] = 'True, False'
    config['ImageFeatures']['vessel'] = 'True, False'
    config['ImageFeatures']['log'] = 'True, False'
    config['ImageFeatures']['phase'] = 'True, False'

    # Parameter settings for PREDICT feature calculation
    # Defines what should be done with the images
    config['ImageFeatures']['image_type'] = 'CT'

    # Define frequencies for gabor filter in pixels
    config['ImageFeatures']['gabor_frequencies'] = 'Float(s)'

    # Gabor, GLCM angles in degrees and radians, respectively
    config['ImageFeatures']['gabor_angles'] = 'Integer(s)'
    config['ImageFeatures']['GLCM_angles'] = 'Float(s)'

    # GLCM discretization levels, distances in pixels
    config['ImageFeatures']['GLCM_levels'] = 'Integer > 0'
    config['ImageFeatures']['GLCM_distances'] = 'Integer(s) > 0'

    # LBP radius, number of points in pixels
    config['ImageFeatures']['LBP_radius'] = 'Integer(s) > 0'
    config['ImageFeatures']['LBP_npoints'] = 'Integer(s) > 0'

    # Phase features minimal wavelength and number of scales
    config['ImageFeatures']['phase_minwavelength'] = 'Integer > 0'
    config['ImageFeatures']['phase_nscale'] = 'Integer > 0'

    # Log features sigma of Gaussian in pixels
    config['ImageFeatures']['log_sigma'] = 'Integer(s)'

    # Vessel features scale range, steps for the range
    config['ImageFeatures']['vessel_scale_range'] = 'Two integers: min and max.'
    config['ImageFeatures']['vessel_scale_step'] = 'Integer > 0'

    # Vessel features radius for erosion to determine boudnary
    config['ImageFeatures']['vessel_radius'] = 'Integer > 0'

    # Tags from which to extract features, and how to name them
    config['ImageFeatures']['dicom_feature_tags'] = "DICOM tag keys, e.g. 0010 0010, separated by comma's"
    config['ImageFeatures']['dicom_feature_labels'] = 'List of strings'

    # Pyradiomics - feature calculation
    config['PyRadiomics'] = dict()
    config['PyRadiomics']['geometryTolerance'] = 'Float'
    config['PyRadiomics']['Preprocessing'] = 'True, False'
    config['PyRadiomics']['normalize'] = 'Boolean'
    config['PyRadiomics']['normalizeScale'] = 'Integer'
    config['PyRadiomics']['resampledPixelSpacing'] = 'Float, Float, Float'
    config['PyRadiomics']['interpolator'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html?highlight=sitkbspline#feature-extractor-level/>`_ .'
    config['PyRadiomics']['preCrop'] = 'True, False'
    config['PyRadiomics']['binCount'] = 'Integer or None' # BinWidth to sensitive for normalization, thus use binCount
    config['PyRadiomics']['binWidth'] = 'Integer or None'
    config['PyRadiomics']['force2D'] = 'True, False'
    config['PyRadiomics']['force2Ddimension'] = '0 = axial, 1 = coronal, 2 = sagital'  # axial slices, for coronal slices, use dimension 1 and for sagittal, dimension 2.
    config['PyRadiomics']['voxelArrayShift'] = 'Integer'
    config['PyRadiomics']['Original'] = 'True, False'
    config['PyRadiomics']['Wavelet'] = 'True, False'
    config['PyRadiomics']['LoG'] = 'True, False'
    config['PyRadiomics']['label'] = 'Integer'

    # Enabled PyRadiomics features
    config['PyRadiomics']['extract_firstorder'] = 'True, False'
    config['PyRadiomics']['extract_shape'] = 'True, False'
    config['PyRadiomics']['texture_GLCM'] = 'True, False'
    config['PyRadiomics']['texture_GLRLM'] = 'True, False'
    config['PyRadiomics']['texture_GLSZM'] = 'True, False'
    config['PyRadiomics']['texture_GLDM'] = 'True, False'
    config['PyRadiomics']['texture_NGTDM'] = 'True, False'

    # ComBat Feature Harmonization
    config['ComBat'] = dict()
    config['ComBat']['language'] = 'python, matlab'
    config['ComBat']['batch'] = 'String'
    config['ComBat']['mod'] = 'String(s), or []'
    config['ComBat']['par'] = '0 or 1'
    config['ComBat']['eb'] = '0 or 1'
    config['ComBat']['per_feature'] = '0 or 1'
    config['ComBat']['excluded_features'] = 'List of strings, comma separated'
    config['ComBat']['matlab'] = 'String'

    # Feature preprocessing before all below takes place
    config['FeatPreProcess'] = dict()
    config['FeatPreProcess']['Use'] = 'Boolean'
    config['FeatPreProcess']['Combine'] = 'Boolean'
    config['FeatPreProcess']['Combine_method'] = 'mean or max'

    # Feature selection
    config['Featsel'] = dict()
    config['Featsel']['Variance'] = 'Float'
    config['Featsel']['GroupwiseSearch'] = 'Boolean(s)'
    config['Featsel']['SelectFromModel'] = 'Float'
    config['Featsel']['SelectFromModel_estimator'] = 'Lasso, LR, RF'
    config['Featsel']['SelectFromModel_lasso_alpha'] = 'Two Floats: loc and scale'
    config['Featsel']['SelectFromModel_n_trees'] = 'Two Integers: loc and scale'
    config['Featsel']['UsePCA'] = 'Float'
    config['Featsel']['PCAType'] = 'Integer(s), 95variance'
    config['Featsel']['StatisticalTestUse'] = 'Float'
    config['Featsel']['StatisticalTestMetric'] = 'ttest, Welch, Wilcoxon, MannWhitneyU'
    config['Featsel']['StatisticalTestThreshold'] = 'Two Integers: loc and scale'
    config['Featsel']['ReliefUse'] = 'Float'
    config['Featsel']['ReliefNN'] = 'Two Integers: loc and scale'
    config['Featsel']['ReliefSampleSize'] = 'Two Floats: loc and scale'
    config['Featsel']['ReliefDistanceP'] = 'Two Integers: loc and scale'
    config['Featsel']['ReliefNumFeatures'] = 'Two Integers: loc and scale'

    # Groupwie Featureselection options
    config['SelectFeatGroup'] = dict()
    config['SelectFeatGroup']['shape_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['histogram_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['orientation_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_Gabor_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLCM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLDM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLCMMS_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLDM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLRLM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLSZM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLDZM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_NGLDM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_NGTDM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_LBP_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['dicom_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['semantic_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['coliage_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['log_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['vessel_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['phase_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['fractal_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['location_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['rgrd_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['wavelet_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['original_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['toolbox'] = 'All, or name of toolbox (PREDICT, PyRadiomics)'

    # Feature OneHotEncoding
    config['OneHotEncoding'] = dict()
    config['OneHotEncoding']['Use'] = 'Boolean(s)'
    config['OneHotEncoding']['feature_labels_tofit'] = 'List of strings'

    # Feature imputation
    config['Imputation'] = dict()
    config['Imputation']['use'] = 'Boolean(s)'
    config['Imputation']['strategy'] = 'mean, median, most_frequent, constant, knn'
    config['Imputation']['n_neighbors'] = 'Two Integers: loc and scale'

    # Resampling options
    config['Resampling'] = dict()
    config['Resampling']['Use'] = 'Float'
    config['Resampling']['Method'] =\
        'RandomUnderSampling, RandomOverSampling, NearMiss, ' +\
        'NeighbourhoodCleaningRule, ADASYN, BorderlineSMOTE, SMOTE, ' +\
        'SMOTEENN, SMOTETomek'
    config['Resampling']['sampling_strategy'] = 'auto, majority, not minority, not majority, all'
    config['Resampling']['n_neighbors'] = 'Two Integers: loc and scale'
    config['Resampling']['k_neighbors'] = 'Two Integers: loc and scale'
    config['Resampling']['threshold_cleaning'] = 'Two Floats: loc and scale'

    # Classification
    config['Classification'] = dict()
    config['Classification']['fastr'] = 'True, False'
    config['Classification']['fastr_plugin'] = 'Any `fastr execution plugin <https://fastr.readthedocs.io/en/develop/_autogen/fastr.reference.html#executionplugin-reference/>`_ .'
    config['Classification']['classifiers'] = 'SVM , SVR, SGD, SGDR, RF, LDA, QDA, ComplementND, GaussianNB, AdaBoostClassifier, XGBClassifier, LR, RFR, Lasso, ElasticNet, LinR, Ridge, AdaBoostRegressor, XGBRegressor. All are estimators from `sklearn <https://scikit-learn.org/stable//>`_ '
    config['Classification']['max_iter'] = 'Integer'
    config['Classification']['SVMKernel'] = 'poly, linear, rbf'
    config['Classification']['SVMC'] = 'Two Integers: loc and scale'
    config['Classification']['SVMdegree'] = 'Two Integers: loc and scale'
    config['Classification']['SVMcoef0'] = 'Two Integers: loc and scale'
    config['Classification']['SVMgamma'] = 'Two Integers: loc and scale'
    config['Classification']['RFn_estimators'] = 'Two Integers: loc and scale'
    config['Classification']['RFmin_samples_split'] = 'Two Integers: loc and scale'
    config['Classification']['RFmax_depth'] = 'Two Integers: loc and scale'
    config['Classification']['LRpenalty'] = 'none, l2, l1'
    config['Classification']['LRC'] = 'Two Floats: loc and scale'
    config['Classification']['LR_solver'] = 'Comma separated list of strings, for the options see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'
    config['Classification']['LR_l1_ratio'] = 'Float between 0.0 and 1.0.'
    config['Classification']['LDA_solver'] = 'svd, lsqr, eigen'
    config['Classification']['LDA_shrinkage'] = 'Two Integers: loc and scale'
    config['Classification']['QDA_reg_param'] = 'Two Integers: loc and scale'
    config['Classification']['ElasticNet_alpha'] = 'Two Integers: loc and scale'
    config['Classification']['ElasticNet_l1_ratio'] = 'Two Integers: loc and scale'
    config['Classification']['SGD_alpha'] = 'Two Integers: loc and scale'
    config['Classification']['SGD_l1_ratio'] = 'Two Integers: loc and scale'
    config['Classification']['SGD_loss'] = 'hinge, squared_hinge, modified_huber'
    config['Classification']['SGD_penalty'] = 'none, l2, l1'
    config['Classification']['CNB_alpha'] = 'Two Integers: loc and scale'
    config['Classification']['AdaBoost_n_estimators'] = 'Two Integers: loc and scale'
    config['Classification']['AdaBoost_learning_rate'] = 'Two Floats: loc and scale'
    config['Classification']['XGB_boosting_rounds'] = 'Two Integers: loc and scale'
    config['Classification']['XGB_max_depth'] = 'Two Integers: loc and scale'
    config['Classification']['XGB_learning_rate'] = config['Classification']['AdaBoost_learning_rate']
    config['Classification']['XGB_gamma'] = 'Two Floats: loc and scale'
    config['Classification']['XGB_min_child_weight'] = 'Two Integers: loc and scale'
    config['Classification']['XGB_colsample_bytree'] = 'Two Floats: loc and scale'

    # CrossValidation
    config['CrossValidation'] = dict()
    config['CrossValidation']['Type'] = 'random_split, LOO'
    config['CrossValidation']['N_iterations'] = 'Integer'
    config['CrossValidation']['test_size'] = 'Float'
    config['CrossValidation']['fixed_seed'] = 'Boolean'

    # Options for the object/patient labels that are used
    config['Labels'] = dict()
    config['Labels']['label_names'] = 'String(s)'
    config['Labels']['modus'] = 'singlelabel, multilabel'

    # Hyperparameter optimization options
    config['HyperOptimization'] = dict()
    config['HyperOptimization']['scoring_method'] = 'Manual metric by WORC: f1_weighted_predictproba, average_precision_weighted, gmean. Other accepted values are any `sklearn metric <https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values/>`_'
    config['HyperOptimization']['test_size'] = 'Float'
    config['HyperOptimization']['N_iterations'] = 'Integer'
    config['HyperOptimization']['n_jobspercore'] = 'Integer'
    config['HyperOptimization']['n_splits'] = 'Integer'
    config['HyperOptimization']['maxlen'] = 'Integer'
    config['HyperOptimization']['ranking_score'] = 'String'
    config['HyperOptimization']['memory'] = 'String consisting of integer + "G"'
    config['HyperOptimization']['refit_workflows'] = 'Boolean'

    # Feature scaling options
    config['FeatureScaling'] = dict()
    config['FeatureScaling']['skip_features'] = 'Comma separated list of strings'
    config['FeatureScaling']['scaling_method'] = 'robust_z_score, z_score, robust, minmax, log_z_score, None'

    # Ensemble options
    config['Ensemble'] = dict()
    config['Ensemble']['Use'] = 'Integer'
    config['Ensemble']['Metric'] = 'Default, generalization'

    # Evaluation options
    config['Evaluation'] = dict()
    config['Evaluation']['OverfitScaler'] = 'True, False'

    # Bootstrap options
    config['Bootstrap'] = dict()
    config['Bootstrap']['Use'] = 'Boolean'
    config['Bootstrap']['N_iterations'] = 'Integer'

    return config


def generate_config_descriptions():
    config = dict()

    # General configuration of WORC
    config['General'] = dict()
    config['General']['cross_validation'] = 'Determine whether a cross validation will be performed or not. Obsolete, will be removed.'
    config['General']['Segmentix'] = 'Determine whether to use Segmentix tool for segmentation preprocessing.'
    config['General']['FeatureCalculators'] = 'Specifies which feature calculation tools should be used. A list can be provided to use multiple tools.'
    config['General']['Preprocessing'] = 'Specifies which tool will be used for image preprocessing.'
    config['General']['RegistrationNode'] = "Specifies which tool will be used for image registration."
    config['General']['TransformationNode'] = "Specifies which tool will be used for applying image transformations."
    config['General']['Joblib_ncores'] = 'Number of cores to be used by joblib for multicore processing.'
    config['General']['Joblib_backend'] = 'Type of backend to be used by joblib for multicore processing.'
    config['General']['tempsave'] = 'Determines whether after every cross validation iteration the result will be saved, in addition to the result after all iterations. Especially useful for debugging.'
    config['General']['AssumeSameImageAndMaskMetadata'] = 'Make the assumption that the image and mask have the same metadata. If True and there is a mismatch, metadata from the image will be copied to the mask.'
    config['General']['ComBat'] = 'Whether to use ComBat feature harmonization on your FULL dataset, i.e. not in a train-test setting. See <https://github.com/Jfortin1/ComBatHarmonization for more information./>`_ .'

    # Segmentix
    config['Segmentix'] = dict()
    config['Segmentix']['mask'] = 'If a mask is supplied, should the mask be subtracted from the contour or multiplied.'
    config['Segmentix']['segtype'] = 'If Ring, then a ring around the segmentation will be used as contour. If Dilate, the segmentation will be dilated per 2-D axial slice with a disc.'
    config['Segmentix']['segradius'] = 'Define the radius of the ring or disc used if segtype is Ring or Dilate, respectively.'
    config['Segmentix']['N_blobs'] = 'How many of the largest blobs are extracted from the segmentation. If None, no blob extraction is used.'
    config['Segmentix']['fillholes'] = 'Determines whether hole filling will be used.'
    config['Segmentix']['remove_small_objects'] = 'Determines whether small objects will be removed.'
    config['Segmentix']['min_object_size'] = 'Minimum of objects in voxels to not be removed if small objects are removed'

    # Preprocessing
    config['Preprocessing'] = dict()
    config['Preprocessing']['Clipping'] = 'Determine whether to use intensity clipping in preprocessing of image or not.'
    config['Preprocessing']['Clipping_Range'] = 'Lower- and upperbound of intensities to be used in clipping.'
    config['Preprocessing']['CheckSpacing'] = 'Determine whether to check the spacing or not. If True, and the spacing of the image is [1x1x1], we assume the spacing is incorrect, and overwrite it using the DICOM metadata.'
    config['Preprocessing']['Normalize'] = 'Determine whether to use normalization in preprocessing of image or not.'
    config['Preprocessing']['Normalize_ROI'] = 'If a mask is supplied and this is set to True, normalize image based on supplied ROI. Otherwise, the full image is used for normalization using the SimpleITK Normalize function. Lastly, setting this to False will result in no normalization being applied.'
    config['Preprocessing']['ROIDetermine'] = 'Choose whether a ROI for normalization is provided, or Otsu thresholding is used to determine one.'
    config['Preprocessing']['ROIdilate'] = 'Determine whether the ROI has to be dilated with a disc element or not.'
    config['Preprocessing']['ROIdilateradius'] = 'Radius of disc element to be used in ROI dilation.'
    config['Preprocessing']['Method'] = 'Method used for normalization if ROI is supplied. Currently, z-scoring or using the minimum and median of the ROI can be used.'
    config['Preprocessing']['Resampling'] = 'Determine whether the image and mask will be resampled or not.'
    config['Preprocessing']['Resampling_spacing'] = 'Spacing to resample image and mask to, if resampling is used.'
    config['Preprocessing']['BiasCorrection'] = 'Determine whether N4 Bias correction will be applied or not.'
    config['Preprocessing']['BiasCorrection_Mask'] = 'Whether withing bias correction, a mask generated through Otsu thresholding is used or not.'
    config['Preprocessing']['CheckOrientation'] = 'Determine whether to check the image orientation or not. If checked, if the orientation is not equal to the OrientationPrimaryAxis, the image is rotated.'
    config['Preprocessing']['OrientationPrimaryAxis'] = 'If CheckOrientation is True, if primary axis is not this one, rotate image such that it is. Currently, only "axial" is supported.'

    # PREDICT - Feature calculation
    # Determine which features are calculated
    config['ImageFeatures'] = dict()
    config['ImageFeatures']['shape'] = 'Determine whether orientation features are computed or not.'
    config['ImageFeatures']['histogram'] = 'Determine whether histogram features are computed or not.'
    config['ImageFeatures']['orientation'] = 'Determine whether orientation features are computed or not.'
    config['ImageFeatures']['texture_Gabor'] = 'Determine whether Gabor texture features are computed or not.'
    config['ImageFeatures']['texture_LBP'] ='Determine whether LBP texture features are computed or not.'
    config['ImageFeatures']['texture_GLCM'] = 'Determine whether GLCM texture features are computed or not.'
    config['ImageFeatures']['texture_GLCMMS'] = 'Determine whether GLCM Multislice texture features are computed or not.'
    config['ImageFeatures']['texture_GLDM'] = 'Determine whether GLDM texture features are computed or not.'
    config['ImageFeatures']['texture_GLRLM'] = 'Determine whether GLRLM texture features are computed or not.'
    config['ImageFeatures']['texture_GLSZM'] = 'Determine whether GLSZM texture features are computed or not.'
    config['ImageFeatures']['texture_NGTDM'] = 'Determine whether NGTDM texture features are computed or not.'
    config['ImageFeatures']['coliage'] = 'Determine whether coliage features are computed or not.'
    config['ImageFeatures']['vessel'] = 'Determine whether vessel features are computed or not.'
    config['ImageFeatures']['log'] = 'Determine whether LoG features are computed or not.'
    config['ImageFeatures']['phase'] = 'Determine whether local phase features are computed or not.'

    # Parameter settings for PREDICT feature calculation
    # Defines what should be done with the images
    config['ImageFeatures']['image_type'] = 'Modality of images supplied. Determines how the image is loaded.'

    # Define frequencies for gabor filter in pixels
    config['ImageFeatures']['gabor_frequencies'] = 'Frequencies of Gabor filters used: can be a single float or a list.'

    # Gabor, GLCM angles in degrees and radians, respectively
    config['ImageFeatures']['gabor_angles'] = 'Angles of Gabor filters in degrees: can be a single integer or a list.'
    config['ImageFeatures']['GLCM_angles'] = 'Angles used in GLCM computation in radians: can be a single float or a list.'

    # GLCM discretization levels, distances in pixels
    config['ImageFeatures']['GLCM_levels'] = 'Number of grayscale levels used in discretization before GLCM computation.'
    config['ImageFeatures']['GLCM_distances'] = 'Distance(s) used in GLCM computation in pixels: can be a single integer or a list.'

    # LBP radius, number of points in pixels
    config['ImageFeatures']['LBP_radius'] = 'Radii used for LBP computation: can be a single integer or a list.'
    config['ImageFeatures']['LBP_npoints'] = 'Number(s) of points used in LBP computation: can be a single integer or a list.'

    # Phase features minimal wavelength and number of scales
    config['ImageFeatures']['phase_minwavelength'] = 'Minimal wavelength in pixels used for phase features.'
    config['ImageFeatures']['phase_nscale'] = 'Number of scales used in phase feature computation.'

    # Log features sigma of Gaussian in pixels
    config['ImageFeatures']['log_sigma'] = 'Standard deviation(s) in pixels used in log feature computation: can be a single integer or a list.'

    # Vessel features scale range, steps for the range
    config['ImageFeatures']['vessel_scale_range'] = 'Scale in pixels used for Frangi vessel filter. Given as a minimum and a maximum.'
    config['ImageFeatures']['vessel_scale_step'] = 'Step size used to go from minimum to maximum scale on Frangi vessel filter.'

    # Vessel features radius for erosion to determine boudnary
    config['ImageFeatures']['vessel_radius'] = 'Radius to determine boundary of between inner part and edge in Frangi vessel filter.'

    # Tags from which to extract features, and how to name them
    config['ImageFeatures']['dicom_feature_tags'] = "DICOM tags to be extracted as features. See https://worc.readthedocs.io/en/latest/static/features.html."
    config['ImageFeatures']['dicom_feature_labels'] = "For each of the DICOM tag values extracted, name that should be assigned to the feature. See https://worc.readthedocs.io/en/latest/static/features.html."

    # Pyradiomics - feature calculation
    config['PyRadiomics'] = dict()
    config['PyRadiomics']['geometryTolerance'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['Preprocessing'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['normalize'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['normalizeScale'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['resampledPixelSpacing'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['interpolator'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html?highlight=sitkbspline#feature-extractor-level/>`_ .'
    config['PyRadiomics']['preCrop'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['binCount'] = 'We advice to use a fixed bin count instead of a fixed bin width, as on imaging modalities such as MRI, the scale of the values varies a lot, which is incompatible with a fixed bin width. See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['binWidth'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['force2D'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['force2Ddimension'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['voxelArrayShift'] = 'See <https://pyradiomics.readthedocs.io/en/latest/customization.html/>`_ .'
    config['PyRadiomics']['Original'] = 'Enable/Disable computation of original image features.'
    config['PyRadiomics']['Wavelet'] = 'Enable/Disable computation of wavelet image features.'
    config['PyRadiomics']['LoG'] = 'Enable/Disable computation of Laplacian of Gaussian (LoG) image features.'
    config['PyRadiomics']['label'] = '"Intensity" of the pixels in the mask to be used for feature extraction. If using segmentix, use 1, as your mask will be boolean. Otherwise, select the integer(s) corresponding to the ROI in your mask.'

    # Enabled PyRadiomics features
    config['PyRadiomics']['extract_firstorder'] = 'Determine whether first order features are computed or not.'
    config['PyRadiomics']['extract_shape'] = 'Determine whether shape features are computed or not.'
    config['PyRadiomics']['texture_GLCM'] = 'Determine whether GLCM features are computed or not.'
    config['PyRadiomics']['texture_GLRLM'] = 'Determine whether GLRLM features are computed or not.'
    config['PyRadiomics']['texture_GLSZM'] = 'Determine whether GLSZM features are computed or not.'
    config['PyRadiomics']['texture_GLDM'] = 'Determine whether GLDM features are computed or not.'
    config['PyRadiomics']['texture_NGTDM'] = 'Determine whether NGTDM features are computed or not.'

    # ComBat Feature Harmonization
    config['ComBat'] = dict()
    config['ComBat']['language'] = 'Name of software implementation to use.'
    config['ComBat']['batch'] = 'Name of batch variable = variable to correct for.'
    config['ComBat']['mod'] = 'Name of moderation variable(s) = variables for which variation in features will be "preserverd".'
    config['ComBat']['par'] = 'Either use the parametric (1) or non-parametric version (0) of ComBat.'
    config['ComBat']['eb'] = 'Either use the emperical Bayes (1) or simply mean shifting version (0) of ComBat.'
    config['ComBat']['per_feature'] = 'Either use ComBat for all features combined (0) or per feature (1), in which case a second feature equal to the single feature plus random noise will be added if eb=1'
    config['ComBat']['excluded_features'] = 'Provide substrings of feature labels of features which should be excluded from ComBat. Recommended to use for features unaffected by the batch variable.'
    config['ComBat']['matlab'] = 'If using Matlab, path to Matlab executable.'

    # Feature preprocessing before all below takes place
    config['FeatPreProcess'] = dict()
    config['FeatPreProcess']['Use'] = 'If True, use feature preprocessor in the classify node. Currently excluded features with >80% NaNs.'
    config['FeatPreProcess']['Combine'] = 'If True, features of multiple objects (e.g. lesions) of the same patient are combined.'
    config['FeatPreProcess']['Combine_method'] = 'If features of multiple objects are combined, this determines the method. Currently included options are mean and max.'

    # Feature selection
    config['Featsel'] = dict()
    config['Featsel']['Variance'] = 'Percentage of times features which have a variance < 0.01 are excluded. Based on ` sklearn"s VarianceThreshold <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html/>`_.'
    config['Featsel']['GroupwiseSearch'] = 'Randomly select which feature groups to use. Parameters determined by the SelectFeatGroup config part, see below.'
    config['Featsel']['SelectFromModel'] = 'Percentage of times features are selected by first training a machine learning model which can rank the features with an ``importance''. See also `sklearn"s SelectFromModel <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html/>`_.'
    config['Featsel']['SelectFromModel_estimator'] = 'Machine learning model / estimator used: can be LASSO, LogisticRegression, or a Random Forest'
    config['Featsel']['SelectFromModel_lasso_alpha'] = "When using LASSO, search space of weigth of L1 term, see also `sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html/>`."
    config['Featsel']['SelectFromModel_n_trees'] = 'When using a random forest, search space of number of trees used.'
    config['Featsel']['UsePCA'] = 'Percentage of times Principle Component Analysis (PCA) is used to select features.'
    config['Featsel']['PCAType'] = 'Method to select number of components using PCA: Either the number of components that explains 95% of the variance, or use a fixed number of components.95variance'
    config['Featsel']['StatisticalTestUse'] = 'Percentage of times a statistical test is used to select features.'
    config['Featsel']['StatisticalTestMetric'] = 'Define the type of statistical test to be used.'
    config['Featsel']['StatisticalTestThreshold'] = 'Specify a threshold for the p-value threshold used in the statistical test to select features. The first element defines the lower boundary, the other the upper boundary. Random sampling will occur between the boundaries.'
    config['Featsel']['ReliefUse'] = 'Percentage of times Relief is used to select features.'
    config['Featsel']['ReliefNN'] = 'Min and max of number of nearest neighbors search range in Relief.'
    config['Featsel']['ReliefSampleSize'] = 'Min and max of sample size search range in Relief.'
    config['Featsel']['ReliefDistanceP'] = 'Min and max of positive distance search range in Relief.'
    config['Featsel']['ReliefNumFeatures'] = 'Min and max of number of features that is selected search range in Relief.'

    # Groupwie Featureselection options
    config['SelectFeatGroup'] = dict()
    config['SelectFeatGroup']['shape_features'] = 'If True, use shape features in model.'
    config['SelectFeatGroup']['histogram_features'] = 'If True, use histogram features in model.'
    config['SelectFeatGroup']['orientation_features'] = 'If True, use orientation features in model.'
    config['SelectFeatGroup']['texture_Gabor_features'] = 'If True, use Gabor texture features in model.'
    config['SelectFeatGroup']['texture_GLCM_features'] = 'If True, use GLCM texture features in model.'
    config['SelectFeatGroup']['texture_GLDM_features'] = 'If True, use GLDM texture features in model.'
    config['SelectFeatGroup']['texture_GLCMMS_features'] = 'If True, use GLCM Multislice texture features in model.'
    config['SelectFeatGroup']['texture_GLRLM_features'] = 'If True, use GLRLM texture features in model.'
    config['SelectFeatGroup']['texture_GLSZM_features'] = 'If True, use GLSZM texture features in model.'
    config['SelectFeatGroup']['texture_GLDZM_features'] = 'If True, use GLDZM texture features in model.'
    config['SelectFeatGroup']['texture_NGTDM_features'] = 'If True, use NGTDM texture features in model.'
    config['SelectFeatGroup']['texture_NGLDM_features'] = 'If True, use NGLDM texture features in model.'
    config['SelectFeatGroup']['texture_LBP_features'] = 'If True, use LBP texture features in model.'
    config['SelectFeatGroup']['dicom_features'] = 'If True, use DICOM features in model.'
    config['SelectFeatGroup']['semantic_features'] = 'If True, use semantic features in model.'
    config['SelectFeatGroup']['coliage_features'] = 'If True, use coliage features in model.'
    config['SelectFeatGroup']['log_features'] = 'If True, use log features in model.'
    config['SelectFeatGroup']['vessel_features'] = 'If True, use vessel features in model.'
    config['SelectFeatGroup']['phase_features'] = 'If True, use phase features in model.'
    config['SelectFeatGroup']['fractal_features'] = 'If True, use fractal features in model.'
    config['SelectFeatGroup']['location_features'] = 'If True, use location features in model.'
    config['SelectFeatGroup']['rgrd_features'] = 'If True, use rgrd features in model.'
    config['SelectFeatGroup']['wavelet_features'] = 'If True, use wavelet features in model.'
    config['SelectFeatGroup']['original_features'] = 'If True, use original features in model.'
    config['SelectFeatGroup']['toolbox'] = 'List of names of toolboxes to be used, or All'

    # Feature OneHotEncoding
    config['OneHotEncoding'] = dict()
    config['OneHotEncoding']['Use'] = 'If True, use OneHotEncoding for specific features as determined by the field below.'
    config['OneHotEncoding']['feature_labels_tofit'] = 'Labels of features for which to use OneHotEncoding. WORC will check whether any of the values specified in this field is a substring of a feature name. For example, if you give gclm, all features for which glcm is in the feature label will be one hot encoded.'

    # Feature imputation
    config['Imputation'] = dict()
    config['Imputation']['use'] = 'If True, use feature imputation methods to replace NaN values. If False, all NaN features will be set to zero.'
    config['Imputation']['strategy'] = 'Method to be used for imputation.'
    config['Imputation']['n_neighbors'] = 'When using k-Nearest Neighbors (kNN) for feature imputation, determines the number of neighbors used for imputation. Can be a single integer or a list.'

    # Resampling options
    config['Resampling'] = dict()
    config['Resampling']['Use'] = 'Percentage of times Object (e.g. patient) resampling is used.'
    config['Resampling']['Method'] = 'One of the methods adopted, see also imbalanced learn <https://imbalanced-learn.readthedocs.io/en/stable/api/>`_. '
    config['Resampling']['sampling_strategy'] = 'Sampling strategy, see also imbalanced learn <https://imbalanced-learn.readthedocs.io/en/stable/api/>`_. '
    config['Resampling']['n_neighbors'] = 'Number of n_neighbors used in resampling. This should be (much) smaller than the number of objects/patients you supply. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).'
    config['Resampling']['k_neighbors'] = 'Number of n_neighbors used in resampling. This should be (much) smaller than the number of objects/patients you supply. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).'
    config['Resampling']['threshold_cleaning'] = 'Threshold for cleaning of samples. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).'

    # Classification
    config['Classification'] = dict()
    config['Classification']['fastr'] = 'Use fastr for the optimization gridsearch (recommended on clusters, default) or if set to False , joblib (recommended for PCs but not on Windows).'
    config['Classification']['fastr_plugin'] = 'Name of execution plugin to be used. Default use the same as the self.fastr_plugin for the WORC object.'
    config['Classification']['classifiers'] = "Select the estimator(s) to use. Most are implemented using `sklearn <https://scikit-learn.org/stable/>`_. For abbreviations, see the options: LR = logistic regression."
    config['Classification']['max_iter'] = 'Maximum number of iterations to use in training an estimator. Only for specific estimators, see `sklearn <https://scikit-learn.org/stable/>`_.'
    config['Classification']['SVMKernel'] = 'When using a SVM, specify the kernel type.'
    config['Classification']['SVMC'] = 'Range of the SVM slack parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale).'
    config['Classification']['SVMdegree'] = 'Range of the SVM polynomial degree when using a polynomial kernel. We sample on a uniform scale: the parameters specify the range (loc, loc + scale). '
    config['Classification']['SVMcoef0'] = 'Range of SVM homogeneity parameter. We sample on a uniform scale: the parameters specify the range (loc, loc + scale). '
    config['Classification']['SVMgamma'] = 'Range of the SVM gamma parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale)'
    config['Classification']['RFn_estimators'] = 'Range of number of trees in a RF. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).'
    config['Classification']['RFmin_samples_split'] = 'Range of minimum number of samples required to split a branch in a RF. We sample on a uniform scale: the parameters specify the range (loc, loc + scale). '
    config['Classification']['RFmax_depth'] = 'Range of maximum depth of a RF. We sample on a uniform scale: the parameters specify the range (loc, loc + scale). '
    config['Classification']['LRpenalty'] = 'Penalty term used in LR.'
    config['Classification']['LRC'] = 'Range of regularization strength in LR. We sample on a uniform scale: the parameters specify the range (loc, loc + scale). '
    config['Classification']['LR_solver'] = 'Solver used in LR.'
    config['Classification']['LR_l1_ratio'] = 'Ratio between l1 and l2 penalty when using elasticnet penalty, see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.'
    config['Classification']['LDA_solver'] = 'Solver used in LDA.'
    config['Classification']['LDA_shrinkage'] = 'Range of the LDA shrinkage parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale).'
    config['Classification']['QDA_reg_param'] = 'Range of the QDA regularization parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale). '
    config['Classification']['ElasticNet_alpha'] = 'Range of the ElasticNet penalty parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale).'
    config['Classification']['ElasticNet_l1_ratio'] = 'Range of l1 ratio in LR. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).'
    config['Classification']['SGD_alpha'] = 'Range of the SGD penalty parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (loc, loc + scale).'
    config['Classification']['SGD_l1_ratio'] = 'Range of l1 ratio in SGD. We sample on a uniform scale: the parameters specify the range (loc, loc + scale).'
    config['Classification']['SGD_loss'] = 'Loss function of SGD.'
    config['Classification']['SGD_penalty'] = 'Penalty term in SGD.'
    config['Classification']['CNB_alpha'] = 'Regularization strenght in ComplementNB. We sample on a uniform scale: the parameters specify the range (loc, loc + scale)'
    config['Classification']['AdaBoost_n_estimators'] = "Number of estimators used in AdaBoost. Default is equal to config['Classification']['RFn_estimators']."
    config['Classification']['AdaBoost_learning_rate'] = 'Learning rate in AdaBoost.'
    config['Classification']['XGB_boosting_rounds'] = "Number of estimators / boosting rounds used in XGB. Default is equal to config['Classification']['RFn_estimators']."
    config['Classification']['XGB_max_depth'] = 'Maximum depth of XGB.'
    config['Classification']['XGB_learning_rate'] = "Learning rate in AdaBoost. Default is equal to config['Classification']['AdaBoost_learning_rate']."
    config['Classification']['XGB_gamma'] = 'Gamma of XGB.'
    config['Classification']['XGB_min_child_weight'] = 'Minimum child weights in XGB.'
    config['Classification']['XGB_colsample_bytree'] = 'Col sample by tree in XGB.'

    # CrossValidation
    config['CrossValidation'] = dict()
    config['CrossValidation']['Type'] = 'If performing a cross-validationm, type of cross-validation used. Currently random-splitting and leave-one-out (LOO) are supported.'
    config['CrossValidation']['N_iterations'] = 'Number of times the data is split in training and test in the outer cross-validation when using random-splitting.'
    config['CrossValidation']['test_size'] = 'The percentage of data to be used for testing when using random-splitting.'
    config['CrossValidation']['fixed_seed'] = 'If True, use a fixed seed for the cross-validation splits when using random-splitting.'

    # Options for the object/patient labels that are used
    config['Labels'] = dict()
    config['Labels']['label_names'] = 'The labels used from your label file for classification.'
    config['Labels']['modus'] = 'Determine whether multilabel or singlelabel classification or regression will be performed.'

    # Hyperparameter optimization options
    config['HyperOptimization'] = dict()
    config['HyperOptimization']['scoring_method'] = 'Specify the optimization metric for your hyperparameter search.'
    config['HyperOptimization']['test_size'] = 'Size of test set in the hyperoptimization cross validation, given as a percentage of the whole dataset.'
    config['HyperOptimization']['N_iterations'] = 'Number of iterations used in the hyperparameter optimization. This corresponds to the number of samples drawn from the parameter grid.'
    config['HyperOptimization']['n_jobspercore'] = 'Number of jobs assigned to a single core. Only used if fastr is set to true in the classfication.'  # only relevant when using fastr in classification
    config['HyperOptimization']['n_splits'] = 'Number of iterations in train-validation cross-validation used for model optimization.'
    config['HyperOptimization']['maxlen'] = 'Number of estimators for which the fitted outcomes and parameters are saved. Increasing this number will increase the memory usage.'
    config['HyperOptimization']['ranking_score'] = 'Score used for ranking the performance of the evaluated workflows.'
    config['HyperOptimization']['memory'] = 'When using DRMAA plugin, e.g. on BIGR cluster, memory usage of a single optimization job. Should be a string consisting of an integer + "G".'
    config['HyperOptimization']['refit_workflows'] = 'If True, refit all workflows in the ensemble automatically during training. This will save time while performing inference, but will take more time during training and make the saved model much larger.'

    # Feature scaling options
    config['FeatureScaling'] = dict()
    config['FeatureScaling']['skip_features'] = 'Determine which features should be skipped. This field should contain a comma separated list of substrings: when one or more of these are in a feature name, the feature is skipped.'
    config['FeatureScaling']['scaling_method'] = 'Determine the scaling method.'

    # Ensemble options
    config['Ensemble'] = dict()
    config['Ensemble']['Use'] = 'Determine whether to use ensembling or not. Provide an integer to state how many estimators to include: 1 equals no ensembling.'
    config['Ensemble']['Metric'] = 'Metric used to determine ranking of estimators in ensemble. When using default, the metric that is used in the hyperoptimization is used.'

    # Evaluation options
    config['Evaluation'] = dict()
    config['Evaluation']['OverfitScaler'] = 'Wheter to fit a separate scaler on the test set (=overfitting) or use scaler on training dataset. Only used for experimental purposes: never overfit your scaler for the actual performance evaluation.'

    # Bootstrap options
    config['Bootstrap'] = dict()
    config['Bootstrap']['Use'] = 'Determine whether to use bootstrapping or not.'
    config['Bootstrap']['N_iterations'] = 'Number of iterations to use for bootstrapping.'

    return config


if __name__ == '__main__':
    generate_config_doc()
