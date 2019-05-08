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
            print(f'[generate_config.py] Documenting field {key}: {subkey}')
            if num == 0:
                field_key.append(key)
            else:
                # After the first field of this key is generated, we do not append the key for better readability
                field_key.append('')

            field_subkey.append(subkey)
            field_default.append(config_defaults[key][subkey])

            try:
                field_description.append(config_descriptions[key][subkey])
            except KeyError:
                print(f'[WARNING] No description for {key}: {subkey}')
                field_description.append('')

            try:
                field_option.append(config_options[key][subkey])
            except KeyError:
                print(f'[WARNING] No options for {key}: {subkey}')
                field_option.append(config_defaults[key][subkey])

    data = [field_key, field_subkey, field_description, field_default, field_option]
    headers = ['Key', 'Subkey', 'Description', 'Default', 'Options',]

    return create_rest_table(data, headers)


def generate_config_doc():
    print('[generate_config.py] Generating config reference...')
    filename = os.path.join(os.path.dirname(__file__), 'autogen', 'WORC.config.rst')

    with open(filename, 'w') as fh_out:
        fh_out.write(generate_config())

    print(f'[generate_config.py] Config reference saved to {filename}')


def generate_config_options():
    config = dict()

    # General configuration of WORC
    config['General'] = dict()
    config['General']['cross_validation'] = 'True, False'
    config['General']['Segmentix'] = 'True, False'
    config['General']['FeatureCalculator'] = 'predict/CalcFeatures:1.0, pyradiomics/CF_pyradiomics:1.0, your own tool reference'
    config['General']['Preprocessing'] = 'worc/PreProcess:1.0, your own tool reference'
    config['General']['RegistrationNode'] = "'elastix4.8/Elastix:4.8', your own tool reference"
    config['General']['TransformationNode'] = "'elastix4.8/Transformix:4.8', your own tool reference"
    config['General']['Joblib_ncores'] = 'Integer > 0'
    config['General']['Joblib_backend'] = 'multiprocessing, threading'
    config['General']['tempsave'] = 'True, False'

    # Segmentix
    config['Segmentix'] = dict()
    config['Segmentix']['mask'] = 'subtract, multiply'
    config['Segmentix']['segtype'] = 'None, Ring'
    config['Segmentix']['segradius'] = 'Integer > 0'
    config['Segmentix']['N_blobs'] = 'Integer > 0'
    config['Segmentix']['fillholes'] = 'True, False'

    # Preprocessing
    config['Normalize'] = dict()
    config['Normalize']['ROI'] = 'True, False, Full'
    config['Normalize']['Method'] = 'z_score, minmed'

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

    # Feature selection
    config['Featsel'] = dict()
    config['Featsel']['Variance'] = 'Boolean(s)'
    config['Featsel']['GroupwiseSearch'] = 'Boolean(s)'
    config['Featsel']['SelectFromModel'] = 'Boolean(s)'
    config['Featsel']['UsePCA'] = 'Boolean(s)'
    config['Featsel']['PCAType'] = 'Inteteger(s), 95variance'
    config['Featsel']['StatisticalTestUse'] = 'Boolean(s)'
    config['Featsel']['StatisticalTestMetric'] = 'ttest, Welch, Wilcoxon, MannWhitneyU'
    config['Featsel']['StatisticalTestThreshold'] = 'Two Integers: loc and scale'
    config['Featsel']['ReliefUse'] = 'Boolean(s)'
    config['Featsel']['ReliefNN'] = 'Two Integers: loc and scale'
    config['Featsel']['ReliefSampleSize'] = 'Two Integers: loc and scale'
    config['Featsel']['ReliefDistanceP'] = 'Two Integers: loc and scale'
    config['Featsel']['ReliefNumFeatures'] = 'Two Integers: loc and scale'

    # Groupwie Featureselection options
    config['SelectFeatGroup'] = dict()
    config['SelectFeatGroup']['shape_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['histogram_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['orientation_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_Gabor_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLCM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLCMMS_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLRLM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_GLSZM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_NGTDM_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['texture_LBP_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['patient_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['semantic_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['coliage_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['log_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['vessel_features'] = 'Boolean(s)'
    config['SelectFeatGroup']['phase_features'] = 'Boolean(s)'

    # Feature imputation
    config['Imputation'] = dict()
    config['Imputation']['use'] = 'Boolean(s)'
    config['Imputation']['strategy'] = 'mean, median, most_frequent, constant, knn'
    config['Imputation']['n_neighbors'] = 'Two Integers: loc and scale'

    # Classification
    config['Classification'] = dict()
    config['Classification']['fastr'] = 'True, False'
    config['Classification']['fastr_plugin'] = 'Any `fastr execution plugin <https://fastr.readthedocs.io/en/develop/_autogen/fastr.reference.html#executionplugin-reference/>`_ .'
    config['Classification']['classifiers'] = 'SVM , SVR, SGD, SGDR, RF, LDA, QDA, ComplementND, GaussianNB, LR, RFR, Lasso, ElasticNet. All are estimators from `sklearn <https://scikit-learn.org/stable//>`_ '
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
    config['Classification']['LRC'] = 'Two Integers: loc and scale'
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

    # CrossValidation
    config['CrossValidation'] = dict()
    config['CrossValidation']['N_iterations'] = 'Integer'
    config['CrossValidation']['test_size'] = 'Float'

    # Options for the object/patient labels that are used
    config['Labels'] = dict()
    config['Labels']['label_names'] = 'String(s)'
    config['Labels']['modus'] = 'singlelabel, multilabel'
    config['Labels']['url'] = 'Not Supported Yet'
    config['Labels']['projectID'] = 'Not Supported Yet'

    # Hyperparameter optimization options
    config['HyperOptimization'] = dict()
    config['HyperOptimization']['scoring_method'] = 'Any `sklearn metric <https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values/>`_'
    config['HyperOptimization']['test_size'] = 'Float'
    config['HyperOptimization']['N_iterations'] = 'Integer'
    config['HyperOptimization']['n_jobspercore'] = 'Integer'

    # Feature scaling options
    config['FeatureScaling'] = dict()
    config['FeatureScaling']['scale_features'] = 'Boolean(s)'
    config['FeatureScaling']['scaling_method'] = 'z_score, minmax'

    # Sample processing options
    config['SampleProcessing'] = dict()
    config['SampleProcessing']['SMOTE'] = 'Boolean(s)'
    config['SampleProcessing']['SMOTE_ratio'] = 'Two Integers: loc and scale'
    config['SampleProcessing']['SMOTE_neighbors'] = 'Two Integers: loc and scale'
    config['SampleProcessing']['Oversampling'] = 'Boolean(s)'

    # Ensemble options
    config['Ensemble'] = dict()
    config['Ensemble']['Use'] = 'Boolean or Integer'

    return config


def generate_config_descriptions():
    config = dict()

    # General configuration of WORC
    config['General'] = dict()
    config['General']['cross_validation'] = 'Determine whether a cross validation will be performed or not. Obsolete, will be removed.'
    config['General']['Segmentix'] = 'Determine whether to use Segmentix tool for segmentation preprocessing.'
    config['General']['FeatureCalculator'] = 'Specifies which feature calculation tool should be used.'
    config['General']['Preprocessing'] = 'Specifies which tool will be used for image preprocessing.'
    config['General']['RegistrationNode'] = "Specifies which tool will be used for image registration."
    config['General']['TransformationNode'] = "Specifies which tool will be used for applying image transformations."
    config['General']['Joblib_ncores'] = 'Number of cores to be used by joblib for multicore processing.'
    config['General']['Joblib_backend'] = 'Type of backend to be used by joblib for multicore processing.'
    config['General']['tempsave'] = 'Determines whether after every cross validation iteration the result will be saved, in addition to the result after all iterations. Especially useful for debugging.'

    # Segmentix
    config['Segmentix'] = dict()
    config['Segmentix']['mask'] = 'If a mask is supplied, should the mask be subtracted from the contour or multiplied.'
    config['Segmentix']['segtype'] = 'If Ring, then a ring around the segmentation will be used as contour.'
    config['Segmentix']['segradius'] = 'Define the radius of the ring used if segtype is Ring.'
    config['Segmentix']['N_blobs'] = 'How many of the largest blobs are extracted from the segmentation. If None, no blob extraction is used.'
    config['Segmentix']['fillholes'] = 'Determines whether hole filling will be used.'

    # Preprocessing
    config['Normalize'] = dict()
    config['Normalize']['ROI'] = 'If a mask is supplied and this is set to True, normalize image based on supplied ROI. Otherwise, the full image is used for normalization using the SimpleITK Normalize function. Lastly, setting this to False will result in no normalization being applied.'
    config['Normalize']['Method'] = 'Method used for normalization if ROI is supplied. Currently, z-scoring or using the minimum and median of the ROI can be used.'

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

    # Feature selection
    config['Featsel'] = dict()
    config['Featsel']['Variance'] = 'If True, exclude features which have a variance < 0.01. Based on ` sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html/>`_.'
    config['Featsel']['GroupwiseSearch'] = 'Randomly select which feature groups to use. Parameters determined by the SelectFeatGroup config part, see below.'
    config['Featsel']['SelectFromModel'] = 'Select features by first training a LASSO model. The alpha for the LASSO model is randomly generated. See also `sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html/>`_.'
    config['Featsel']['UsePCA'] = 'If True, Use Principle Component Analysis (PCA) to select features.'
    config['Featsel']['PCAType'] = 'Method to select number of components using PCA: Either the number of components that explains 95% of the variance, or use a fixed number of components.95variance'
    config['Featsel']['StatisticalTestUse'] = 'If True, use statistical test to select features.'
    config['Featsel']['StatisticalTestMetric'] = 'Define the type of statistical test to be used.'
    config['Featsel']['StatisticalTestThreshold'] = 'Specify a threshold for the p-value threshold used in the statistical test to select features. The first element defines the lower boundary, the other the upper boundary. Random sampling will occur between the boundaries.'
    config['Featsel']['ReliefUse'] = 'If True, use Relief to select features.'
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
    config['SelectFeatGroup']['texture_GLCMMS_features'] = 'If True, use GLCM Multislice texture features in model.'
    config['SelectFeatGroup']['texture_GLRLM_features'] = 'If True, use GLRLM texture features in model.'
    config['SelectFeatGroup']['texture_GLSZM_features'] = 'If True, use GLSZM texture features in model.'
    config['SelectFeatGroup']['texture_NGTDM_features'] = 'If True, use NGTDM texture features in model.'
    config['SelectFeatGroup']['texture_LBP_features'] = 'If True, use LBP texture features in model.'
    config['SelectFeatGroup']['patient_features'] = 'If True, use patient features in model.'
    config['SelectFeatGroup']['semantic_features'] = 'If True, use semantic features in model.'
    config['SelectFeatGroup']['coliage_features'] = 'If True, use coliage features in model.'
    config['SelectFeatGroup']['log_features'] = 'If True, use log features in model.'
    config['SelectFeatGroup']['vessel_features'] = 'If True, use vessel features in model.'
    config['SelectFeatGroup']['phase_features'] = 'If True, use phase features in model.'

    # Feature imputation
    config['Imputation'] = dict()
    config['Imputation']['use'] = 'If True, use feature imputation methods to replace NaN values. If False, all NaN features will be set to zero.'
    config['Imputation']['strategy'] = 'Method to be used for imputation.'
    config['Imputation']['n_neighbors'] = 'When using k-Nearest Neighbors (kNN) for feature imputation, determines the number of neighbors used for imputation. Can be a single integer or a list.'

    # Classification
    config['Classification'] = dict()
    config['Classification']['fastr'] = 'Use fastr for the optimization gridsearch (recommended on clusters, default) or if set to False , joblib (recommended for PCs but not on Windows).'
    config['Classification']['fastr_plugin'] = 'Name of execution plugin to be used. Default use the same as the self.fastr_plugin for the WORC object.'
    config['Classification']['classifiers'] = "Select the estimator(s) to use. Most are implemented using `sklearn <https://scikit-learn.org/stable/>`_. For abbreviations, see above."
    config['Classification']['max_iter'] = 'Maximum number of iterations to use in training an estimator. Only for specific estimators, see `sklearn <https://scikit-learn.org/stable/>`_.'
    config['Classification']['SVMKernel'] = 'When using a SVM, specify the kernel type.'
    config['Classification']['SVMC'] = 'Range of the SVM slack parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b).'
    config['Classification']['SVMdegree'] = 'Range of the SVM polynomial degree when using a polynomial kernel. We sample on a uniform scale: the parameters specify the range (a, a + b). '
    config['Classification']['SVMcoef0'] = 'Range of SVM homogeneity parameter. We sample on a uniform scale: the parameters specify the range (a, a + b). '
    config['Classification']['SVMgamma'] = 'Range of the SVM gamma parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b)'
    config['Classification']['RFn_estimators'] = 'Range of number of trees in a RF. We sample on a uniform scale: the parameters specify the range (a, a + b).'
    config['Classification']['RFmin_samples_split'] = 'Range of minimum number of samples required to split a branch in a RF. We sample on a uniform scale: the parameters specify the range (a, a + b). '
    config['Classification']['RFmax_depth'] = 'Range of maximum depth of a RF. We sample on a uniform scale: the parameters specify the range (a, a + b). '
    config['Classification']['LRpenalty'] = 'Penalty term used in LR.'
    config['Classification']['LRC'] = 'Range of regularization strength in LR. We sample on a uniform scale: the parameters specify the range (a, a + b). '
    config['Classification']['LDA_solver'] = 'Solver used in LDA.'
    config['Classification']['LDA_shrinkage'] = 'Range of the LDA shrinkage parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b).'
    config['Classification']['QDA_reg_param'] = 'Range of the QDA regularization parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b). '
    config['Classification']['ElasticNet_alpha'] = 'Range of the ElasticNet penalty parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b).'
    config['Classification']['ElasticNet_l1_ratio'] = 'Range of l1 ratio in LR. We sample on a uniform scale: the parameters specify the range (a, a + b).'
    config['Classification']['SGD_alpha'] = 'Range of the SGD penalty parameter. We sample on a uniform log scale: the parameters specify the range of the exponent (a, a + b).'
    config['Classification']['SGD_l1_ratio'] = 'Range of l1 ratio in SGD. We sample on a uniform scale: the parameters specify the range (a, a + b).'
    config['Classification']['SGD_loss'] = 'hinge, Loss function of SG'
    config['Classification']['SGD_penalty'] = 'Penalty term in SGD.'
    config['Classification']['CNB_alpha'] = 'Regularization strenght in ComplementNB. We sample on a uniform scale: the parameters specify the range (a, a + b)'

    # CrossValidation
    config['CrossValidation'] = dict()
    config['CrossValidation']['N_iterations'] = 'Number of times the data is split in training and test in the outer cross-validation.'
    config['CrossValidation']['test_size'] = 'The percentage of data to be used for testing.'

    # Options for the object/patient labels that are used
    config['Labels'] = dict()
    config['Labels']['label_names'] = 'The labels used from your label file for classification.'
    config['Labels']['modus'] = 'Determine whether multilabel or singlelabel classification or regression will be performed.'
    config['Labels']['url'] = 'WIP'
    config['Labels']['projectID'] = 'WIP'

    # Hyperparameter optimization options
    config['HyperOptimization'] = dict()
    config['HyperOptimization']['scoring_method'] = 'Specify the optimization metric for your hyperparameter search.'
    config['HyperOptimization']['test_size'] = 'Size of test set in the hyperoptimization cross validation, given as a percentage of the whole dataset.'
    config['HyperOptimization']['N_iterations'] = 'Number of iterations used in the hyperparameter optimization. This corresponds to the number of samples drawn from the parameter grid.'
    config['HyperOptimization']['n_jobspercore'] = 'Number of jobs assigned to a single core. Only used if fastr is set to true in the classfication.'  # only relevant when using fastr in classification

    # Feature scaling options
    config['FeatureScaling'] = dict()
    config['FeatureScaling']['scale_features'] = 'Determine whether to use feature scaling is.'
    config['FeatureScaling']['scaling_method'] = 'Determine the scaling method.'

    # Sample processing options
    config['SampleProcessing'] = dict()
    config['SampleProcessing']['SMOTE'] = 'Determine whether to use SMOTE oversampling, see also ` imbalanced learn <https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html/>`_. '
    config['SampleProcessing']['SMOTE_ratio'] = 'Determine the ratio of oversampling. If 1, the minority class will be oversampled to the same size as the majority class. We sample on a uniform scale: the parameters specify the range (a, a + b). '
    config['SampleProcessing']['SMOTE_neighbors'] = 'Number of neighbors used in SMOTE. This should be much smaller than the number of objects/patients you supply. We sample on a uniform scale: the parameters specify the range (a, a + b).'
    config['SampleProcessing']['Oversampling'] = 'Determine whether to random oversampling.'

    # Ensemble options
    config['Ensemble'] = dict()
    config['Ensemble']['Use'] = 'Determine whether to use ensembling or not. Either provide an integer to state how many estimators to include, or True, which will use the default ensembling method.'


    return config


if __name__ == '__main__':
    generate_config_doc()
