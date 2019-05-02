#!/usr/bin/env python
import os

import WORC
from fastr.helpers.rest_generation import create_rest_table


def generate_config(fields):
    field_key = []
    field_subkey = []
    field_default = []
    field_description = []
    field_option = []

    a = WORC.WORC()

    config_defaults = a.defaultconfig()
    config_options = generate_config_options()
    config_desciptions = generate_config_descriptions()

    for key in config_defaults.keys():
        for subkey in config_defaults[key].keys():
            print(f'[generate_config.py] Documenting field {key}: {subkey}')
            field_key.append(key)
            field_subkey.append(subkey)
            field_default.append(config_defaults[key][subkey])

            try:
                field_description.append(config_desciptions[key][subkey])
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
        fh_out.write(generate_config(fastr.config.DEFAULT_FIELDS))

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
    config['ImageFeatures']['phase_minwavelength'] = 'Integer(s)'
    config['ImageFeatures']['phase_nscale'] = 'Integer(s) > 0'

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
    config['Classification']['fastr_plugin'] = 'Any [fastr execution plugin](https://fastr.readthedocs.io/en/develop/_autogen/fastr.reference.html#executionplugin-reference)'
    config['Classification']['classifiers'] = 'SVM , SVR, SGD, SGDR, RF, LDA, QDA, ComplementND, GaussianNB, LR, RFR, Lasso, ElasticNet. All are estimators from [sklearn](https://scikit-learn.org/stable/) '
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
    config['HyperOptimization']['scoring_method'] = 'Any [sklearn metric](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)'
    config['HyperOptimization']['test_size'] = 'Float'
    config['HyperOptimization']['N_iterations'] = 'Integer'
    config['HyperOptimization']['n_jobspercore'] = 'Integer'

    # Feature scaling options
    config['FeatureScaling'] = dict()
    config['FeatureScaling']['scale_features'] = 'Boolean(s)'
    config['FeatureScaling']['scaling_method'] = 'z_score'

    # Sample processing options
    config['SampleProcessing'] = dict()
    config['SampleProcessing']['SMOTE'] = 'Boolean(s)'
    config['SampleProcessing']['SMOTE_ratio'] = 'Two Integers: loc and scale'
    config['SampleProcessing']['SMOTE_neighbors'] = 'Two Integers: loc and scale'
    config['SampleProcessing']['Oversampling'] = 'Boolean(s)'

    # Ensemble options
    config['Ensemble'] = dict()
    config['Ensemble']['Use'] = 'Boolean or Integer'

    # BUG: the FASTR XNAT plugin can only retreive folders. We therefore need to add the filenames of the resources manually
    # This should be fixed from fastr > 2.0.0: need to update.
    config['FASTR_bugs'] = dict()
    config['FASTR_bugs']['images'] = 'String'
    config['FASTR_bugs']['segmentations'] = 'String'

    return config


def generate_config_descriptions():
    config = dict()

    # General configuration of WORC
    config['General'] = dict()
    config['General']['cross_validation'] = 'True, False'
    config['General']['Segmentix'] = 'Use Segmentix tool for segmentation preprocessing'
    config['General']['PCE'] = 'True, False'
    config['General']['FeatureCalculator'] = 'Specifies which feature calculation tool should be used'
    config['General']['Preprocessing'] = 'Specifies which tool will be used for image preprocessing'
    config['General']['RegistrationNode'] = "Specifies which tool will be used for image registration"
    config['General']['TransformationNode'] = "Specifies which tool will be used for applying image transformations"
    config['General']['Joblib_ncores'] = 'Number of cores to be used by joblib for multicore processing.'
    config['General']['Joblib_backend'] = 'Type of backend to be used by joblib for multicore processing.'
    config['General']['tempsave'] = 'Determines whether after every cross validation iteration the result will be saved, in addition to the result after all iterations. Especially useful for debugging.'

    # Segmentix
    config['Segmentix'] = dict()
    config['Segmentix']['mask'] = 'If a mask is supplied, should the mask be subtracted from the contour or multiplied'
    config['Segmentix']['segtype'] = 'If Ring, then a ring around the segmentation will be used as contour.'
    config['Segmentix']['segradius'] = 'Define the radius of the ring used if segtype is Ring'
    config['Segmentix']['N_blobs'] = 'How many of the largest blobs are extracted from the segmentation. If None, no blob extraction is used'
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
    config['ImageFeatures']['texture_Gabor'] = 'Determine whether Gabor texture features are computed or not.'
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
    config['ImageFeatures']['GLCM_angles'] = '0, 0.79, 1.57, 2.36'

    # GLCM discretization levels, distances in pixels
    config['ImageFeatures']['GLCM_levels'] = '16'
    config['ImageFeatures']['GLCM_distances'] = '1, 3'

    # LBP radius, number of points in pixels
    config['ImageFeatures']['LBP_radius'] = '3, 8, 15'
    config['ImageFeatures']['LBP_npoints'] = '12, 24, 36'

    # Phase features minimal wavelength and number of scales
    config['ImageFeatures']['phase_minwavelength'] = '3'
    config['ImageFeatures']['phase_nscale'] = '5'

    # Log features sigma of Gaussian in pixels
    config['ImageFeatures']['log_sigma'] = '1, 5, 10'

    # Vessel features scale range, steps for the range
    config['ImageFeatures']['vessel_scale_range'] = '1, 10'
    config['ImageFeatures']['vessel_scale_step'] = '2'

    # Vessel features radius for erosion to determine boudnary
    config['ImageFeatures']['vessel_radius'] = '5'

    # Feature selection
    config['Featsel'] = dict()
    config['Featsel']['Variance'] = 'Boolean(s)'
    config['Featsel']['GroupwiseSearch'] = 'True, False
    config['Featsel']['SelectFromModel'] = 'True, False
    config['Featsel']['UsePCA'] = 'True, False
    config['Featsel']['PCAType'] = '95variance'
    config['Featsel']['StatisticalTestUse'] = 'True, False
    config['Featsel']['StatisticalTestMetric'] = 'ttest, Welch, Wilcoxon, MannWhitneyU'
    config['Featsel']['StatisticalTestThreshold'] = '0.02, 0.2'
    config['Featsel']['ReliefUse'] = 'True, False
    config['Featsel']['ReliefNN'] = '2, 4'
    config['Featsel']['ReliefSampleSize'] = '1, 1'
    config['Featsel']['ReliefDistanceP'] = '1, 3'
    config['Featsel']['ReliefNumFeatures'] = '25, 200'

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
    config['SelectFeatGroup']['texture_LBP_features'] = 'True, False'
    config['SelectFeatGroup']['patient_features'] = 'True, False
    config['SelectFeatGroup']['semantic_features'] = 'True, False
    config['SelectFeatGroup']['coliage_features'] = 'True, False
    config['SelectFeatGroup']['log_features'] = 'True, False
    config['SelectFeatGroup']['vessel_features'] = 'True, False
    config['SelectFeatGroup']['phase_features'] = 'True, False

    # Feature imputation
    config['Imputation'] = dict()
    config['Imputation']['use'] = 'True, False
    config['Imputation']['strategy'] = 'mean, median, most_frequent, constant, knn'
    config['Imputation']['n_neighbors'] = '5, 5'

    # Classification
    config['Classification'] = dict()
    config['Classification']['fastr'] = 'True, False
    config['Classification']['fastr_plugin'] = self.fastr_plugin
    config['Classification']['classifiers'] = 'SVM'
    config['Classification']['max_iter'] = '100000'
    config['Classification']['SVMKernel'] = 'poly'
    config['Classification']['SVMC'] = '0, 6'
    config['Classification']['SVMdegree'] = '1, 6'
    config['Classification']['SVMcoef0'] = '0, 1'
    config['Classification']['SVMgamma'] = '-5, 5'
    config['Classification']['RFn_estimators'] = '10, 90'
    config['Classification']['RFmin_samples_split'] = '2, 3'
    config['Classification']['RFmax_depth'] = '5, 5'
    config['Classification']['LRpenalty'] = 'l2, l1'
    config['Classification']['LRC'] = '0.01, 1.0'
    config['Classification']['LDA_solver'] = 'svd, lsqr, eigen'
    config['Classification']['LDA_shrinkage'] = '-5, 5'
    config['Classification']['QDA_reg_param'] = '-5, 5'
    config['Classification']['ElasticNet_alpha'] = '-5, 5'
    config['Classification']['ElasticNet_l1_ratio'] = '0, 1'
    config['Classification']['SGD_alpha'] = '-5, 5'
    config['Classification']['SGD_l1_ratio'] = '0, 1'
    config['Classification']['SGD_loss'] = 'hinge, squared_hinge, modified_huber'
    config['Classification']['SGD_penalty'] = 'none, l2, l1'
    config['Classification']['CNB_alpha'] = '0, 1'

    # CrossValidation
    config['CrossValidation'] = dict()
    config['CrossValidation']['N_iterations'] = '100'
    config['CrossValidation']['test_size'] = '0.2'

    # Options for the object/patient labels that are used
    config['Labels'] = dict()
    config['Labels']['label_names'] = 'Label1, Label2'
    config['Labels']['modus'] = 'singlelabel'
    config['Labels']['url'] = 'WIP'
    config['Labels']['projectID'] = 'WIP'

    # Hyperparameter optimization options
    config['HyperOptimization'] = dict()
    config['HyperOptimization']['scoring_method'] = 'f1_weighted'
    config['HyperOptimization']['test_size'] = '0.15'
    config['HyperOptimization']['N_iterations'] = '10000'
    config['HyperOptimization']['n_jobspercore'] = '2000'  # only relevant when using fastr in classification

    # Feature scaling options
    config['FeatureScaling'] = dict()
    config['FeatureScaling']['scale_features'] = 'True, False
    config['FeatureScaling']['scaling_method'] = 'z_score'

    # Sample processing options
    config['SampleProcessing'] = dict()
    config['SampleProcessing']['SMOTE'] = 'Boolean(s)'
    config['SampleProcessing']['SMOTE_ratio'] = '1, 0'
    config['SampleProcessing']['SMOTE_neighbors'] = '5, 15'
    config['SampleProcessing']['Oversampling'] = 'True, False

    # Ensemble options
    config['Ensemble'] = dict()
    config['Ensemble']['Use'] = 'True, False  # Still WIP

    # BUG: the FASTR XNAT plugin can only retreive folders. We therefore need to add the filenames of the resources manually
    # This should be fixed from fastr > 2.0.0: need to update.
    config['FASTR_bugs'] = dict()
    config['FASTR_bugs']['images'] = 'image.nii.gz'
    config['FASTR_bugs']['segmentations'] = 'mask.nii.gz'

    return config


if __name__ == '__main__':
    generate_config_doc()
