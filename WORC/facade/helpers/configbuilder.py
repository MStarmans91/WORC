from WORC.detectors.detectors import BigrClusterDetector, CartesiusClusterDetector, DebugDetector
import configparser
import fastr


class ConfigBuilder():
    def __init__(self):
        # initalize the main config object and the custom overrids
        self._config = configparser.ConfigParser()
        self._custom_overrides = {}

        # Detect when using a cluster and override relevant config fields
        self._cluster_config_overrides()

    def build_config(self, defaultconfig):
        defaultconfig.read_dict({**self._config})
        defaultconfig.read_dict({**self._custom_overrides})
        defaultconfig.read_dict({**self._debug_config_overrides()})

        self._config = defaultconfig
        return defaultconfig

    def custom_config_overrides(self, config):
        self._custom_overrides.update(config)

    def _cluster_config_overrides(self):
        if BigrClusterDetector().do_detection():
            overrides = {
                'General': {'Joblib_ncores': '1',
                            'Joblib_backend': 'threading'},
                'Classification': {'fastr': 'True',
                                   'fastr_plugin': 'DRMAAExecution'},
                'HyperOptimization': {'n_jobspercore': '4000'}
            }
        elif CartesiusClusterDetector().do_detection():
            overrides = {
                'Classification': {'fastr': 'True',
                                   'fastr_plugin': 'ProcessPoolExecution'},
                'HyperOptimization': {'n_jobspercore': '4000'}
            }
        else:
            overrides = {}  # not a cluster or unsupported

        self._custom_overrides.update(overrides)
        return overrides

    def estimator_scoring_overrides(self, estimators, scoring_method):
        overrides = {
            'Classification': {'classifiers': ', '.join(estimators)},
            'HyperOptimization': {'scoring_method': scoring_method}
        }
        self._custom_overrides.update(overrides)
        return overrides

    def coarse_overrides(self):
        overrides = {
            'ImageFeatures': {
                'texture_Gabor': 'False',
                'vessel': 'False',
                'log': 'False',
                'phase': 'False',
            },
            'SelectFeatGroup': {
                'texture_Gabor_features': 'False',
                'log_features': 'False',
                'vessel_features': 'False',
                'phase_features': 'False',
            },
            'CrossValidation': {'N_iterations': '3'},
            'Classification': {'classifiers': 'SVM'},
            'HyperOptimization': {'n_splits': '2',
                                  'N_iterations': '1000',
                                  'n_jobspercore': '500'},
            'Ensemble': {'Use': '1'},
            'SampleProcessing': {'SMOTE': 'False'},
        }
        self._custom_overrides.update(overrides)
        return overrides

    def full_overrides(self):
        overrides = {
            'ImageFeatures': {
                'texture_Gabor': 'True',
                'vessel': 'True',
                'log': 'True',
                'phase': 'True',
            },
            'SelectFeatGroup': {
                'texture_Gabor_features': 'True, False',
                'log_features': 'True, False',
                'vessel_features': 'True, False',
                'phase_features': 'True, False',
            },
            'Classification': {'classifiers': 'SVM, SVM, SVM, RF, LR, LDA, QDA, GaussianNB'},
            'CrossValidation': {'N_iterations': '100'},
            'HyperOptimization': {'N_iterations': '100000',
                                  'n_jobspercore': '4000'},
            'Ensemble': {'Use': '50'},
            'SampleProcessing': {'SMOTE': 'True'},
        }
        self._custom_overrides.update(overrides)
        return overrides

    def fullprint(self):
        '''
        Print the full contents of the config to the console.
        '''
        for k, v in self._config.items():
            print(f"{k}:")
            for k2, v2 in v.items():
                print(f"\t {k2}: {v2}")
            print("\n")

    def _debug_config_overrides(self):
        if DebugDetector().do_detection():
            overrides = {
                'ImageFeatures': {
                    'texture_Gabor': 'False',
                    'vessel': 'False',
                    'log': 'False',
                    'phase': 'False',
                    'texture_LBP': 'False',
                    'texture_GLCMMS': 'False',
                    'texture_GLRLM': 'False',
                    'texture_NGTDM': 'False',
                },
                'SelectFeatGroup': {
                    'texture_Gabor_features': 'False',
                    'log_features': 'False',
                    'vessel_features': 'False',
                    'phase_features': 'False',
                },
                'CrossValidation': {'N_iterations': '2',
                                    'fixed_seed': ' True'},
                'HyperOptimization': {'N_iterations': '10',
                                      'n_jobspercore': '10',
                                      'n_splits': '2'},
                'Ensemble': {'Use': '1'},
                'SampleProcessing': {'SMOTE': 'False'},
            }

            # Additionally, turn queue reporting system on
            fastr.config.queue_report_interval = 120
        else:
            overrides = {} # not a cluster or unsupported

        return overrides
