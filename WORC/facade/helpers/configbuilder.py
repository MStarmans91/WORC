from WORC.detectors.detectors import BigrClusterDetector, CartesiusClusterDetector, DebugDetector
import configparser
import fastr
import collections.abc


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


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
        _deep_update(self._custom_overrides, config)

    def _cluster_config_overrides(self):
        if BigrClusterDetector().do_detection():
            overrides = {
                'General': {'Joblib_ncores': '1',
                            'Joblib_backend': 'threading'},
                'Classification': {'fastr': 'True',
                                   'fastr_plugin': 'DRMAAExecution'},
            }
        elif CartesiusClusterDetector().do_detection():
            overrides = {
                'Classification': {'fastr': 'True',
                                   'fastr_plugin': 'ProcessPoolExecution'},
                'HyperOptimization': {'n_jobspercore': '4000'}
            }
        else:
            overrides = {}  # not a cluster or unsupported

        self.custom_config_overrides(overrides)
        return overrides

    def estimator_scoring_overrides(self, estimators, scoring_method):
        overrides = {
            'Classification': {'classifiers': ', '.join(estimators)},
            'HyperOptimization': {'scoring_method': scoring_method}
        }
        self.custom_config_overrides(overrides)
        return overrides

    def coarse_overrides(self):
        overrides = {
            'General': {
                # Use only one feature extraction toolbox
                'FeatureCalculators': '[predict/CalcFeatures:1.0]'
            },
            'ImageFeatures': {
                # Extract only a subset of features
                'texture_Gabor': 'False',
                'vessel': 'False',
                'log': 'False',
                'phase': 'False',
            },
            'SelectFeatGroup': {
                # No search has to be conducted for excluded features
                'texture_Gabor_features': 'False',
                'log_features': 'False',
                'vessel_features': 'False',
                'phase_features': 'False',
                'toolbox': 'PREDICT'
            },
            # Do not use any resampling
            'SampleProcessing': {
                'SMOTE': 'False',
                },
            'CrossValidation': {
                # Only perform a 3x random-split cross-validation
                'N_iterations': '3',
                # Set a fixed seed, so we get the same result every time
                'fixed_seed': ' True'
                },
            # Hyperoptimization is very minimal
            'HyperOptimization': {'n_splits': '2',
                                  'N_iterations': '1000',
                                  'n_jobspercore': '500'},
            # No ensembling
            'Ensemble': {'Use': '1'}
        }
        self.custom_config_overrides(overrides)
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
                'PyRadiomics': {
                    'Wavelet': 'False',
                    'LoG': 'False'
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
