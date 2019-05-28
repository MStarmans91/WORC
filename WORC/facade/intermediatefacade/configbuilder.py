from WORC.detectors.detectors import BigrClusterDetector, CartesiusClusterDetector



class ConfigBuilder():
    def __init__(self):
        self._config = {**self._cluster_config_overrides()}
        self._custom_overrides = {}

    def build_config(self, defaultconfig):
        defaultconfig.read_dict({**self._config, **self._custom_overrides})
        return defaultconfig

    def custom_config_overrides(self, config):
        self._custom_overrides = config

    def _cluster_config_overrides(self):
        if BigrClusterDetector().do_detection():
            return {
                'General': {'Joblib_ncores': '1'},
                'General': {'Joblib_backend': 'threading'},
                'Classification': {'fastr': 'True'},
                'Classification': {'fastr_plugin': 'DRMAAExecution'},
                'HyperOptimization': {'n_jobspercore': '4000'}
            }
        elif CartesiusClusterDetector().do_detection():
            return {
                'Classification': {'fastr': 'True'},
                'Classification': {'fastr_plugin': 'ProcessPoolExecution'},
                'HyperOptimization': {'n_jobspercore': '4000'}
            }

        return {}  # not a cluster or unsupported

    def estimator_scoring_overrides(self, estimators, scoring_method):
        return {
            'Classification': {'classifiers': ', '.join(estimators)},
            'HyperOptimization': {'scoring_method': scoring_method}
        }

    def coarse_overrides(self):
        return {
            'ImageFeatures': {
                'texture_Gabor': 'False',
                'vessel': 'False',
                'log': 'False',
                'phase': 'False'
            },
            'SelectFeatGroup': {
                'texture_Gabor_features': 'False',
                'log_features': 'False',
                'vessel_features': 'False',
                'phase_features': 'False'
            },
            'CrossValidation': {'N_iterations': '5'},
            'HyperOptimization': {'N_iterations': '10000'},
            'Ensemble': {'Use': 'False'}
        }

    def full_overrides(self):
        return {
            'ImageFeatures': {
                'texture_Gabor': True,
                'vessel': True,
                'log': True,
                'phase': True
            },
            'SelectFeatGroup': {
                'texture_Gabor_features': 'True, False',
                'log_features': 'True, False',
                'vessel_features': 'True, False',
                'phase_features': 'True, False'
            },
            'CrossValidation': {'N_iterations': '100'},
            'HyperOptimization': {'N_iterations': '100000'},
            'Ensemble': {'Use': '50'}
        }
