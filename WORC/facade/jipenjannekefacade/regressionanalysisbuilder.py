from WORC.facade.jipenjannekefacade.abstractworcbuilder import AbstractWorcBuilder


class RegressionAnalysisBuilder(AbstractWorcBuilder):
    def __init__(self, classifier='SVR', scoring_method='r2', *args, **kwargs):
        super(RegressionAnalysisBuilder, self).__init__(*args, **kwargs)

        self._config['Classification'] = {'classifiers': classifier}
        self._config['HyperOptimization'] = {'scoring_method': scoring_method}
