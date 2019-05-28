from WORC.facade.jipenjannekefacade.binaryclassificationbuilder import BinaryClassificationBuilder
from WORC.facade.jipenjannekefacade.regressionanalysisbuilder import RegressionAnalysisBuilder


class JipEnJannekeFacade():
    def __init__(self):
        pass

    def want_to_do_a_binary_classification(self):
        return BinaryClassificationBuilder()

    def want_to_do_a_regression_analysis(self):
        return RegressionAnalysisBuilder()