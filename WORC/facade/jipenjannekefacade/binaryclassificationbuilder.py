from WORC.facade.jipenjannekefacade.abstractworcbuilder import AbstractWorcBuilder


class BinaryClassificationBuilder(AbstractWorcBuilder):
    def __init__(self, *args, **kwargs):
        super(BinaryClassificationBuilder, self).__init__(*args, **kwargs)
