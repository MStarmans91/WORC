class InvalidOrderException(Exception):
    def __init__(self, function, execute_first):
        super(InvalidOrderException, self).__init__(f'Invalid order for function {path} call {execute_first} before calling this function')


class InvalidCsvFileException(Exception):
    def __init__(self, path):
        super(InvalidCsvFileException, self).__init__(f'Invalid or unreadable csv file: {path}')


class PathNotFoundException(Exception):
    def __init__(self, path):
        super(PathNotFoundException, self).__init__(f'Path not found: {path}')


class NoFeaturesFoundException(Exception):
    def __init__(self, path):
        super(NoFeaturesFoundException, self).__init__(f'No features found in directory {path}')


class NoImagesFoundException(Exception):
    def __init__(self, path):
        super(NoImagesFoundException, self).__init__(f'No images found in directory {path}')


class NoSegmentationsFoundException(Exception):
    def __init__(self, path):
        super(NoSegmentationsFoundException, self).__init__(f'No segmentations found in directory {path}')
