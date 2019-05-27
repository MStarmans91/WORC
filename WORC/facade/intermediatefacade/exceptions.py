class PathNotFoundException(Exception):
    def __init__(self, path):
        super(PathNotFoundException, self).__init__(f'Path not found: {path}')

class NoImagesFoundException(Exception):
    def __init__(self, path):
        super(PathNotFoundException, self).__init__(f'No images found in directory {path}')

class NoSegmentationsFoundException(Exception):
    def __init__(self, path):
        super(PathNotFoundException, self).__init__(f'No segmentations found in directory {path}')