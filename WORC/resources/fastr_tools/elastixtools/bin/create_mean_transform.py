#!/usr/bin/env python
import argparse
import copy
from elastixparameterfile import ElastixParameterFile


def create_mean_transform_bin(input_transformations, output_transformation):
    """
    This function reads a number of elastix transform files, creates the mean transformation and writes it back to the specified output path.

    :param input_transformations: the path of the input transformation files
    :type input_transformations: iterable of str
    :param output_transformation: path of the output file to be created
    """
    if not all(isinstance(x, str) for x in input_transformations):
        raise TypeError('Input transformations need to be a str containing the path of the transformation file!')

    if not isinstance(output_transformation, str):
        raise TypeError('Output transformation needs to be a str where to write the new transformation to')

    input_transformations = [x for x in input_transformations]
    reference_transformation = ElastixParameterFile(input_transformations[0])

    # Outsource to dedicated function
    output_transform = create_mean_transform(reference_transformation, input_transformations)

    # Save transform to file
    output_transform.write(output_transformation)


def create_mean_transform(reference_transform, input_transformations):
    """
    Given a list of transformation paths, and a reference object, create a new object that is the mean transform

    :param ElastixParameterFile reference_transform: The reference transformation object (for defining the target space etc)
    :param input_transformations: The paths of the transforms to create a mean of
    :type input_transformations: list of str
    """
    if not all(isinstance(x, str) for x in input_transformations):
        raise TypeError('Input transformations need to be str paths!')

    if not isinstance(reference_transform, ElastixParameterFile):
        raise TypeError('The reference transform needs to be an ElastixParameterFile object')

    output_transform = copy.deepcopy(reference_transform)

    # Get the weighting vector for the transforms
    weighting = [1.0 / len(input_transformations)] * len(input_transformations)

    # Substitute relevant fields
    output_transform['Transform'] = 'WeightedCombinationTransform'
    output_transform['TransformParameters'] = weighting
    output_transform['NumberOfParameters'] = len(weighting)
    output_transform['InitialTransformParametersFileName'] = 'NoInitialTransform'
    output_transform['ResultImagePixelType'] = 'float'
    output_transform['SubTransforms'] = input_transformations

    return output_transform


if __name__ == '__main__':
    # Parse arguments and call main function
    parser = argparse.ArgumentParser(description='Combine elastix transformation to create a MeanTransform')
    parser.add_argument('--in', metavar='IN', dest='input', type=str, nargs='+', required=True, help='The transforms to combine')
    parser.add_argument('--out', metavar='OUT', dest='output', type=str, required=True, help='The path to write the resulting transform to')

    args = parser.parse_args()
    create_mean_transform_bin(args.input, args.output)
