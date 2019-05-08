#!/usr/bin/env python
import argparse
import os
import subprocess
from elastixparameterfile import ElastixParameterFile


def create_def_transform(input_transformation, output_transformation, deformation_file=None, verbose=False):
    """
    Create a deformation transformation from any other transformation. This
    will call elastix to create the deformation field and creates a new
    transformation file based on the input transform.

    :param str input_transformation: path of the input transformation
    :param str output_Trasnformation: path to write the output transformation
    :param bool verbose: flag to control verbosity
    """
    output_dir = os.path.dirname(output_transformation)

    if deformation_file is None:
        command = ['transformix',
                   '-tp', input_transformation,
                   '-out', output_dir,
                   '-def', 'all']

        if verbose:
            print('Running transformix...')

        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()

        if proc.returncode == 0:
            if verbose:
                print('Finished successfully!')
        else:
            if verbose:
                print('Encountered errors:\n{}'.format(stderr))
                print('Aborting!')
            return

        # Create the output deformation field file
        if verbose:
            print('Clean up output...')
        output_deformation = os.path.splitext(output_transformation)[0] + '.def.nii.gz'
        os.rename(os.path.join(output_dir, 'deformationField.nii.gz'), output_deformation)
    else:
        output_deformation = deformation_file

    # Remove log
    if os.path.exists(os.path.join(output_dir, 'transformix.log')):
        os.remove(os.path.join(output_dir, 'transformix.log'))

    # Create the deformation file
    if verbose:
        print('Creating transformation file...')
    create_def_transform_file(input_transformation, output_transformation, os.path.abspath(output_deformation))


def create_def_transform_file(input_transformation, output_transformation, deformation_field):
    """
    This function reads an elastix transform file and creates a deformation transformation.
    It create the fields for the deformation tranformation and copies the fields related to
    the output sampling etc from the input transformation.

    :param str input_transformations: the path of the input transformation file
    :param str output_transformation: path of the output file to be created
    """
    if not isinstance(input_transformation, str):
        raise TypeError('Input transformation need to be a str containing the path of the transformation file!')

    if not isinstance(output_transformation, str):
        raise TypeError('Output transformation needs to be a str where to write the new transformation to')

    # Read the input transform
    input_transform = ElastixParameterFile(input_transformation)

    # Create a ElastixParameterFile object
    output_transform = ElastixParameterFile()
    output_transform.addline(comment=' Initial transform')
    output_transform['InitialTransformParametersFileName'] = 'NoInitialTransform'
    output_transform['HowToCombineTransforms'] = 'Compose'
    output_transform.addline()
    output_transform.addline(comment=' Deformation transform')
    output_transform['Transform'] = 'DeformationFieldTransform'
    output_transform['DeformationFieldFileName'] = deformation_field
    output_transform['DeformationFieldInterpolationOrder'] = 1
    output_transform['NumberOfParameters'] = 0

    # Copy fields from input transform to output
    output_transform.addline()
    output_transform.addline(comment=' Image specific')
    output_transform['FixedImageDimension'] = input_transform['FixedImageDimension']
    output_transform['MovingImageDimension'] = input_transform['MovingImageDimension']
    output_transform['FixedInternalImagePixelType'] = input_transform['FixedInternalImagePixelType']
    output_transform['MovingInternalImagePixelType'] = input_transform['MovingInternalImagePixelType']
    output_transform.addline()
    output_transform.addline(comment=' ResampleInterpolator specific')
    output_transform['ResampleInterpolator'] = input_transform['ResampleInterpolator']
    output_transform['FinalBSplineInterpolationOrder'] = input_transform['FinalBSplineInterpolationOrder']
    output_transform.addline()
    output_transform.addline(comment=' Resampler specific')
    output_transform['Resampler'] = input_transform['Resampler']
    output_transform['DefaultPixelValue'] = input_transform['DefaultPixelValue']
    output_transform['ResultImageFormat'] = input_transform['ResultImageFormat']
    output_transform['ResultImagePixelType'] = input_transform['ResultImagePixelType']

    # Save transform to file
    output_transform.write(output_transformation)


if __name__ == '__main__':
    # Parse arguments and call main function
    parser = argparse.ArgumentParser(description='Create a deformation field transform based on another elastix transformation file')
    parser.add_argument('--in', metavar='IN', dest='input', type=str, required=True, help='The transform convert to deformation field tranform')
    parser.add_argument('--out', metavar='OUT', dest='output', type=str, required=True, help='The path to write the resulting transform to')
    parser.add_argument('--def', metavar='DEFORMATION_FIELD', dest='deform', type=str, required=False,
                        help='DeformationField to use for the Transform (if not given, it creates one using the elastix on the PATH')

    args = parser.parse_args()
    create_def_transform(args.input, args.output, args.deform, verbose=True)
