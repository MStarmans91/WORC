import pandas as pd
import os
from WORC.addexceptions import WORCKeyError

# All standard texture features accepted
texture_features = ['GLCM', 'GLDZM', 'GLRLM', 'GLSZM', 'NGLDM', 'NGTDM']


def convert_radiomix_features(input_file, output_folder):
    '''
    Convert .xlsx from RadiomiX to WORC compatible .hdf5 format

    Input:
    --------------

    input_file: .xlsx in which the feature are stored.
    output_folder: folder in which features are stored
    '''

    print('Converting .xlsx from RadiomiX to WORC compatible .hdf5 format...')
    # Check if output folder exists: otherwise create
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Read the input file and extract relevant fields
    f = pd.read_excel(input_file)
    pids = f.values[:, 4]
    segs = f.values[:, 5]
    features = f.values[:, 10:]

    # Read the feature labels, and rename them according to the group they belong to
    feature_labels = list(f.keys()[10:])
    for i in range(0, len(feature_labels)):
        l = feature_labels[i]
        if any(l.startswith(j) for j in texture_features):
            # Texture feature
            feature_labels[i] = 'tf_' + 'RadiomiX_' + l
        elif any(l.startswith(j) for j in ['IH_', 'Stats_']):
            # Histogram feature
            feature_labels[i] = 'hf_' + 'RadiomiX_' + l
        elif l.startswith('Shape_'):
            # Shape feature
            feature_labels[i] = 'sf_' + 'RadiomiX_' + l
        elif l.startswith('LoG_'):
            # LoG feature
            feature_labels[i] = 'logf_' + 'RadiomiX_' + l
        elif l.startswith('Fractal_'):
            # Fractal feature
            feature_labels[i] = 'fracf_' + 'RadiomiX_' + l
        elif l.startswith('LocInt_'):
            # Location feature
            feature_labels[i] = 'locf_' + 'RadiomiX_' + l
        elif l.startswith('RGRD_'):
            # RGRD feature
            feature_labels[i] = 'rgrdf_' + 'RadiomiX_' + l
        elif l.startswith('Wavelet_'):
            # RGRD feature
            feature_labels[i] = 'waveletf_' + 'RadiomiX_' + l
        else:
            raise WORCKeyError(f'Unknown feature {l}.')

    # Initiate labels for pandas file
    panda_labels = ['feature_values',
                    'feature_labels']

    # For each patient, convert features
    for i_patient in range(0, len(pids)):
        feature_values = features[i_patient, :].tolist()

        # Make an output folder per patient, remove invalid symbols.
        output = pids[i_patient] + segs[i_patient]
        output = output.replace(' ', '_')
        output = output.replace('(', '_')
        output = output.replace(')', '_')
        output = os.path.join(output_folder, output)

        # Check if output folder exists: otherwise create
        if not os.path.exists(output):
            os.mkdir(output)

        output = os.path.join(output, 'features.hdf5')

        print(f'\t Writing {output}')

        # Convert to pandas Series and save as hdf5
        panda_data = pd.Series([feature_values,
                                feature_labels],
                               index=panda_labels,
                               name='Image features'
                               )

        # Save the features to the .hdf5 file
        print('\t Saving image features')
        panda_data.to_hdf(output, 'image_features')
