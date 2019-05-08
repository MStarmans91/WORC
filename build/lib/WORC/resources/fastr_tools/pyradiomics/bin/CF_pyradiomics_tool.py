#!/usr/bin/env python

# Copyright 2017-2018 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from CF_pyradiomics import AllFeatures as CalcFeatures


def main():
    parser = argparse.ArgumentParser(description='Feature extraction')
    parser.add_argument('-im', '--im', metavar='image', nargs='+',
                        dest='im', type=str, required=True,
                        help='Images to calculate features on')
    parser.add_argument('-md', '--md', metavar='metadata', dest='md',
                        type=str, required=False, nargs='+',
                        help='Clinical data on patient (DICOM)')
    parser.add_argument('-sem', '--sem', metavar='semantics', dest='sem',
                        type=str, required=False, nargs='+',
                        help='Semantic Features')
    parser.add_argument('-seg', '--seg', metavar='segmentation', dest='seg',
                        type=str, required=True, nargs='+',
                        help='Segmentation to calculate features on')
    parser.add_argument('-para', '--para', metavar='Parameters', nargs='+',
                        dest='para', type=str, required=False,
                        help='Parameters')
    parser.add_argument('-out', '--out', metavar='Features',
                        dest='out', type=str, required=False,
                        help='Patient features output (HDF)')
    args = parser.parse_args()

    if type(args.im) is list:
        args.im = ''.join(args.im)

    if type(args.seg) is list:
        args.seg = ''.join(args.seg)

    if type(args.out) is list:
        args.out = ''.join(args.out)

    featureVector = CalcFeatures(image=args.im, mask=args.seg)

    if 'rdf' in args.out:
        # Write output to rdf
        # import rdflib and some namespace
        from rdflib import Graph, URIRef, BNode, Literal, Namespace
        from rdflib.namespace import RDF, FOAF

        # convert python object to RDF
        print "-----------------------------------------------------------"
        print "			RDF Output:"
        print ""
        Img = Graph()
        lung1_image = URIRef("http://example.org/CT-Image")
        Img.add((lung1_image, RDF.type, FOAF.Image))

        list_key = featureVector.keys()
        list_value = featureVector.values()
        for i in range(len(list_key)):
            tmp_value = Literal(list_value[i])
            tmp_name = list_key[i]
            Img.add((lung1_image, FOAF.tmp_name, tmp_value))

        print Img.serialize(format='turtle')
        # Create a rdf file for storing output
        Img.serialize(args.out, format="pretty-xml")

    elif 'hdf5' in args.out:
        # Write output to hdf5
        import numpy as np
        import pandas as pd

        # Assign features to corresponding groups
        shape_labels = list()
        shape_features = list()
        histogram_labels = list()
        histogram_features = list()
        GLCM_labels = list()
        GLCM_features = list()
        GLRLM_labels = list()
        GLRLM_features = list()
        GLSZM_labels = list()
        GLSZM_features = list()

        for featureName in featureVector.keys():
            if 'shape' in featureName:
                shape_labels.append(featureName)
                shape_features.append(featureVector[featureName])
            if 'firstorder' in featureName:
                histogram_labels.append(featureName)
                histogram_features.append(featureVector[featureName])
            if 'glcm' in featureName:
                GLCM_labels.append(featureName)
                GLCM_features.append(featureVector[featureName])
            if 'glrlm' in featureName:
                GLRLM_labels.append(featureName)
                GLRLM_features.append(featureVector[featureName])
            if 'glszm' in featureName:
                GLSZM_labels.append(featureName)
                GLSZM_features.append(featureVector[featureName])

        # Convert feature to single dictionary containing PD series
        features = dict()
        pandas_dict = dict(zip(shape_labels, shape_features))
        shape_dict = dict()
        shape_dict['all'] = pd.Series(pandas_dict)
        shape_features = pd.Series(shape_dict)
        features['shape_features'] = shape_features

        pandas_dict = dict(zip(histogram_labels, histogram_features))
        histogram_dict = dict()
        histogram_dict['all'] = pd.Series(pandas_dict)
        histogram_features = pd.Series(histogram_dict)
        features['histogram_features'] = histogram_features

        GLCM_dict = dict(zip(GLCM_labels, GLCM_features))
        GLRLM_dict = dict(zip(GLRLM_labels, GLRLM_features))
        GLSZM_dict = dict(zip(GLSZM_labels, GLSZM_features))

        texture_features = dict()
        texture_features['GLCM'] = pd.Series(GLCM_dict)
        texture_features['GLRLM'] = pd.Series(GLRLM_dict)
        texture_features['GLSZM'] = pd.Series(GLSZM_dict)

        texture_features = pd.Series(texture_features)
        features['texture_features'] = texture_features

        # We also return just the arrray
        image_feature_array = list()

        for _, feattype in features.iteritems():
            for _, imfeatures in feattype.iteritems():
                image_feature_array.extend(imfeatures.values)

        image_feature_array = np.asarray(image_feature_array)
        image_feature_array = image_feature_array.ravel()

        panda_labels = ['image_features', 'image_features_array']
        panda_data = pd.Series([features, image_feature_array],
                               index=panda_labels,
                               name='Image features'
                               )

        panda_data.to_hdf(args.out, 'image_features')


if __name__ == '__main__':
    main()
