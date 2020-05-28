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
import pandas as pd
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
                        dest='para', type=str, required=True,
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

    if type(args.para) is list:
        args.para = ''.join(args.para)

    featureVector, settings = CalcFeatures(image=args.im, mask=args.seg,
                                           parameters=args.para)

    if 'rdf' in args.out:
        # Write output to rdf
        # import rdflib and some namespace
        from rdflib import Graph, URIRef, BNode, Literal, Namespace
        from rdflib.namespace import RDF, FOAF

        # convert python object to RDF
        print("-----------------------------------------------------------")
        print("			RDF Output:")
        print("")
        Img = Graph()
        lung1_image = URIRef("http://example.org/CT-Image")
        Img.add((lung1_image, RDF.type, FOAF.Image))

        list_key = featureVector.keys()
        list_value = featureVector.values()
        for i in range(len(list_key)):
            tmp_value = Literal(list_value[i])
            tmp_name = list_key[i]
            Img.add((lung1_image, FOAF.tmp_name, tmp_value))

        print(Img.serialize(format='turtle'))
        # Create a rdf file for storing output
        Img.serialize(args.out, format="pretty-xml")

    elif 'hdf5' in args.out:
        # Write output to hdf5
        image_type = 'MR'
        feature_values = featureVector.values()
        feature_labels = featureVector.keys()
        panda_labels = ['image_type', 'parameters', 'feature_values',
                        'feature_labels']

        panda_data = pd.Series([image_type, settings, feature_values,
                                feature_labels],
                               index=panda_labels,
                               name='Image features'
                               )

        print('Saving image features')
        panda_data.to_hdf(args.out, 'image_features')


if __name__ == '__main__':
    main()
