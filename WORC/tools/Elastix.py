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

import SimpleITK as sitk
import fastr
from fastr.api import ResourceLimit
import numpy as np
import os
import WORC.addexceptions as WORCexceptions


class Elastix(object):
    def __init__(self):
        # Which version of elastix and transformix tools should be used
        self.elastix_toolname = 'Elastix'
        self.transformix_toolname = 'Transformix'

        # self.Elastix = sitk.SimpleElastix()
        self.create_network('pairwise')
        self.FixedImage = []
        self.MovingImage = []
        self.FixedMask = []
        self.MovingMask = []
        self.ToTransform = []
        self.ParameterMaps = []  # sitk.VectorOfParameterMap()

        self.TransformedImage = 'vfs://tmp/WORC_Elastix/results/elastix_output_image_{sample_id}_{cardinality}.nii.gz'
        self.TransformedSeg = 'vfs://tmp/WORC_Elastix/results/elastix_output_seg_{sample_id}_{cardinality}.nii.gz'
        self.TransformParameters = 'vfs://tmp/WORC_Elastix/results/elastix_output_trans_{sample_id}_{cardinality}.txt'
        self.fastr_tmpdir = os.path.join(fastr.config.mounts['tmp'], 'WORC_Elastix')
        # TODO: Add initial transformation

    def getparametermap(self, model='affine', size=(512, 512, 128)):
        nvoxels = size[0]*size[1]*size[2]
        if model == 'rigid':
            parameter_map = sitk.GetDefaultParameterMap("rigid")
            parameter_map["AutomaticTransformInitialization"] = ["true"]
            parameter_map["Metric"] = ["AdvancedMattesMutualInformation"]
            parameter_map["NumberOfResolutions"] = ["4"]  # save time/memory
            parameter_map["NumberOfSpatialSamples"] = [str(nvoxels*0.0001)]  # save time/memory
            parameter_map["MaximumNumberOfIterations"] = ["1000"]  # save time/memory
            parameter_map["WriteResultImage"] = ["true"]  # save time/memory
            parameter_map["ErodeMask"] = ["true"]

        elif model == 'affine':
            parameter_map = sitk.GetDefaultParameterMap("affine")
            parameter_map["AutomaticTransformInitialization"] = ["false"]
            parameter_map["Metric"] = ["AdvancedMattesMutualInformation"]
            parameter_map["NumberOfResolutions"] = ["4"]  # save time/memory
            parameter_map["NumberOfSpatialSamples"] = [str(nvoxels*0.0001)]  # save time/memory
            parameter_map["MaximumNumberOfIterations"] = ["1000"]  # save time/memory
            parameter_map["WriteResultImage"] = ["true"]  # save time/memory
            parameter_map["ErodeMask"] = ["true"]

        elif model == 'bspline':
            parameter_map = sitk.GetDefaultParameterMap("bspline")
            parameter_map["AutomaticTransformInitialization"] = ["false"]
            # parameter_map["Metric"] = ["AdvancedMattesMutualInformation"]
            parameter_map["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
            parameter_map["Metric0Weight"] = ["1.0"]
            parameter_map["Metric1Weight"] = ["750.0"] # *100 is about 5 percent
            parameter_map["NumberOfResolutions"] = ["4"]  # save time/memory
            parameter_map["NumberOfSpatialSamples"] = [str(nvoxels*0.0001)]  # save time/memory
            parameter_map["WriteResultImage"] = ["true"]  # save time/memory
            parameter_map["MaximumNumberOfIterations"] = ["1000"]  # save time/memory
            # parameter_map["ImageSampler"] = ["RandomSamplerSparseMask"]
            parameter_map["ErodeMask"] = ["true"]

        else:
            raise KeyError("Model {} cannot be found!").format(model)

        return parameter_map

    def create_network(self, nettype):
        if nettype == 'pairwise':
            # Create the network
            self.network = fastr.create_network(id="elastix_pair")

            # Create Sources
            self.FixedImageSource = self.network.create_source('ITKImageFile', id='FixedImage')
            self.FixedMaskSource = self.network.create_source('ITKImageFile', id='FixedMask')
            self.MovingImageSource = self.network.create_source('ITKImageFile', id='MovingImage')
            self.MovingMaskSource = self.network.create_source('ITKImageFile', id='MovingMask')
            self.ToTransformSource = self.network.create_source('ITKImageFile', id='ToTransform')
            self.ParameterMapSource = self.network.create_source('ElastixParameterFile', id='ParameterMaps', node_group='par')
            # Elastix requires the output folder as a sink
            # self.OutputFolderSource = self.network.create_sink('Directory', id_='Out')

            # Create Elastix node and links
            self.elastix_node = self.network.create_node('self.elastix_toolname', tool_version='unknown', id='elastix')
            self.elastix_node.inputs['fixed_image'] = self.FixedImageSource.output
            self.elastix_node.inputs['fixed_mask'] = self.FixedMaskSource.output
            self.elastix_node.inputs['moving_image'] = self.MovingImageSource.output
            self.elastix_node.inputs['moving_mask'] = self.MovingMaskSource.output
            # self.OutputFolderSource.input = self.elastix_node.outputs['directory']
            self.link_param = self.network.create_link(self.ParameterMapSource.output, self.elastix_node.inputs['parameters'])
            self.link_param.collapse = 'par'

            # Create Sinks
            self.outtrans = self.network.create_sink('ElastixTransformFile', id='sink_trans')
            self.outimage = self.network.create_sink('ITKImageFile', id='sink_image')
            self.outseg = self.network.create_sink('ITKImageFile', id='sink_seg')
            self.outtrans.inputs['input'] = self.elastix_node.outputs['transform']

            # Transform output image
            self.transformix_node = self.network.create_node('self.transformix_toolname', tool_version='unknown', id='transformix')
            self.transformix_node.inputs['image'] = self.MovingImageSource.output
            self.transformix_node.inputs['transform'] = self.elastix_node.outputs['transform'][-1]
            self.outimage.inputs['input'] = self.transformix_node.outputs['image']

            # First change the FinalBSplineInterpolationOrder to  0 for the segmentation
            self.changeorder_node = self.network.create_node('elastixtools/EditElastixTransformFile:0.1', tool_version='0.1', id='editelpara')
            self.link_trans = self.network.create_link(self.elastix_node.outputs['transform'][-1], self.changeorder_node.inputs['transform'])
            # self.link_trans.converge = 0
            # self.link_trans.collapse = 'FixedImage'
            # self.link_trans.expand = True

            # Co[y metadata from image to segmentation as Elastix uses this
            self.copymetadata_node = self.network.create_node('itktools/0.3.2/CopyMetadata:1.0', tool_version='1.0', id='copymetadata')
            self.copymetadata_node.inputs['source'] = self.MovingImageSource.output
            self.copymetadata_node.inputs['destination'] = self.ToTransformSource.output

            # Then transform the segmentation
            self.transformix_node_seg = self.network.create_node('self.transformix_toolname', tool_version='unknown', id='transformix_seg')
            self.transformix_node_seg.inputs['image'] = self.copymetadata_node.outputs['output']
            self.transformix_node_seg.inputs['transform'] = self.changeorder_node.outputs['transform'][-1]
            self.outseg.inputs['input'] = self.transformix_node_seg.outputs['image']
        else:
            # Create the network
            self.network = fastr.create_network(id="elastix_group")

            # Create Sources
            self.FixedImageSource = self.network.create_source('ITKImageFile', id='FixedImage')
            self.FixedMaskSource = self.network.create_source('ITKImageFile', id='FixedMask')
            self.ToTransformSource = self.network.create_source('ITKImageFile', id='ToTransform')
            self.ParameterMapSource = self.network.create_source('ElastixParameterFile', id='ParameterMaps', node_group='par')
            # Elastix requires the output folder as a sink
            # self.OutputFolderSource = self.network.create_sink('Directory', id_='Out')

            # Create Elastix node and links
            self.elastix_node = self.network.create_node('self.elastix_toolname', tool_version='unknown', id='elastix')
            self.elastix_node.inputs['fixed_image'] = self.FixedImageSource.output
            self.elastix_node.inputs['fixed_mask'] = self.FixedMaskSource.output
            self.elastix_node.inputs['moving_image'] = self.FixedImageSource.output
            self.elastix_node.inputs['moving_mask'] = self.FixedMaskSource.output
            # self.OutputFolderSource.input = self.elastix_node.outputs['directory']
            self.link_param = self.network.create_link(self.ParameterMapSource.output, self.elastix_node.inputs['parameters'])
            self.link_param.collapse = 'par'

            # Create Sinks
            self.outtrans = self.network.create_sink('ElastixTransformFile', id='sink_trans')
            self.outimage = self.network.create_sink('ITKImageFile', id='sink_image')
            self.outseg = self.network.create_sink('ITKImageFile', id='sink_seg')
            self.outtrans.inputs['input'] = self.elastix_node.outputs['transform']

            # Transform output image
            self.transformix_node = self.network.create_node('self.transformix_toolname', tool_version='unknown', id='transformix')
            self.transformix_node.inputs['image'] = self.MovingImageSource.output
            self.transformix_node.inputs['transform'] = self.elastix_node.outputs['transform'][-1]
            self.outimage.inputs['input'] = self.transformix_node.outputs['image']

            # First change the FinalBSplineInterpolationOrder to  0 for the segmentation
            self.changeorder_node = self.network.create_node('elastixtools/EditElastixTransformFile:0.1', tool_version='0.1', id='editelpara')
            self.changeorder_node.inputs['set'] = ["FinalBSplineInterpolationOrder=0"]
            self.link_trans = self.network.create_link(self.elastix_node.outputs['transform'], self.changeorder_node.inputs['transform'][-1])
            # self.link_trans.converge = 0
            # self.link_trans.collapse = 'FixedImage'
            # self.link_trans.expand = True

            # Co[y metadata from image to segmentation as Elastix uses this
            self.copymetadata_node = self.network.create_node('itktools/0.3.2/CopyMetadata:1.0', tool_version='1.0', id='copymetadata')
            self.copymetadata_node.inputs['source'] = self.MovingImageSource.output
            self.copymetadata_node.inputs['destination'] = self.ToTransformSource.output

            # Then transform the segmentation
            self.transformix_node_seg = self.network.create_node('self.transformix_toolname', tool_version='unknown', id='transformix_seg')
            self.transformix_node_seg.inputs['image'] = self.copymetadata_node.outputs['output']
            self.transformix_node_seg.inputs['transform'] = self.changeorder_node.outputs['transform'][-1]
            self.outseg.inputs['input'] = self.transformix_node_seg.outputs['image']

    def addchangeorder(self):
        # For the last file, change also the dependence on the previous files
        N_parameterfiles = 1
        sources = list()
        for num in range(0, N_parameterfiles):
            if num != 0:
                # We also need to refer to the correct initial transform files
                intrans = 'InitialTransformParametersFileName=' +\
                    os.path.join(self.fastr_tmpdir, 'editelpara', 'transform_' + str(num - 1) + '.txt')
                sources.append("FinalBSplineInterpolationOrder=0" + intrans)
            else:
                sources.append("FinalBSplineInterpolationOrder=0")

        # self.set = self.network.create_source("AnyType", id_='setorder')
        # self.source_data['setorder'] = sources
        # self.changeorder_node.inputs['set'] = self.set.output
        self.changeorder_node.inputs['set'] = sources

    def create_bbox(self, seg, pad=[2, 25, 25]):
        '''
        Create a bounding box around an input segmentation
        with a certain padding
        '''
        segim = sitk.ReadImage(seg)
        segim = sitk.GetArrayFromImage(segim)
        segim[segim != 0] = 1

        nonzero_x = np.nonzero(np.sum(segim, (1, 2)))[0]
        nonzero_y = np.nonzero(np.sum(segim, (0, 2)))[0]
        nonzero_z = np.nonzero(np.sum(segim, (0, 1)))[0]

        x1, x2 = nonzero_x[0], nonzero_x[-1]
        y1, y2 = nonzero_y[0], nonzero_y[-1]
        z1, z2 = nonzero_z[0], nonzero_z[-1]
        mask = np.zeros(segim.shape)
        x1 = max(0, x1 - pad[0])
        x2 = min(segim.shape[0], x2 + pad[0])
        y1 = max(0, y1 - pad[1])
        y2 = min(segim.shape[1], y2 + pad[1])
        z1 = max(0, z1 - pad[2])
        z2 = min(segim.shape[2], z2 + pad[2])
        mask[x1:x2, y1:y2, z1:z2] = 1
        mask = sitk.GetImageFromArray(mask)

        return mask

    def execute(self):
        # Check if minimal input is supplied
        if not self.FixedImage:
            message = "You need to supply a fixed image for registration."
            raise WORCexceptions.WORCNotImplementedError(message)

        if not self.MovingImage:
            message = "You need to supply a moving image for registration."
            raise WORCexceptions.WORCNotImplementedError(message)

        if len(self.ParameterMaps) == 0:
            message = "You need to supply at leat one parameter map for registration."
            raise WORCexceptions.WORCNotImplementedError(message)

        # Set moving and fixed image sources
        self.source_data = dict()
        self.source_data['FixedImage'] = self.FixedImage
        self.source_data['MovingImage'] = self.MovingImage

        # Create a temporary directory to use
        tempdir = os.path.join(fastr.config.mounts['tmp'], 'WORC_Elastix')
        if not os.path.exists(tempdir):
            os.makedirs(tempdir)

        # Set the parameter map sources
        if type(self.ParameterMaps) is list:
            # Files, thus just provide them to the elastix node
            self.source_data['ParameterMaps'] = self.ParameterMaps
        else:
            # Use SimpleTransformix to save the maps and add them
            SimpleElastix = sitk.SimpleElastix()
            self.source_data['ParameterMaps'] = list()
            for num, f in enumerate(self.ParameterMaps):
                filename = ('ParameterMap{}.txt').format(str(num))
                fname = os.path.join(tempdir, filename)
                SimpleElastix.WriteParameterFile(f, fname)
                sourcename = 'vfs://tmp/WORC_Elastix/' + filename
                self.source_data['ParameterMaps'].append(sourcename)

        # Based on number of parameterfiles, add nodes to train FinalBSplineInterpolationOrder
        self.addchangeorder()

        # Set the mask sources if provided
        # if self.FixedMask is not None:
        self.source_data['FixedMask'] = self.FixedMask

        # if self.MovingMask is not None:
        self.source_data['MovingMask'] = self.MovingMask

        # Add other images to transform if given
        # if self.ToTransform is not None:
        self.source_data['ToTransform'] = self.ToTransform

        # Set the network sinks
        self.sink_data = dict()
        self.sink_data['sink_trans'] = self.TransformParameters
        self.sink_data['sink_image'] = self.TransformedImage
        self.sink_data['sink_seg'] = self.TransformedSeg

        # Set outputfolder if given
        # if self.OutputFolder:
        #     self.sink_data['Out'] = self.OutputFolder
        # else:
        #     self.sink_data['Out'] = 'vfs://tmp/WORC_Elastix/output'

        # print self.sink_data['Out']

        # Execute the network
        self.network.draw(file_path='WORC_Elastix.svg', img_format='svg')
        self.network.dumpf('{}.json'.format(self.network.id), indent=2)
        self.network.execute(self.source_data, self.sink_data, tmpdir=self.fastr_tmpdir)

        # Automatically read in the images (and parameter maps if requested?)
