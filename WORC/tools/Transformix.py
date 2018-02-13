import SimpleITK as sitk
import fastr
import numpy as np
import os
import WORC.addexceptions as WORCexceptions


class Transformix(object):
    def __init__(self):
        self.network = self.create_network()
        self.MovingImage = None
        self.TransformParameterMap = None

    def create_network(self):
        self.network = fastr.Network(id_="transformix")

        self.MovingImageSource = self.network.create_source('ITKImageFile', id_='MovingImage')
        self.ParameterMapSource = self.network.create_source('ElastixTransformFile', id_='ParameterFile')

        self.transformix_node = self.network.create_node('transformix_dev', id_='transformix')
        self.transformix_node.inputs['image'] = self.MovingImageSource.output
        self.transformix_node.inputs['transform'] = self.ParameterMapSource.output

        self.outimage = self.network.create_sink('ITKImageFile', id_='sink_image')
        self.outimage.inputs['input'] = self.transformix_node.outputs['image']

        self.network.draw_network(img_format='svg')
        self.network.dumpf('{}.json'.format(self.network.id), indent=2)

    def execute(self):
        SimpleTransformix = sitk.SimpleTransformix()
