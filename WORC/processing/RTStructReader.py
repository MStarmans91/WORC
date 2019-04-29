#!/usr/bin/env python

# Copyright 2016-2019 Biomedical Imaging Group Rotterdam, Departments of
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



"""
@author: Zhenwei.Shi, Maastro

usage:

    paste this file in same directory as your main script file
    and use the following lines

    import DCM_ImgRT_Reader_ZW

    path1=r'c:\ABC\FolderContainingDicomOfimageCTscan'
    path2=r'c:\ABC\FolderContainingDicomOfRTstruct'
    DCM_ImgRT_Reader_ZW.Img_Bimask(image_path=path1,rtstruct_path=path2,ROI_name)

"""

import dicom as pydicom
import os
import numpy as np
from skimage import draw
import SimpleITK as sitk
import re
import glob
import natsort


def get_dicoms(input_dir, logfile=None):
    dicom_files = glob(input_dir + '/*.dcm')
    dicom_files = natsort.natsorted(dicom_files, alg=ns.IGNORECASE)

    dicoms = list()
    for i_dicom in dicom_files:
        dicoms.append(pydicom.read_file(i_dicom))

    return dicoms


def load_dicom(dicom_folder):
    dicom_reader = sitk.ImageSeriesReader()
    dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(dicom_folder)
    dicom_reader.SetFileNames(dicom_file_names)
    dicom_image = dicom_reader.Execute()

    return dicom_image


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def get_pixels(scan):
    image = np.stack([s.pixel_array for s in scan])
    image = image.astype(np.int16)
#    image[image == -2000] = 0
    intercept = scan[0].RescaleIntercept
    slope = scan[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)


def ROI_ref(rtstruct_path, ROI_name):
    mask_vol = load_dicom(rtstruct_path)
    M = mask_vol[0]
    for i in range(len(M.StructureSetROISequence)):
        if str(ROI_name) == M.StructureSetROISequence[i].ROIName:
            ROI_number = M.StructureSetROISequence[i].ROINumber
            break
    return int(ROI_number)


def match_ROIid(rtstruct_path, ROI_number):
    mask_vol = load_dicom(rtstruct_path)
    M = mask_vol[0]
    for j in range(len(M.StructureSetROISequence)):
		if ROI_number == M.ROIContourSequence[j].RefdROINumber:
			break
    return j


def RTStructReader(RTStructFile, ROI_name, image_folder):
    '''
    Input:
        image_path is directory containing dicom files
        rtstruct_path is directory containing RTstruct files
		ROI_name is the ROI stored in RTstructure

    Output:
        mask file written in current directory
        image volume file written in current directory

    '''
	# Volume of DICOM files and RTstruct files
    image_vol = load_dicom(image_folder)
    L = pydicom.read_file(glob.glob(image_folder + '/*.dcm')[0])
    mask_vol = pydicom.read_file(RTStructFile)
    print dir(mask_vol)

	# Slices all have the same basic information including slice size, patient position, etc.
    # L = image_vol[0]
    print image_vol
    # LP = get_pixels(image_vol)
    LP = image_vol
    num_slice = LP.GetSize()[2]

    mask = np.zeros([num_slice, L.Rows, L.Columns],dtype=np.uint8)

    xres=np.array(L.PixelSpacing[0])
    yres=np.array(L.PixelSpacing[1])

    # ROI_number = ROI_ref(rtstruct_path,ROI_name)
    # Check the number of the ROI corresponding to the ROI name to extract
    for i in range(len(mask_vol.StructureSetROISequence)):
        if str(ROI_name) == mask_vol.StructureSetROISequence[i].ROIName:
            ROI_number = mask_vol.StructureSetROISequence[i].ROINumber
            break

    ROI_id = match_ROIid(rtstruct_path,ROI_number)

    for k in range(len(M.ROIContourSequence[ROI_id].ContourSequence)):
        UIDrt=''

        UIDrt = M.ROIContourSequence[ROI_id].ContourSequence[k].ContourImageSequence[0].ReferencedSOPInstanceUID

        for i in range(num_slice):
            UIDslice = image_vol[i].SOPInstanceUID

            if UIDrt==UIDslice:
                sliceOK = i
                break

        x=[]
        y=[]
        z=[]

        m=M.ROIContourSequence[ROI_id].ContourSequence[k].ContourData

        for i in range(0,len(m),3):
            x.append(m[i+1])
            y.append(m[i+0])
            z.append(m[i+2])

        x=np.array(x)
        y=np.array(y)
        z=np.array(z)

        x-= L.ImagePositionPatient[1]
        y-= L.ImagePositionPatient[0]
        z-= L.ImagePositionPatient[2]

        pts = np.zeros([len(x),3])
        pts[:,0] = x
        pts[:,1] = y
        pts[:,2] = z
        a=0
        b=1
        p1 = xres;
        p2 = yres;

        # What is this exactly: Orientation times voxels spacing in a 2x2 matrix?
        m=np.zeros([2,2])
        m[0,0]=image_vol[sliceOK].ImageOrientationPatient[a]*p1
        m[0,1]=image_vol[sliceOK].ImageOrientationPatient[a+3]*p2
        m[1,0]=image_vol[sliceOK].ImageOrientationPatient[b]*p1
        m[1,1]=image_vol[sliceOK].ImageOrientationPatient[b+3]*p2

        # Transform points from reference frame to image coordinates
        m_inv=np.linalg.inv(m)
        pts = (np.matmul((m_inv),(pts[:,[a,b]]).T)).T

        # So you create the mask by making a polygon: how to treat boundary pixels? How was the original contour made?
        mask[sliceOK,:,:] = np.logical_or(mask[sliceOK,:,:],poly2mask(pts[:,0],pts[:,1],[LP.shape[1],LP.shape[2]]))

        LP-=np.min(LP)
        LP=LP.astype(np.float32)
        LP/=np.max(LP)
        LP*=255

        image=sitk.GetImageFromArray(LP.astype(np.float32))
        Mask=sitk.GetImageFromArray(mask)

    return image, Mask


if __name__ == '__main__':
    folder = 'HN1004_20170922_' #interobs05_20170910_CT
    RTStructFile = glob.glob('/home/martijn/Documents/test/' + folder + '/SCANS/DCM/DICOM/*.dcm')[0]
    # image_files = glob.glob('/home/martijn/Documents/test/' + folder + '/SCANS/DCM/DICOM/*.dcm')
    image_folder = '/home/martijn/Documents/test/' + folder + '/SCANS/DCM/DICOM'
    name = 'GTV-1auto(DR)'
    RTStructReader(RTStructFile, name, image_folder)
