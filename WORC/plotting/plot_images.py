#!/usr/bin/env python

# Copyright 2016-2022 Biomedical Imaging Group Rotterdam, Departments of
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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from matplotlib.ticker import NullLocator
import matplotlib.colors as colors
import SimpleITK as sitk
from skimage import morphology
import WORC.addexceptions as ae
import scipy.ndimage as nd


def extract_boundary(contour, radius=2):
    returnitk = False
    if type(contour) is sitk.Image:
        contour = sitk.GetArrayFromImage(contour)
        returnitk = True

    disk = morphology.disk(radius)
    if len(contour.shape) == 3:
        for ind in range(contour.shape[0]):
            contour_d = morphology.binary_dilation(contour[ind, :, :], disk)
            contour_e = morphology.binary_erosion(contour[ind, :, :], disk)
            contour[ind, :, :] = np.bitwise_xor(contour_d, contour_e)
    else:
        contour_d = morphology.binary_dilation(contour, disk)
        contour_e = morphology.binary_erosion(contour, disk)
        contour = np.bitwise_xor(contour_d, contour_e)

    contour = contour.astype(np.uint16)  # To be compatible with SimpleITK
    if returnitk:
        return sitk.GetImageFromArray(contour)
    else:
        return contour


def slicer(image, mask=None, output_name=None, output_name_zoom=None,
           thresholds=[-5, 5], zoomfactor=4, dpi=500, normalize=True,
           expand=False, boundary=False, square=False, flip=True, rot90=0,
           alpha=0.40, axis='axial', index=None, color='cyan', radius=2,
           colormap='gray', fill=False):
    """Plot slice of image where mask is largest, with mask as overlay.

    image and mask should both be arrays
    """
    # Determine figure size by spacing
    if len(image.GetSize()) == 2:
        # 2D Image
        spacing = [float(image.GetSpacing()[0]), float(image.GetSpacing()[1])]
        imsize = [float(image.GetSize()[0]), float(image.GetSize()[1])]
        figsize = (imsize[0]*spacing[0]/100.0, imsize[1]*spacing[1]/100.0)
    else:
        # 3D Image
        # Determine figure size by spacing
        spacing = [float(image.GetSpacing()[0]), float(image.GetSpacing()[1]), float(image.GetSpacing()[2])]
        imsize = [float(image.GetSize()[0]), float(image.GetSize()[1]), float(image.GetSize()[2])]

        if axis == 'axial':
            figsize = (imsize[0]*spacing[0]/100.0, imsize[1]*spacing[1]/100.0)
        elif axis == 'coronal':
            figsize = (imsize[0]*spacing[0]/100.0, imsize[2]*spacing[2]/100.0)
        elif axis == 'transversal':
            figsize = (imsize[1]*spacing[1]/100.0, imsize[2]*spacing[2]/100.0)

    # Convert images to numpy arrays
    image = sitk.GetArrayFromImage(image)
    if mask is not None:
        mask = sitk.GetArrayFromImage(mask)

    if len(image.shape) == 2:
        # 2D image, so image and mask are already single slices
        imslice = image
        maskslice = mask
    else:
        # Determine which axial slice has the largest area
        if index is None:
            if mask is not None:
                if axis == 'axial':
                    areas = np.sum(mask, axis=2).tolist()
                elif axis == 'coronal':
                    areas = np.sum(mask, axis=1).tolist()
                elif axis == 'transversal':
                    areas = np.sum(mask, axis=0).tolist()

                index = areas.index(max(areas))
            else:
                if axis == 'axial':
                    index = int(image.shape[2]/2)
                elif axis == 'coronal':
                    index = int(image.shape[1]/2)
                elif axis == 'transversal':
                    index = int(image.shape[0]/2)

        if axis == 'axial':
            imslice = image[index, :, :]
        elif axis == 'coronal':
            imslice = image[:, index, :]
        elif axis == 'transversal':
            imslice = image[:, :, index]
        else:
            raise ValueError(f'{axis} is not a valid value for the axis, should be axial, coronal, or transversal.')

        if mask is not None:
            if axis == 'axial':
                maskslice = mask[index, :, :]
            elif axis == 'coronal':
                maskslice = mask[:, index, :]
            elif axis == 'transversal':
                maskslice = mask[:, :, index]
            else:
                raise ValueError(f'{axis} is not a valid value for the axis, should be axial, coronal, or transversal.')
        else:
            maskslice = None

    if rot90 != 0:
        print(f'\t Rotating {rot90 * 90} degrees.')
        imslice = np.rot90(imslice, rot90)
        if mask is not None:
            maskslice = np.rot90(maskslice, rot90)

    if flip:
        print('\t Flipping up-down.')
        imslice = np.flipud(imslice)
        if mask is not None:
            maskslice = np.flipud(maskslice)

    if fill:
        print('\t Filling holes.')
        maskslice = nd.binary_fill_holes(maskslice)
        maskslice = maskslice.astype(np.uint8)

    if mask is not None:
        if boundary:
            print('\t Extracting boundary.')
            maskslice = extract_boundary(maskslice, radius)

    if normalize:
        print('\t Normalizing.')
        imslice = sitk.GetImageFromArray(imslice)
        imslice = sitk.Normalize(imslice)
        imslice = sitk.GetArrayFromImage(imslice)

    if square:
        sz = imslice.shape
        if sz[0] != sz[1]:
            print('\t Making image square')
            if sz[0] > sz[1]:
                diff = sz[0] - sz[1]
                newimslice = np.ones((sz[0], sz[0])) * np.min(imslice)
                newimslice[:, int(diff/2.0):sz[1] + int(diff/2.0)] = imslice
                imslice = newimslice

                if mask is not None:
                    newmaskslice = np.zeros((sz[0], sz[0]))
                    newmaskslice[:, int(diff/2.0):sz[1] + int(diff/2.0)] = maskslice
                    maskslice = newmaskslice
            else:
                diff = sz[1] - sz[0]
                newimslice = np.ones((sz[1], sz[1])) * np.min(imslice)
                newimslice[int(diff/2.0):sz[0] + int(diff/2.0), :] = imslice
                imslice = newimslice

                if mask is not None:
                    newmaskslice = np.zeros((sz[1], sz[1]))
                    newmaskslice[int(diff/2.0):sz[0] + int(diff/2.0), :] = maskslice
                    maskslice = newmaskslice

    if expand:
        print('\t Expanding.')
        imslice = sitk.GetImageFromArray(imslice)
        if mask is not None:
            maskslice = sitk.GetImageFromArray(maskslice)

        newsize = (4, 4)
        imslice = sitk.Expand(imslice, newsize)
        if mask is not None:
            maskslice = sitk.Expand(maskslice, newsize)

        # Adjust the size
        if axis == 'axial':
            figsize = (imsize[0]*spacing[0]/100.0, imsize[1]*spacing[1]/100.0)
        elif axis == 'coronal':
            figsize = (imsize[0]*spacing[0]/100.0, imsize[2]*spacing[2]/100.0)
        elif axis == 'transversal':
            figsize = (imsize[1]*spacing[1]/100.0, imsize[2]*spacing[2]/100.0)

        imslice = sitk.GetArrayFromImage(imslice)
        if mask is not None:
            maskslice = sitk.GetArrayFromImage(maskslice)

    # Threshold the image if desired
    if thresholds:
        print("\t Thresholding.")
        imslice[imslice < thresholds[0]] = thresholds[0]
        imslice[imslice > thresholds[1]] = thresholds[1]

    # Plot the image and overlay the mask
    fig = plot_im_and_overlay(imslice, maskslice, figsize=figsize, alpha=alpha,
                              color=color, colormap=colormap)

    # Save Output
    print('\t Saving output.')
    fig.savefig(output_name, bbox_inches='tight', pad_inches=0, dpi=dpi)

    # Save some memory
    # del fig

    if mask is not None:
        if output_name_zoom is not None:
            # Create a bounding box and save zoomed image
            imslice, maskslice = bbox_2D(imslice, maskslice, padding=[20, 20])
            imsize = [float(imslice.shape[0]), float(imslice.shape[1])]

            # NOTE: As these zoomed images get small, we double the spacing
            spacing = spacing * zoomfactor
            figsize = (imsize[0]*spacing[0]/100.0, imsize[1]*spacing[1]/100.0)
            fig = plot_im_and_overlay(imslice, maskslice, figsize=figsize, alpha=alpha)
            fig.savefig(output_name_zoom, bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close('all')

            # Save some memory
            del fig, image, mask

    return imslice, maskslice


def plot_im_and_overlay(image, mask=None, figsize=(3, 3), alpha=0.40,
                        color='cyan', colormap='gray', colorbar=False):
    """Plot an image in a matplotlib figure and overlay with a mask."""
    # Define colormap
    validmaps = ['gray', 'turbo', 'jet']
    if colormap not in validmaps:
        raise ae.WORCKeyError(f'Colormap {colormap} is not valid. Should be one of {validmaps}.')

    # Create a normalized colormap for the image and mask
    imin = np.min(image)
    imax = np.max(image)
    norm_im = colors.Normalize(vmin=imin, vmax=imax, clip=False)

    cmap = plt.get_cmap("Blues")
    cmap.set_under(color="white", alpha=0)
    cmap.set_over(color=color, alpha=1)
    normO = colors.Normalize(vmin=0.5, vmax=0.75, clip=False)

    # Plot and save the full image
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    mappable = ax.imshow(image, cmap=colormap, norm=norm_im, interpolation="bilinear")
    if mask is not None:
        ax.imshow(mask, cmap=cmap, norm=normO, alpha=alpha, interpolation="bilinear")

    # Add colorbar
    if colorbar:
        fig.colorbar(mappable)

    # Alter aspect ratio according to figure size
    aspect = figsize[0]/figsize[1]
    ax.set_aspect(aspect)

    # Set locator to zero to make sure padding is removed upon saving
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())

    # Turn axis grid of
    ax.axis('off')

    return fig


def bbox_2D(img, mask, padding=[1, 1], img2=None):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # print rmin, rmax, cmin, cmax
    rmin = max(0, rmin - padding[0])
    rmax = min(img.shape[0], rmax+padding[0]+1)
    cmin = max(0, cmin - padding[1])
    cmax = min(img.shape[1], cmax+padding[1]+1)

    img = img[rmin:rmax, cmin:cmax]
    mask = mask[rmin:rmax, cmin:cmax]
    if img2 is None:
        return img, mask
    else:
        img2 = img2[rmin:rmax, cmin:cmax]
        return img, mask, img2
