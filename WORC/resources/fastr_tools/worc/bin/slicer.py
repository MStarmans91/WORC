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
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
import matplotlib.colors as colors


def slicer(image, mask, output_name, output_name_zoom, thresholds=[-240, 160],
           zoomfactor=4, reverse=True, flipud=True):
    '''
    image and mask should both be arrays
    '''

    # Determine figure size by spacing
    spacing = float(image.GetSpacing()[0])
    imsize = [float(image.GetSize()[0]), float(image.GetSize()[1])]
    figsize = (imsize[0]*spacing/100.0, imsize[1]*spacing/100.0)
    # dpi = int(200/spacing)
    dpi = 100.0

    # Convert images to numpy arrays
    image = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(mask)

    # Manipulate mask if required
    if reverse:
        print('Reversing mask in zero axis.')
        mask = mask[::-1, :, :]

    if flipud:
        print("Flipping scan on first axis.")
        mask = np.flip(mask, 1)

    # Determine which axial slice has the largest area
    areas = np.sum(mask, axis=1).tolist()
    max_ind = areas.index(max(areas))
    imslice = image[max_ind, :, :]
    maskslice = mask[max_ind, :, :]

    # Threshold the image if desired
    if thresholds:
        imslice[imslice < thresholds[0]] = thresholds[0]
        imslice[imslice > thresholds[1]] = thresholds[1]

    # Plot the image and overlay the mask
    fig = plot_im_and_overlay(imslice, maskslice, figsize=figsize)

    # Save Output
    fig.savefig(output_name, bbox_inches='tight', pad_inches=0, dpi=dpi)

    # Save some memory
    del fig

    # Create a bounding box and save zoomed image
    imslice, maskslice = bbox_2D(imslice, maskslice, padding=[20, 20])
    imsize = [float(imslice.shape[0]), float(imslice.shape[1])]

    # NOTE: As these zoomed images get small, we double the spacing
    spacing = spacing * zoomfactor
    figsize = (imsize[0]*spacing/100.0, imsize[1]*spacing/100.0)
    fig = plot_im_and_overlay(imslice, maskslice, figsize=figsize)
    fig.savefig(output_name_zoom, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close('all')

    # Save some memory
    del fig, image, mask
    return imslice, maskslice


def plot_im_and_overlay(image, mask, figsize=(3, 3), alpha=0.15):
    '''
    Plot an image in a matplotlib figure and overlay with a mask.
    '''
    # Create a normalized colormap for the image and mask
    imin = np.min(image)
    imax = np.max(image)
    norm_im = colors.Normalize(vmin=imin, vmax=imax, clip=False)

    cmap = plt.get_cmap("Reds")
    cmap.set_under(color="white", alpha=0)
    cmap.set_over(color="r", alpha=1)
    normO = colors.Normalize(vmin=0.5, vmax=0.75, clip=False)

    # Plot and save the full image
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image, cmap=plt.cm.gray, norm=norm_im, interpolation="bilinear")
    ax.imshow(mask, cmap=cmap, norm=normO, alpha=alpha, interpolation="bilinear")

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


def main():
    parser = argparse.ArgumentParser(description='Feature extraction')
    parser.add_argument('-im', '--im', metavar='image', nargs='+',
                        dest='im', type=str, required=True,
                        help='Images to calculate features on')
    parser.add_argument('-seg', '--seg', metavar='seg', dest='seg',
                        type=str, required=True, nargs='+',
                        help='Segmentation that can be used in normalization')
    parser.add_argument('-out', '--out', metavar='out',
                        dest='out', type=str, required=True,
                        help='Image output (PNG)')
    parser.add_argument('-outzoom', '--outzoom', metavar='outzoom',
                        dest='outzoom', type=str, required=False,
                        help='Image output zoomed in (PNG)')
    args = parser.parse_args()

    # Convert list inputs to strings
    if type(args.im) is list:
        args.im = ''.join(args.im)

    if type(args.seg) is list:
        args.seg = ''.join(args.seg)

    if type(args.out) is list:
        args.out = ''.join(args.out)

    if type(args.outzoom) is list:
        args.outzoom = ''.join(args.outzoom)

    # Load the image and segmentation
    im = sitk.ReadImage(args.im)
    seg = sitk.ReadImage(args.seg)

    # Apply slicing
    slicer(im, seg, args.out, args.outzoom)


if __name__ == '__main__':
    main()
