import xnat
import os
import sys
import shutil
from glob import glob

from xnat.exceptions import XNATResponseError


def download_subject(project, subject, datafolder, session, verbose=False):
    # Download all data and keep track of resources
    download_counter = 0
    resource_labels = list()
    for e in subject.experiments:
        resmap = {}
        experiment = subject.experiments[e]

        # FIXME: Need a way to smartly check whether we have a matching RT struct and image
        # Current solution: We only download the CT sessions, no PET / MRI / Other scans
        # Specific for STW Strategy BMIA XNAT projects

        if experiment.session_type is None:  # some files in project don't have _CT postfix
            print(f"\tSkipping patient {subject.label}, experiment {experiment.label}: type is not CT but {experiment.session_type}.")
            continue

        if '_CT' not in experiment.session_type:
            print(f"\tSkipping patient {subject.label}, experiment {experiment.label}: type is not CT but {experiment.session_type}.")
            continue

        for s in experiment.scans:
            scan = experiment.scans[s]
            print(("\tDownloading patient {}, experiment {}, scan {}.").format(subject.label, experiment.label,
                                                                             scan.id))
            for res in scan.resources:
                resource_label = scan.resources[res].label
                if resource_label == 'NIFTI':
                    # Create output directory
                    outdir = datafolder + '/{}'.format(subject.label)
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)

                    resmap[resource_label] = scan
                    print(f'resource is {resource_label}')
                    scan.resources[res].download_dir(outdir)
                    resource_labels.append(resource_label)
                    download_counter += 1

    # Parse resources and throw warnings if they not meet the requirements
    subject_name = subject.label
    if download_counter == 0:
        print(f'[WARNING] Skipping subject {subject_name}: no (suitable) resources found.')
        return False

    if 'NIFTI' not in resource_labels:
        print(f'[WARNING] Skipping subject {subject_name}: no NIFTI resources found.')
        return False

    if resource_labels.count('NIFTI') < 2:
        print(f'[WARNING] Skipping subject {subject_name}: only one NIFTI resource found, need two (mask and image).')
        return False
    elif resource_labels.count('NIFTI') > 2:
        count = resource_labels.count('NIFTI')
        print(f'[WARNING] Skipping subject {subject_name}: {str(count)} NIFTI resources found, need two (mask and image).')
        return False

    # Check what the mask and image folders are
    NIFTI_folders = glob(os.path.join(outdir, '*', 'scans', '*', 'resources', 'NIFTI', 'files'))
    if 'mask' in glob(os.path.join(NIFTI_folders[0], '*.nii.gz'))[0]:
        NIFTI_image_folder = NIFTI_folders[1]
        NIFTI_mask_folder = NIFTI_folders[0]
    else:
        NIFTI_image_folder = NIFTI_folders[0]
        NIFTI_mask_folder = NIFTI_folders[1]

    NIFTI_files = glob(os.path.join(NIFTI_image_folder, '*'))
    if len(NIFTI_files) == 0:
        print(f'[WARNING] Skipping subject {subject_name}: image NIFTI resources is empty.')
        shutil.rmtree(outdir)
        return False

    NIFTI_files = glob(os.path.join(NIFTI_mask_folder, '*'))
    if len(NIFTI_files) == 0:
        print(f'[WARNING] Skipping subject {subject_name}: mask NIFTI resources is empty.')
        shutil.rmtree(outdir)
        return False

    # Patient is included, so cleanup folder structure
    shutil.move(os.path.join(NIFTI_image_folder, 'image.nii.gz'),
              os.path.join(outdir, 'image.nii.gz'))
    shutil.move(os.path.join(NIFTI_mask_folder, 'mask_GTV-1.nii.gz'),
              os.path.join(outdir, 'mask.nii.gz'))

    for folder in glob(os.path.join(outdir, '*', 'scans')):
        folder = os.path.dirname(folder)
        shutil.rmtree(folder)

    return True


def download_project(project_name, xnat_url, datafolder, nsubjects=10,
                     verbose=True):

    # Connect to XNAT and retreive project
    session = xnat.connect(xnat_url)
    project = session.projects[project_name]

    # Create the data folder if it does not exist yet
    datafolder = os.path.join(datafolder, project_name)
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)

    subjects_len = len(project.subjects)
    if nsubjects == 'all':
        nsubjects = subjects_len
    else:
        nsubjects = min(nsubjects, subjects_len)

    subjects_counter = 1
    downloaded_subjects_counter = 0
    for s in range(0, subjects_len):
        s = project.subjects[s]
        print(f'Working on subject {subjects_counter}/{subjects_len}')
        subjects_counter += 1

        success = download_subject(project_name, s, datafolder, session, verbose)
        if success:
            downloaded_subjects_counter += 1

        # Stop downloading if we have reached the required number of subjects
        if downloaded_subjects_counter == nsubjects:
            break

    # Disconnect the session
    session.disconnect()
    if downloaded_subjects_counter < nsubjects:
        raise ValueError(f'Number of subjects downloaded {downloaded_subjects_counter} is smaller than the number required {nsubjects}.')

    print('Done downloading!')


def download_HeadAndNeck(datafolder=None, nsubjects=10):
    if datafolder is None:
        # Download data to path in which this script is located + Data
        cwd = os.getcwd()
        datafolder = os.path.join(cwd, 'Data')
        if not os.path.exists(datafolder):
            os.makedirs(datafolder)

    xnat_url = 'https://xnat.bmia.nl/'
    project_name = 'stwstrategyhn1'
    download_project(project_name, xnat_url, datafolder, nsubjects=nsubjects,
                     verbose=True)


if __name__ == '__main__':
    download_HeadAndNeck()
