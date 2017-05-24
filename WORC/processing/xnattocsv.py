import xnat
import csv
import sys
from natsort import natsorted


def xnattocsv(url, project, output):
    subjects = xnat.connect(url).projects[project].subjects
    patient_IDs = natsorted(subjects.keys())

    data = dict()
    for pid in patient_IDs:
        sub = subjects[pid]
        plabel = sub.data['label']
        print(("Processing patient {}.").format(plabel))
        data[plabel] = dict()
        experiment_IDs = sub.experiments.keys()

        for eid in experiment_IDs:
            experiment = sub.experiments[eid]
            elabel = experiment.data['label']
            data[plabel][elabel] = dict()
            scans = experiment.scans
            scan_IDs = scans.keys()

            for sid in scan_IDs:
                scandata = scans[sid].data
                scan_num = scandata['ID']
                if 'series_description' in scandata.keys():
                    series_descr = scandata['series_description']
                else:
                    series_descr = 'unknown'
                data[plabel][elabel][scan_num] = series_descr

    with open(output, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['PatientID', 'Experiment', 'ScanID', 'SeriesDescr.'])
        for patientID in data.keys():
            for enum, experimentID in enumerate(data[patientID].keys()):
                for snum, scanID in enumerate(data[patientID][experimentID].keys()):
                    seriesdescr = data[patientID][experimentID][scanID].encode('utf-8')
                    if enum == 0 and snum == 0:
                        writer.writerow([patientID, experimentID, scanID, seriesdescr])
                    elif snum == 0:
                        writer.writerow(['', experimentID, scanID, seriesdescr])
                    else:
                        writer.writerow(['', '', scanID, seriesdescr])


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise IOError("This function accepts three arguments")
    else:
        url = str(sys.argv[1])
        project = str(sys.argv[2])
        output = str(sys.argv[3])
    xnattocsv(url, project, output)
