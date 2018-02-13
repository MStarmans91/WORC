import WORC

# Create network
network = WORC.WORC('example')

# Below are the minimal inputs: you can also use features instead of images and segmentations.
# You need to change these to your sources: check the FASTR documentation.
# See the WORC Github Wiki for specifications of the various attributes, such as the ones used below
network.images_train.append('xnat://xnat.example.nl/search?project=projectid&subjects=subject[0-9][0-9][0-9]&experiments=*_LIVER&scans=T2W&resources=NIFTI&insecure=true')
network.segmentations_train.append('vfsregex://exmaplemount/somedir/.*.nii')
network.labels_train.append('vfs://mount/somedir/labels.txt')

# WORC passes a config.ini file to all nodes which contains all parameters.
# You can get the default configuration with the defaultconfig function
config = network.defaultconfig()

# This configparser object can be directly supplied to wROC: it will save
# it to a .ini file. You can interact with it as a dictionary and change fields:
# for example, lets change the classifier (default SVM) to Random Forest (RF):
config['Classification']['classifier'] = 'RF'

# See the Github Wiki for all available fields and their explanation.
network.configs.append(config)

# Build the network: the returned network.network object is a FASTR network.
network.build()

# Convert your appended sources to actual sources in the network.
network.set()

# Execute the network
network.execute()
