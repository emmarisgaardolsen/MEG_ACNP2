#%% IMPORTS
import os
import mne
from os.path import join
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pickle
from preproc_funcs import *


raw_path = '/work/834761/'
subjects_dir = '/work/835482'

epochs_list = preprocess_sensor_space_data('0109', '20230926_000000',
        raw_path=raw_path,
        decim=10) ##CHANGE TO YOUR PATHS # don't go above decim=10

times = epochs_list[0].times # get time points for later


X_sensor,y = get_X_and_y(epochs_list)


# Saving the objects:
#f = open('/work/807746/emma_folder/notebooks/12b_MEG_analysis/python/00_CLEAN/wholebrocas_epochs_matrices/MEG_InSpe_epochs.plk','wb')
#pickle.dump([epochs_list,times,X_sensor,y], f)
#f.close()

# Getting back the objects:
#f = open('/work/807746/emma_folder/notebooks/12b_MEG_analysis/python/00_CLEAN/wholebrocas_epochs_matrices/MEG_InSpe_epochs.plk,'rb')
#[epochs_list,times,X_sensor,y] = pickle.load(f)
#f.close()



subjects_dir = '/work/835482'
labels_to_merge = ['lh.parsopercularis.label', 'lh.parsorbitalis.label', 'lh.parstriangularis.label']
output_path = '/work/807746/emma_folder/notebooks/12b_MEG_analysis/python/00_CLEAN/custom_labels/LIFG.label'  # Set the output path and label name

LIFG_label = merge_labels_and_save(subjects_dir, '0109', labels_to_merge, output_path, 'LIFG.label')

LIFG, y = preprocess_source_space_data('0109', '20230926_000000', 
                                       raw_path=raw_path, 
                                       subjects_dir=subjects_dir,
                                       epochs_list=epochs_list,
                                       label='ROI', 
                                       custom_label_path='/work/807746/emma_folder/notebooks/12b_MEG_analysis/python/00_CLEAN/custom_labels/LIFG-lh.label')
    
RIFG, y = preprocess_source_space_data('0109', '20230926_000000', 
                                       raw_path=raw_path, 
                                       subjects_dir=subjects_dir,
                                       epochs_list=epochs_list,
                                       label='ROI', 
                                       custom_label_path='/work/807746/emma_folder/notebooks/12b_MEG_analysis/python/00_CLEAN/custom_labels/RIFG-rh.label')

# for self vs other analysis
mPFC, y = preprocess_source_space_data('0109', '20230926_000000',
        raw_path=raw_path, 
        subjects_dir=subjects_dir,
        label='rh.superiorfrontal.label',
        epochs_list=epochs_list)


LPC, y = preprocess_source_space_data('0109', '20230926_000000', 
                                       raw_path=raw_path, 
                                       subjects_dir=subjects_dir,
                                       epochs_list=epochs_list,
                                       label='lh.precentral.label')


LV1, y = preprocess_source_space_data('0109', '20230926_000000',
        raw_path=raw_path, 
        subjects_dir=subjects_dir,
        label='lh.V1_exvivo.label',
        epochs_list=epochs_list)


RV1, y = preprocess_source_space_data('0109', '20230926_000000',
        raw_path=raw_path, 
        subjects_dir=subjects_dir,
        label='rh.V1_exvivo.label',
        epochs_list=epochs_list)




                                  
# Assuming you have already obtained RIFG, LIFG, and mPFC along with y
# Save LIFG and y as plk files

#f = open('/work/807746/emma_folder/notebooks/12b_MEG_analysis/python/00_CLEAN/wholebrocas_epochs_matrices/LIFG.plk','wb')
#pickle.dump([LIFG,y], f)
#f.close()


#f = open('/work/807746/emma_folder/notebooks/12b_MEG_analysis/python/00_CLEAN/wholebrocas_epochs_matrices/RIFG.plk','wb')
#pickle.dump([RIFG,y], f)
#f.close()


#f = open('/work/807746/emma_folder/notebooks/12b_MEG_analysis/python/00_CLEAN/wholebrocas_epochs_matrices/mPFC.plk','wb')
#pickle.dump([mPFC,y], f)
#f.close()


                                        

