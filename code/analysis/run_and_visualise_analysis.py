#%% IMPORTS
import os
import mne
from os.path import join
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.inspection import permutation_importance
from analysis_funcs import *


#%% COLLAPSE EVENTS (if you want to)
def collapse_events(y, new_value, old_values=list()):
    new_y = y.copy()
    for old_value in old_values:
        new_y[new_y == old_value] = new_value
    return new_y


# Classify self vs other
import copy
y2=copy.copy(y)
y2[y2==11]=15 # self positive gets 15
y2[y2==12]=15 # self negative gets 15
y2[y2==21]=25 # other positive gets 25
y2[y2==22]=25 # other negative gets 25

#Classify positive vs negative
y3=copy.copy(y)
y3[y3==11]=16 # self pos gets 16
y3[y3==12]=17 # self negative gets 17
y3[y3==21]=16 # other pos gets 16
y3[y3==22]=17 # other negative gets 17

# Classify negative vs button
y4=copy.copy(y)
y4[y4==12]=19 # self neg gets 19
y4[y4==22]=19 # other neg gets 19
y4[y4==23]=23 # button


# Classify positive vs button
y5=copy.copy(y)
y5[y5==11]=18 # self positive
y5[y5==21]=18 # other positive
y5[y5==23]=23 # button


# Self-talk vs button
y6=copy.copy(y)
y6[y6==12]=30 # self neg gets 19
y6[y6==22]=30 # other neg gets 19
y6[y6==11]=30 # self positive
y6[y6==21]=30 # other positive
y6[y6==23]=31 # button


#### ---------- SANITY CHECKS ----------- #

## motor cortex OBS CREATE THE CORRECT MATRIX FOR MOTOR CORTEX
## lh.precentral.label: Left precentral gyrus (motor cortex)
mean_scores_talkbut_MC, y_pred_all_talkbut_MC, y_true_all_talkbut_MC, permutation_scores_talkbut_MC, pvalues_talkbut_MC, feature_importance_talkbut_MC = simple_classification(LPC, y6, triggers=[30,31]) 


# --------------  Left precentral gyrus (motor cortex): Self talk VS Button Sanity Check -------------- #
# Get the 1 and 99 percentiles of the permutation
permutation_scores_talkbut_MC99=np.quantile(permutation_scores_talkbut_MC,0.99,axis=1)
permutation_scores_talkbut_MC01=np.quantile(permutation_scores_talkbut_MC,0.01,axis=1)

plt.figure()
plt.plot(times, mean_scores_talkbut_MC)
plt.fill_between(times, permutation_scores_talkbut_MC01,permutation_scores_talkbut_MC99,color="green", alpha=0.25)
plt.hlines(0.50, times[0], times[-1], linestyle='dashed', color='k')
plt.ylabel('Proportion classified correctly (99% perm. interval)')
plt.xlabel('Time (s)')
plt.title('Left Precentral Gyrus: Self talk vs Button Press')

plt.savefig("figures/precentral_gyrus_talk_button.png", dpi=300)
plt.show()

# Generate and print the classification report
classification_report_str = classification_report(np.concatenate(y_pred_all_talkbut_MC), np.concatenate(y_pred_all_talkbut_MC))
print("Overall Classification Report:")
print(classification_report_str)




## visual cortex, lh.V1_exvivo.label: Left primary visual cortex (V1)

mean_scores_talkbut_LV1, y_pred_all_talkbut_LV1, y_true_all_talkbut_LV1, permutation_scores_talkbut_LV1, pvalues_talkbut_LV1, feature_importance_talkbut_LV1 = simple_classification(LV1,y6, triggers=[30,31])

# -------------- Left IFG Positive Self talk VS Button Sanity Check for LV1 -------------- #
# Get the 1 and 99 percentiles of the permutation for LV1
permutation_scores_talkbut_LV199 = np.quantile(permutation_scores_talkbut_LV1, 0.99, axis=1)
permutation_scores_talkbut_LV101 = np.quantile(permutation_scores_talkbut_LV1, 0.01, axis=1)

plt.figure()
plt.plot(times, mean_scores_talkbut_LV1)
#plt.plot(times, permutations99,linestyle='dashed', color='g')
#plt.plot(times, permutations01,linestyle='dashed', color='g')
plt.fill_between(times, permutation_scores_talkbut_LV101, permutation_scores_talkbut_LV199, color="green", alpha=0.25)
#plt.fill_between(times, permutations05)
plt.hlines(0.50, times[0], times[-1], linestyle='dashed', color='k')
plt.ylabel('Proportion classified correctly (99% perm. interval)')
plt.xlabel('Time (s)')
plt.title('LV1: Self talk vs Button Press')

plt.savefig("figures/talkmotor_sanity_LV1_perms.png", dpi=300)
plt.show()



# Generate and print the classification report for LV1
classification_report_str_LV1 = classification_report(np.concatenate(y_true_all_talkbut_LV1), np.concatenate(y_pred_all_talkbut_LV1))
print("Overall Classification Report for LV1:")
print(classification_report_str_LV1)



# -------------- Left IFG: positive negative self-talk -------------- #


mean_scores_LIFG, y_pred_LIFG, y_true_LIFG, permutation_scores_LIFG, pvalues_LIFG, feature_importance_LIFG = simple_classification(LIFG, 
                                                                   y3,
                                                                   triggers=[16, 17],
                                                                   penalty='l2', 
                                                                   C=1e-3)


# Get the 1 and 99 percentiles of the permutation
permutation_scores_LIFG99=np.quantile(permutation_scores_LIFG,0.99,axis=1)
permutation_scores_LIFG01=np.quantile(permutation_scores_LIFG,0.01,axis=1)


plt.figure()
plt.plot(times, mean_scores_LIFG)
#plt.plot(times, permutations99,linestyle='dashed', color='g')
#plt.plot(times, permutations01,linestyle='dashed', color='g')
plt.fill_between(times, permutation_scores_LIFG01,permutation_scores_LIFG99,color="green", alpha=0.25)
#plt.fill_between(times, permutations05)
plt.hlines(0.50, times[0], times[-1], linestyle='dashed', color='k')
plt.ylabel('Proportion classified correctly (99% perm. interval)')
plt.xlabel('Time (s)')
plt.title('Left Inferior Frontal Gyrus: Positive vs. Negative Self talk')

plt.savefig("figures/LIFG_perms.png", dpi=300)
plt.show()

