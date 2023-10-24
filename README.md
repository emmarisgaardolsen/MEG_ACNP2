# Portfolio 3: MEG Acquisition - Analysis & Report

The current repository contains the code used for portfolio 3 (MEG Acquisition - Analysis & Report) assignment for the course  *Advanced Cognitive Neuroscience (F23)*  at the Cognitive Science MSc, Aarhus University. The code includes preprocessing of raw MEG data obtained at Aarhus University Hospital’s neuroimaging facilities, as well as the data analysis and visualisation. 

The project attempts to train a Gaussian Naive Bayes (GNB) machine learning classifier to discern between positive and negative self-talk within the left inferior frontal gyrus, a brain region believed to be involved in several language abilities, including inner speech. 

## This repository contains the following

```

├── README.md
├── setup.sh  
├── env_to_ipynb_kernel.sh 
├── requirements.txt       
├── code                      <--- code for replicating analysis
    ├── analysis   
      ├── analysis_funcs.py   <--- functions used in analysis
      ├── run_and_visualise_analysis.py <-- script for running analysis and saving figures
      ├── figures.            
        ├── figures outputted from running `analysis_funcs.py`
    ├── preprocessing        
      ├── preproc_funcs.py    <--- functions used for preprocessing
      ├── run_preproc.py.     <--- code for running preprocessing

```

## Study group members
Members of study group 2 are: 

- Rafał Prus (RPR), [202100779@post.au.dk](mailto:202100779@post.au.dk)
- Nanna Marie Steenholdt (NMS), [201805892@post.au.dk](mailto:201805892@post.au.dk)
- Marc Barcelos (MAB), [202302260@post.au.dk](mailto:202302260@post.au.dk)
- Emma Risgaard Olsen (EOL), [202006507@post.au.dk](mailto:202006507@post.au.dk)**

##  Triggers in MEG data
|       Event Trigger       |   Explanation   |
|------------------|-----------|
|     11       |    Positive self-talk, self-chosen word     |
|     21       |    Positive self-talk, word provided by experimenter (ugly/pretty)     |
|     12       |    Negative self-talk, self-chosen word     |
|     22       |    22: Negative self-talk, word provided by experiment     |
|     23       |    Button press stimuli     |
|  202   |   Button pressed by participant     |

## Data Organisation
The code cannot be run without access to the raw data which cannot be published on GitHub due to GDPR constraints, but all analysis steps can be seen by inspecting the scripts and notebooks. The data was structured in the following manner, and running the code requires you to have the data organised in a corresponding folder structure.

````
├── Repository/directory containing the code
├── 834761          <--- Raw MEG neuroimaging data
│   └── 0subid      <--- `subid` should be replaced by subject ID
├── 835482          <--- Freesurfer output data and labels
│   ├── 0subid     <--- `subid` should be replcaed by subject ID
│   └── fsaverage
````
