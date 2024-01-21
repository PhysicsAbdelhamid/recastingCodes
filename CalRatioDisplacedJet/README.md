

# To process the recasting, follow these instructions:

> [!IMPORTANT]
> The main codes present in this git have been develpped by Mr. Thomas Chehab from [this git](https://github.com/ThomasChehab/recastingCodes) under the supervision of Dr. Louie Corpe and Dr. Andreas Goudelis. [Have a look at their note](./Notes_on_recasting_the_ATLAS_search_for_neutral_LLPs.pdf).

## Setup:

1. First of all, you will need to install this version of madgraph *MG5_aMC_v3.4.2* from https://launchpad.net/mg5amcnlo/+download by executing:

```wget https://launchpad.net/mg5amcnlo/3.0/3.4.x/+download/MG5_aMC_v3.4.2.tar.gz ; tar -xf MG5_aMC_v3.4.2.tar.gz ; rm -rf MG5_aMC_v3.4.2.tar.gz ; cd MG5_aMC_v3_4_2```

**All the tasks will be performed within this MG5_aMC_v3_4_2 folder**

1. Now you need to download and place the model named *HAHM_MG5model_v3* from http://insti.physics.sunysb.edu/~curtin/hahm_mg.html and extract the tarball using: 

```wget https://feynrules.irmp.ucl.ac.be/raw-attachment/wiki/HAHM/HAHM_MG5model_v3.zip ; unzip HAHM_MG5model_v3.zip ; rm -rf HAHM_MG5model_v3.zip```

1. You will also need the ATLAS public results from HEP data file in this github repository:

```git clone https://github.com/ThomasChehab/recastingCodes.git ; scp recastingCodes/CalRatioDisplacedJet/HEPData*..yaml .```

1. Now we can launch madgraph by: 

```./bin/mg5_aMC ```

1. In case you face an issue with python version (less that 3.7), you may upgrade your python version if you have the sudo rights to do, or you can create and activate a virtual python environment ( ```python3 -m venv env ; ./env/Scripts/activate``` ), equivalently and if you have access to cvmfs, you can setup ATLAS and Athena by: 

```source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh ; asetup AnalysisBase,24.2.35```

1. After entering in the MG konsole, install Pythia8 by writing:

```install pythia8```

It will probably ask you to install LHAPDF6 as well, just accept and proceed.

1. Now you (MG) need to convert the model file by writing :

```convert model ./HAHM_MG5model_v3/HAHM_variableMW_v3_UFO```

MG is now ready to generate events on your machine ! You can exit it.

```exit```

## Lunching MadGraph jobs and generating the plots:

1. Now, you are ready to create the scripts that MG will use to genetage the events, go to *recastingCodes/CalRatioDisplacedJet/*:

```cd recastingCodes/CalRatioDisplacedJet/```

And **open the "Jobs_submitter.py" file and update it according to your needs** (instructions are written in it) and lunch it when you finish::

```python3 Jobs_submitter.py```

The generation can be relatively long depending on your computing ressources. 

1. To complete the task, you can now process the data and obtain the plots and limits as well:


> [!NOTE]
> Do not forget to activate your python environment or source ATLAS and Athena each time you connect.

You need to add the python madgraph library to Python Path:

```export PYTHONPATH=Full/Path/To/MG5_aMC_v3_4_2/:$PYTHONPATH```

And **update the output path in the Computation_Map.py** code before executing it: 

```python3 Computation_Map.py```

> [!NOTE]
> You may need to ```pip install numpy scipy matplotlib tqdm uproot```

The limits and values will be saved in text files within the *recastingCodes/CalRatioDisplacedJet/Plots_High* and *Plots_Low* folders so that you do not have to redo the entire runs.


