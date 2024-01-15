To process the recasting, please follow these instructions:

- First of all, you will need to install this version of madgraph : MG5_aMC_v3.4.2.tar.gz from : https://launchpad.net/mg5amcnlo/+download by executing:

> wget https://launchpad.net/mg5amcnlo/3.0/3.4.x/+download/MG5_aMC_v3.4.2.tar.gz ; tar -xf MG5_aMC_v3.4.2.tar.gz ; rm -rf MG5_aMC_v3.4.2.tar.gz ; cd MG5_aMC_v3_4_2

All the tasks will be performed within this MG5_aMC_v3_4_2 folder.

- Now you need to download and place the model named : HAHM_MG5model_v3 from http://insti.physics.sunysb.edu/~curtin/hahm_mg.html and extract the tarball in the MG5_aMC_v3_4_2 folder using: 

> wget https://feynrules.irmp.ucl.ac.be/raw-attachment/wiki/HAHM/HAHM_MG5model_v3.zip ; unzip HAHM_MG5model_v3.zip ; rm -rf HAHM_MG5model_v3.zip

- You will also need the ATLAS public results from HEP data file in this github repository and place them in your MG5_aMC_v3_4_2 folder:

> git clone https://github.com/ThomasChehab/recastingCodes.git ; scp recastingCodes/CalRatioDisplacedJet/HEPData*..yaml .

- Now we can launch madgraph by: 

> ./bin/mg5_aMC

- In case you face an issue with python version (less that 3.7), you may upgrade your python version if you have the sudo rights to do, or you can create a virtual python environment, equivalently, you can setup ATLAS and Athena releases if you have access to cvmfs by: 

> source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh ; asetup AnalysisBase,24.2.35

- After entering in the MG konsole, install Pythia8 by writing:

MG5_aMC> install pythia8

It will probably ask you to install LHAPDF6 as well, just accept and proceed.

- Now you (MG) need to convert the model file by writing :

MG5_aMC> convert model ./HAHM_MG5model_v3/HAHM_variableMW_v3_UFO

MG is now ready to generate events on your machine ! You can exit it.

MG5_aMC> exit

- Now, you are ready to create the scripts that MG will use to genetage the events, go to "recastingCodes/CalRatioDisplacedJet/"

> cd recastingCodes/CalRatioDisplacedJet/

And open the "Jobs_submitter.py" file and update it according to your needs (instructions are written in it). Lunch it when you finish:

> python3 Jobs_submitter.py

The generation can be relatively long depending on your computing ressources. 

# Some text about plotting the map and limits..

