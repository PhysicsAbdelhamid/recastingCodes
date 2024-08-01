
# Dossier asymétrique : recastingCodes/CalRatioDisplacedJet/Asym

## Détails du folder recastingCodes/CalRatioDisplacedJet/Asym

### Results 
Folder qui concentre les résultats de la version "lifetime parametrized" (1 fichier .txt avec 1 valeur d'efficacité pour 1 paire de [[cτ1];[cτ2]]) et le UFO asymétrique.
Ce folder est l'output de [comp_map.py]

### Results_ancien
Folder qui concentre les résultats de la version "UFO David Curtin (simplifié)" à travers le code "lifetime parametrized" (1 fichier .txt avec n valeurs d'efficacités pour n cτ)
Ce folder est l'output de [comp_map_Curtin.py]

### comp_function.py
[comp_function.py](./comp_function.py) ou toutes les fonctions sont décrites pour être utilisées par comp_map.py

### comp_map_Curtin.py
[comp_map_Curtin.py](./comp_map_Curtin.py) pour la version "UFO David Curtin (simplifié)" à travers le code "lifetime parametrized".
Il faut rentrer les valeurs mass_Phi, mass_S1, mass_S2, nevent, InDir, OutDir, ctau
L’output est un fichier .txt : "./Results_ancien/Efficiency_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_nevents{nevent}.txt"
   
### comp_map.py
[comp_map.py](./comp_map.py) pour la version "lifetime parametrized" et le UFO asymétrique
L’output est un fichier .txt : "./Asym/Results/Efficiency_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_{ct1}_ct2_{ct2}_nevents{nevent}.txt"

### Jobs_submitter_asym.py 
[Jobs_submitter_asym.py](./Jobs_submitter_asym.py) qui lance les jobs du modèle "lifetime parametrized version" sur le serveur prod1C7 ou prod2C7 
Il faut rentrer les valeurs nevent, InDir, OutDir, mass_Phi, mass_S1, mass_S2, model, (mode=new a laisser), ctau (Lambda = 100 a laisser), (k=1 a laisser)

### Comparaison_efficiencies
Folder où l’on trouve plot + résultats pour comparer les efficacités de "lifetime parametrized version" VS "original version (David UFO simplifié)" en fonction de ctau.
[comparaison_eff_global.py](./Comparaison_efficiencies/Comparaison_eff_global.py) : Il faut rentrer les valeurs mass_Phi, mass_S1, mass_S2, nevent que l’on veut comparer.
Output : "Comparaison_global_mH{mass_Phi}_ms1_{mass_S1}_ms2_{mass_S2}_nevents{nevent}.png" ou ".pdf"

### Global
Folder où l’on trouve plot + résultats pour afficher le plot efficacité VS ctau tout seul sans comparaison pour 1."lifetime parametrized version" et 2."original version (David UFO simplifié)"
[global.py](./Global/Global.py) : Il faut rentrer les valeurs mass_Phi, mass_S1, mass_S2, nevent 
Output : 
Si 1 -> "Global_mH{mass_Phi}_ms1_{mass_S1}_ms2_{mass_S2}_nevents{nevent}_lifetime_parametrized.png" ou ".pdf"
Si 2 -> "Global_mH{mass_Phi}_ms1_{mass_S1}_ms2_{mass_S2}_nevents{nevent}_original_version.png" ou ".pdf"

### writing_script.py
.py pour l'écriture du script: quel UFO model suivre, génération d’évenements, la valeur des paramètres...

## Ordre à suivre pour avoir les résultats :

1. Choisir si l'on veut un résultat avec 1."lifetime parametrized model" ou 2."original version (UFO Curtin simplifié)"
 [!NOTE] Attention, si on veut comparer 1. et 2. il faut prendre la même longuer de ctau
2. Si 1 -> Lancer [Jobs_submitter_asym.py](./Jobs_submitter_asym.py) avec les paramètres souhaités
   Si 2 -> Lancer [comp_map_Curtin.py](./comp_map_Curtin.py) avec les paramètres souhaités
3. Si on veut juste affichier les résultats de 1. et 2. on va dans le folder Global et on lance [global.py](./Global/Global.py) avec les paramètres souhaités
4. Si on veut comparer les efficacités de 1. et 2. VS ctau on va dans le folder Comparaison_efficiencies et on lance [comparaison_eff_global.py](./Comparaison_efficiencies/Comparaison_eff_global.py) avec les paramètres souhaités
5. On lit les résultats !
