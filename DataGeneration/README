These files allow you to extract the simulation data from a root file, into three text files. The created text files will be
called XB_det_data.txt, XB_gun_vals.txt and XB_sum_of_dep_energies.txt. If any of the particles deposits less than
90% (this value can be changed) of its energy in the detector the event is thrown away.

The XB_det_data.txt contain information about how much energy that was deposited in every crystal in the detector. For example,
since the XB-detector (Crystal Ball detector) will register data for 162 crystals, each event results in 162 data points.
The text file will be formatted with space delimiter and with every row corresponding to one event. The first data point in each
row is the deposited energy in crystal with index number, the second data point in each row is the deposited energy in
crystal with index number two etc.

The XB_gun_vals.txt file contain information about the energy and the direction (cosine of angle from the beam pipe or cos(theta)),
thus this file contains the correct answer that you wish to reconstruct using the data in the XB_det_data.txt file. Each row
contains the information for each event, and the data is formatted with space delimiter and the energy and cos(theta) for each
particle are printed in pairs. Thus for an event with two particles, called 1, 2 and 3, the output would be:
E_1 cos(theta_1) E_2 cos(theta_2) E_3 cos(theta_3).

The XB_sum_of_dep_energies.txt file just contain the total deposited energy for each event, i.e. each row in this file is just
one number, and that number is the sum of all the elements in the corresponding row in the XB_det_data.txt file.

The simulation in ggland has to be made with the options: 
--det=tree-gun-edep and --tree=gunlist,<NAME OF ROOT-FILE>.root
For example, the gun-edep option is needed to get information on how much energy each particle deposited in each event.
To for example simulate 1 000 000 events with three gammas fired at each event, one can use:

./land_geant4 --gun=gamma,E=0.01:10MeV,isotropic 
 --gun=gamma,E=0.01:10MeV,isotropic --gun=gamma,E=0.01:10MeV,isotropic
 --events=1000000 --xb=tree-gun-edep --tree=filename.root --np


To use the program: standing in the directory of your root-file, of course with execution right for the files, do: 
./gen_data_gunedep_<NAME OF DETECTOR>.sh <NAME OF ROOT-FILE>.root
and the required txt-files should be generated.

The h102.C- and h102.h-files are deleted after process is finished.

The supported detectors are XB and DALI2 (DALI2 is just called DALI in the scripts).
