Companion code for the manuscript "Signal in the noise: temporal variation in
exponentially growing populations" by Eric Jones, Joshua Derrick, Roger Nisbet,
Will Ludington, and David Sivak.

Raw data from more than 1300 bacterial growth experiments in E. coli and S.
aureus are provided in the .xlsx files. Each file contains data from one
96-well plate, corresponding to two inoculum sizes of >40 replicates each.
* ecoli_growth_rep_* files give optical density measurements over time
* spot_plate_counting_* files give spot plate counts, used to measure inoculum
  size
* well_inoculations_rep_* files specify inoculum sizes for each well of the
  96-well plate

To make Figures 1, 2, 3, 4, and 5 of the main text, run the python program
intrinsic_variability_github.py. This python program inputs, cleans, and
analyzes data; performs calculations and simulations; and plots results. 
