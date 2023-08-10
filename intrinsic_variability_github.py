#!/usr/bin/env python
#
# intrinsic_variability_CLEAN_12-15-22.py
# Eric W. Jones (jones.eric93@gmail.com)
#
# This software file performs numerical simulations and generates figures for
# the manuscript "The signal in the noise: intrinsic variability in
# exponentially growing systems"
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import birdepy as bd
import pandas as pd
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.special as special
import random
import datetime
import warnings
import pickle
import os


##############################
# HELPER CLASSES
##############################

class SimulateSBP():
    """Encapsulated class that generates stochastic trajectories of the simple
       birth (Markov) process (SBP).
       Inputs: p ... self-referential class name
               ic ... int, initial condition of the Markov process
               t_max ... float, simulation end time
               t_len ... int, number of time points for recording abundance
               num_trajs ... int, number of stochastic trajectories to simulate
               mu ... float, growth rate of the simple birth process 
                      (birth rate B_n = mu*n)
               seed ... int, seed for trajectory simulations
               h ... float, simulation timestep for the Euler approximation"""

    def __init__(p, ic=5, t_max=15, t_len=1001, num_trajs=1000, mu=1, seed=0, h=2e-3):
        """Initializes class with passed parameter values """
        p.ic = ic
        p.t_max = t_max
        p.t_len = t_len
        p.num_trajs = num_trajs
        p.mu = mu
        p.seed = seed # seed for trajectory simulations
        p.h = h

        # personalized filename extensions
        p.ext_ME = ('ic_{}_tmax_{}_tlen_{}_numtraj_{}_mu_{}_seed_{}'
                    .format(p.ic, p.t_max, p.t_len, p.num_trajs, p.mu, p.seed))

    def b_rate(p, x, mu):
        """Birth rate B_n for the simple birth process is mu*x """
        return mu*x

    def d_rate(p, x, mu):
        """Death rate D_n for the simple birth process is 0 """
        return 0

    def get_SBP_trajs(p, load_data=True):
        """Perform simulations of the simple birth process, using the python
           package birdepy.
           Inputs: load_data ... boolean, indicates whether data should be
                                 loaded (if possible)
           Outputs: ts ... length t_len list, spanning from 0 to t_max
                    trajs ... length num_trajs list of length t_len lists,
                              abundance trajectory realizations of the SBP """
        if load_data:
            try:
                with open('vars/SBP_numerical_{}.pi'.format(p.ext_ME), 'rb') as f:
                    ts, trajs = pickle.load(f)
                print('  loaded simple birth process simulations: {}'.format(p.ext_ME))
                return ts, trajs
            except FileNotFoundError:
                print('  running simple birth process simulations: {}'.format(p.ext_ME))

        ts = np.linspace(0, p.t_max, p.t_len)
        trajs = bd.simulate.discrete(
            p.mu, model='custom', z0=p.ic, times=ts,
            b_rate=p.b_rate, d_rate=p.d_rate,
            k=p.num_trajs, seed=p.seed, method='ea', display=False, tau=p.h)

        with open('vars/SBP_numerical_{}.pi'.format(p.ext_ME), 'wb') as f:
            pickle.dump((ts, trajs), f)

        return ts, trajs

##############################
# RAW DATA INPUT/CLEANING
##############################

def load_exp_data_e_coli(load_data=True, verbose=False, if_plot=False,
                         return_raw_data=False):
    """Loads the raw data for the E. coli experimental system.
    Inputs: load_data ... bool, indicates whether to load cached data
            verbose ... bool, indicates whether to print diagnostic text
    Outputs: exp_data_dict ... dict containing information about growth
                               experiments
    Usage: exp_data_dict[dataset][inoculum][replicate] = [ts, ys]
           dataset ... str, date that the experiment was performed
           inoculum ... float, predicted inoculum size for a set of replicates
                        (based on taking the mean of the corresponding spot
                        plate counts)
           replicate ... int, ranging from 0 to the number of replicates in an
                         experiment minus 1
           ts ... times of the growth trajectory
           ys ... abundances (either OD or counts) of the growth trajectory """

    if load_data:
        try:
            with open('vars/e_coli_growth_curves.pi', 'rb') as f:
                exp_gcs = pickle.load(f)
                print('loaded bacterial growth curves')
                return exp_gcs
        except FileNotFoundError:
            print('loading bacterial growth curves')

    # scan over raw data files to identify when experiments were performed
    # E. coli experiments: 05-03-22, 05-19-22, 05-25-22, 06-01-22, 06-15-22
    # S. aureus experiments: 06-27-22, 07-01-22
    exp_dates = []
    for file in os.listdir():
        if file[:12] == 'ecoli_growth':
            exp_num = file.split('_')[3]
            exp_date = file.split('_')[4][:8]
            exp_dates.append(exp_date)
    exp_dates = set(exp_dates)
    exp_dates = sorted(exp_dates)

    data_directory = {exp_date: [[], [], None] for exp_date in exp_dates}
    # usage: data_directory['05-19-22'] = 
    #        [[ecoli_growth_rep_3_05-19-22.xlsx,
    #          ecoli_growth_rep_4_05-19-22.xlsx,
    #          ecoli_growth_rep_5_05-19-22.xlsx], 
    #         [well_inoculations_rep_3_05-19-22.xlsx,
    #          well_inoculations_rep_4_05-19-22.xlsx,
    #          well_inoculations_rep_5_05-19-22.xlsx],
    #         spot_plate_counting_05-19-22.xlsx]

    for file in os.listdir():
        if file[:12] == 'ecoli_growth':
            exp_date = file.split('_')[4][:8]
            data_directory[exp_date][0].append(file)
        if file[:17] == 'well_inoculations':
            exp_date = file.split('_')[4][:8]
            data_directory[exp_date][1].append(file)
        if file[:19] == 'spot_plate_counting':
            exp_date = file.split('_')[3][:8]
            data_directory[exp_date][2] = file

    exp_num_to_date = {}
    # usage: exp_num_to_date[1] = '05-19-22'
    for i,date in enumerate(exp_dates):
        exp_num_to_date[i] = date

    all_exps = list(range(len(exp_dates)))
    exp_gcs = {}

    # omit experient 0 on 5/3/22 (trial run)
    # omit experiment 2 on 5/25/22 (no spot plating measurements)
    for dataset in all_exps[1:]:
    #for dataset in [1]:
        if verbose: print()
        if verbose: print(exp_num_to_date[dataset])
        traj_filenames = data_directory[exp_num_to_date[dataset]][0]
        well_plate_filenames = data_directory[exp_num_to_date[dataset]][1]
        spot_plate_filename = data_directory[exp_num_to_date[dataset]][2]

        traj_filenames = sorted(traj_filenames)
        well_plate_filenames = sorted(well_plate_filenames)

        gcs = []
        for i,(f1,f2) in enumerate(zip(traj_filenames, well_plate_filenames)):
            gc = get_growth_curves(f1, f2, exp_num_to_date[dataset], i,
                                   if_plot=if_plot,
                                   return_raw_data=return_raw_data)
            gcs.append(gc)
        gcs = pd.concat(gcs, axis=1)

        time_column = gcs['Time'].iloc[:,0]
        gcs = gcs.drop(columns='Time', axis=1)
        gcs['Time'] = time_column

        gcs = calculate_inocula_with_spot_plating(gcs, spot_plate_filename,
                                                  verbose=verbose)

        if if_plot:
            plot_growth_trajectories(gcs, exp_num_to_date[dataset])

        exp_gcs[dataset] = gcs
        if verbose: print()

    with open('vars/e_coli_growth_curves.pi', 'wb') as f:
        pickle.dump(exp_gcs, f)

    return exp_gcs

def calculate_inocula_with_spot_plating(gc, spot_plate_filename, verbose=False):
    """Modifies inoculum sizes (columns of gc) to agree with the spot plate
    counting data 'spot_plate_counting_##-##-##.xlsx'. Primarily,
    'well_inoculations_rep_##_##-##-##.xlsx' uses inoculum sizes that are based
    on the average of the spot plate counts (including zeros); this function
    replaces those values by the average of the spot plate count (excluding
    zeros).
    Inputs: gc ... pandas database, contains growth trajectories with column
                   names corresponding to inoculum sizes
            spot_plate_filename ... str, filename corresponding to spot plate
                                    counts for an inoculum size 
            verbose ... bool, indicates whether to print diagnostic text
    Outputs: gc ... pandas database, contains growth trajectories with inoculum
                    sizes updated based on spot plate counting data
    Usage: gc['Time'] = [0, ..., t_end]
           gc[inoculum] = [[0.001, ..., 0.989], [0.014, ..., 1.132], ...] """

    # load spot plate counting data 
    sp = pd.read_excel(spot_plate_filename)
    expected_inocula = [elem for elem in sp.columns if type(elem) != str]
    # expected_inocula values are the same as in well_inoculations_*.xlsx

    inocula_order = list(gc.columns) # inocula sizes (before modification)
    inocula = sorted([elem for elem in set(inocula_order) if (type(elem) != str
                                                              and elem > 0)])
    corrected_inocula = [] # inocula sizes (after modification)
    replacement_dict = {} # replacement_dict[old_inoculum] = new_inoculum

    # fill in replacement_dict for all available spot plate data
    for i,inoc in enumerate(expected_inocula):
        # discard the final entry (an average, not data)
        spot_plate_measurements = list(sp[inoc])[:-1]
        nonzero_elems = [elem for elem in spot_plate_measurements if elem > 0]
        nonzero_average = np.round(np.mean(nonzero_elems), 1)
        corrected_inocula.append(nonzero_average)
        replacement_dict[inocula[i]] = nonzero_average

    # Some inoculum sizes are too large to spot plate count. Instead, we
    # multiply old_inoculum by a scaling factor (based on the largest
    # spot-plated abundance)
    max_val = max(replacement_dict.keys())
    for inoc in inocula:
        if inoc not in replacement_dict:
            adjusted_inoc = np.round(inoc*replacement_dict[max_val]/max_val, 1)
            replacement_dict[inoc] = adjusted_inoc

    # get inocula sizes (after modification)
    for i,inoc in enumerate(inocula_order):
        if type(inoc) == str: continue
        if inoc == 0: continue
        inocula_order[i] = replacement_dict[inoc]
    gc.columns = inocula_order

    if verbose:
        print('based on spot plate counts, the following inocula were adjusted:')
        for inoc in replacement_dict:
            print('  ', inoc, '->', replacement_dict[inoc])

    return gc


def get_growth_curves(f1, f2, date, plate_num, if_plot=False,
                      return_raw_data=False, verbose=False):
    """Loads raw data files f1 and f2 and extracts growth curve data and
    inoculation data (via spot plate counting).
    Inputs: f1 ... str, filename corresponding to bacterial growth curves for
                   an inoculum size 
            f2 ... str, filename linking well plate locations to inoculum sizes
            date ... str, date on which experiments were performed (used to
                     load spot plate counts)
            plate_num ... int, specifies 96-well plate on a given date
            if_plot ... boolean, specifies whether to plot growth curves of
                        each 96-well plate
            return_raw_data ... boolean, if True returns raw data (i.e. without
                                omitted growth trajectories, with background
                                OD)
            verbose ... bool, indicates whether to print diagnostic text
    Outputs: gc ... dict, contains abundance data across many replicates for an
                    inoculum size
    Usage: gc['Time'] = [0, ..., t_end]
           gc[inoculum] = [[0.001, ..., 0.989], [0.014, ..., 1.132], ...] """

    # notation: gc = growth curve; plate = spot plate counts
    gc = pd.read_excel(f1)
    plate = pd.read_excel(f2, index_col=0)
    order = ['Time', 'Temp'] # 'order' will become column names

    # 'fake_order' is used to differentiate replicates of same inoculum size
    fake_order = ['Time', 'Temp']
    count = 0
    for well in plate.itertuples():
        for entry in well[1:]:
            order.append(entry)
            # if entry is 'anerobic' or 'contam'
            if type(entry) == str:
                fake_order.append(entry + str(count))
                #print(entry + str(count))
            else:
                fake_order.append(float(entry + count*0.00001))

            count += 1

    # set column names to be inoculation dose size
    gc.columns = fake_order
    num_measurements = len(gc['Time'])

    # calculate time increment for OD measurements
    if type(gc['Time'][0]) == int or type(gc['Time'][0]) == np.int64:
        delta_t = gc['Time'][1] - gc['Time'][0]
    if type(gc['Time'][0]) == datetime.time:
        delta_t = int(str(gc['Time'][1])[3:5]) - int(str(gc['Time'][0])[3:5])
    gc['Time'] = [delta_t*t for t in range(num_measurements)]
    gc.columns = fake_order

    if if_plot: fig, ax = plt.subplots()

    # filter growth trajectories if initial condition is 'anerobic' or 'contam'
    for elem in gc:
        if elem == 'Time': continue
        if elem == 'Temp': continue
        if type(elem) == str:
            #print('omitting elem, anerobic/contam:', elem)
            del gc[elem]
            fake_order_idx = fake_order.index(elem)
            fake_order.pop(fake_order_idx)
            order.pop(fake_order_idx)

    if return_raw_data:
        gc.columns = order
        return gc

    # filter growth trajectories that are unnaturally high at an early time
    for inoculum in inocula:
        # used to parameter sweep consequences of the cutoff value of 0.125:
        #if gc[inoculum][int(60/delta_t)] > 0.13:
        if gc[inoculum][int(60/delta_t)] > 0.125:
            if verbose:
                print('omitting elem, high initial OD:', gc[inoculum][int(60/delta_t)])
            if if_plot:
                plt.plot(gc['Time'], gc[inoculum], color='r', lw=2, zorder=5)
            del gc[inoculum]
            fake_order_idx = fake_order.index(inoculum)
            fake_order.pop(fake_order_idx)
            order.pop(fake_order_idx)
        else:
            if if_plot: plt.plot(gc['Time'], gc[inoculum], color='k', lw=1)

    # subtract background from growth trajectories
    inocula = sorted([elem for elem in set(gc.columns) if type(elem) != str])
    background_od = np.mean([gc[inoculum][0] for inoculum in inocula])
    if verbose: print('background OD:', background_od)
    for inoculum in inocula:
        gc[inoculum] = gc[inoculum].apply(lambda x: x - background_od)

    if if_plot:
        fontsize=22
        plt.axhline(y=0.125, color='orange', lw=2, ls='--')
        ax.set_xlabel('time [min]', fontsize=fontsize)
        ax.set_ylabel('optical density (with background)', fontsize=fontsize)
        ax.set_title('Plate {} on {}'.format(plate_num, date), fontsize=fontsize)
        ax.set_yscale('log')
        ax.axis([0, max(gc['Time']), 0.005, 1.2])
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.savefig('figs/plate_trajectories_{}_{}.pdf'.format(date, plate_num), bbox_inches='tight')

    gc.columns = order
    return gc

##############################
# HELPER FUNCTIONS
##############################

def get_std_fpt(data_dict, threshold, verbose=False):
    """Calculates the standard deviation of the first passage time distribution
    at a threshold, for each set of replicate trajectories stored in data_dict.
    Inputs: data_dict ... dict, contains replicate trajectories
                      usage: data_dict[dataset][inoculum][replicate] = [ts, ys]
            threshold ... float, abundance threshold of first passage time
                          distribution
            verbose ... bool, indicates whether to print diagnostic text
    Outputs: inocula ... length N list, list of the inoculum sizes that were
                         experimentally tested. N is the number of inoculum
                         sizes.
             std_fpt ... dict, contains standard deviation of the first passage
                         time distributions for each inoculum size for each
                         experiment
                         Usage: std_fpt[exp][inoculum] = std of fpt dist """

    fpts = {}
    for exp in data_dict:
        gc = data_dict[exp]

        inocula = set([elem for elem in gc.columns if (type(elem) != str
                                                      and elem > 0)])
        fpts[exp] = {inoculum: [] for inoculum in inocula}
        for inoculum in inocula:
            try:
                num_columns = len(gc[inoculum].columns)
            except AttributeError:
                num_columns = 1

            for replicate in range(num_columns):
                if num_columns == 1:
                    trajectory = gc[inoculum]
                else:
                    trajectory = gc[inoculum].iloc[:, replicate]
                #print(trajectory)
                for i,measurement in enumerate(trajectory):
                    if i < 30: continue
                    if measurement > threshold:
                        fpts[exp][inoculum].append(gc['Time'][i]/60)
                        break
    return fpts

def get_growth_rates(data_dict, threshold, verbose=False, if_plot=False):
    """Calculates the growth rates of exponentially growing trajectories
    (stored in data_dict) at a threshold abundance. 
    Inputs: data_dict ... dict, contains replicate trajectories
                      usage: data_dict[dataset][inoculum][replicate] = [ts, ys]
            threshold ... float, abundance threshold of first passage time
                          distribution
            verbose ... bool, indicates whether to print diagnostic text
            if_plot ... boolean, specifies whether to plot histograms of growth
                        rates for each inoculum size of each experiment
    Outputs: growth_rates ... dict (same structure as data_dict), contains
                              growth rates for each replicate of each inoculum
                              of each experiment """

    growth_rates = {}

    for exp in data_dict:
        gc = data_dict[exp]
        delta_x = list(gc['Time'])[1] - list(gc['Time'])[0]
        #skip = int(6/delta_x) # "smooth" derivative over 30 minute interval
        xs = list(gc['Time'])

        inocula = sorted([elem for elem in set(gc.columns) if (type(elem) != str
                                                               and elem > 0)])
        growth_rates[exp] = {inoculum: [] for inoculum in inocula}

        for inoculum in inocula:
            ys = gc[inoculum]
            num_columns = len(gc[inoculum].columns)
            for replicate in range(num_columns):
                rep_traj = gc[inoculum].iloc[:,replicate]
                try:
                    threshold_idx = next(x[0] for x in enumerate(rep_traj) if
                                         (x[1] > threshold and x[0] >= 60))
                except StopIteration:
                    continue

                subset_ys = np.log(rep_traj[threshold_idx-7:threshold_idx+7])
                subset_xs = xs[threshold_idx-7:threshold_idx+7]
                slope, intercept, r, p, std_err = stats.linregress(subset_xs,
                                                                   subset_ys)

                #with warnings.catch_warnings():
                #    # don't print RuntimeWarning for log of negative numbers
                #    warnings.simplefilter('ignore')
                #    delta_ys = np.diff(np.log(gc[inoculum].iloc[:,replicate][::skip]))/(skip*delta_x)
                growth_rates[exp][inoculum].append(60*slope)

    for exp in growth_rates:
        for inoculum in growth_rates[exp]:
            growth_rates[exp][inoculum] = sorted(growth_rates[exp][inoculum])
            #print(exp, inoculum, np.round(growth_rates[exp][inoculum], 2))

            res = stats.bootstrap((growth_rates[exp][inoculum],), np.mean, confidence_level=0.95)
            ci_l, ci_u = res.confidence_interval
            #print(exp, inoculum, np.mean(growth_rates[exp][inoculum]), ci_l, ci_u)

    if if_plot:
        for exp in growth_rates:
            exp_growth_rates = np.concatenate([growth_rates[exp][elem] for elem in
                                           growth_rates[exp]])
            for inoculum in growth_rates[exp]:
                ax.hist(growth_rates[exp][inoculum],
                        bins=np.linspace(0, 3, 21),
                        histtype='step', label='{:.1f}'.format(inoculum),
                        density=True)
            ax.hist(exp_growth_rates,
                    bins=np.linspace(0, 3, 21), color='k', lw=3,
                    histtype='step', label='all inocula'.format(inoculum),
                    density=True)
            legend = ax.legend(title='mean inoculum size', fontsize=12)
            legend.get_title().set_fontsize('12')
            ax.set_xlabel(r'growth rate $\mu$ [1/hour]')
            ax.set_ylabel(r'probability density')
            ax.axis([0, 3, 0, None])
            plt.savefig('figs/growth_rates_exp_{}.pdf'.format(exp), bbox_inches='tight')

    return growth_rates

def get_sub_growth_rates(growth_rates, return_dist=False):
    growth_rates_a = []
    growth_rates_b = []
    growth_rates_c = []
    for exp in growth_rates:
        if exp in [1, 2, 3]:
            for inoc in growth_rates[exp]:
                for elem in growth_rates[exp][inoc]:
                    growth_rates_a.append(elem)
        if exp in [4]:
            for inoc in growth_rates[exp]:
                for elem in growth_rates[exp][inoc]:
                    growth_rates_b.append(elem)
        if exp in [5,6]:
            for inoc in growth_rates[exp]:
                for elem in growth_rates[exp][inoc]:
                    growth_rates_c.append(elem)
    mu_a = np.mean(growth_rates_a)
    mu_b = np.mean(growth_rates_b)
    mu_c = np.mean(growth_rates_c)

    if return_dist:
        return growth_rates_a, growth_rates_b, growth_rates_c

    return mu_a, mu_b, mu_c



##############################
# HELPER PLOTTING FUNCTIONS 
##############################

def lighten_color(color, amount=0.5):
    """ From Ian Hincks, https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])


##############################
# PLOTTING FUNCTIONS 
##############################

def plot_growth_trajectories(gc, date=None, plotted_inocula=None, ax=None,
                             fontsize=22, time_units='min', plot_std=False,
                             if_plot=False):
    """Plot a subset of bacterial growth trajectories.
    Inputs: gc ... pandas database, "(g)rowth (c)urves", contains growth
                   trajectories with column names corresponding to inoculum
                   sizes
            date ... str, date on which experiments were performed
            plotted_inocula ... list, which inocula from gc to plot
            ax ... axes, canvas for the plot
            fontsize ... float
            time_units ... str, 'min' or 'hr', sets time units for plotting
            plot_std ... bool, indicates whether to plot the standard deviation
                         across replicates for each inoculum size
            if_plot ... boolean, specifies whether to save standalone figure
    Outputs: ax ... axes, canvas for the plot """

    # all inocula provided in gc
    inocula = sorted([elem for elem in set(gc.columns) if (type(elem) != str and
                                                          elem > 0)])
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if not ax:
        fig, ax = plt.subplots()
    for i,inoculum in enumerate(inocula):
        if plotted_inocula:
            if inoculum not in plotted_inocula:
                continue
        num_replicates = len(gc[inoculum].columns)
        if time_units == 'hr':
            xs = [elem/60 for elem in gc['Time']]
        else:
            xs = [elem for elem in gc['Time']]
        for j,replicate in enumerate(range(num_replicates)):
            ys = gc[inoculum].iloc[:,replicate]
            ax.plot(xs, ys, lw=0.3, color=cycle[i], zorder=4)
        # phantom plot for legend:
        ax.plot([0, 0], [0, 0], color=cycle[i], lw=1, label=r'{} ({})'
                .format(inoculum, num_replicates), zorder=10)
        if plot_std:
            inoculum_average = gc[inoculum].mean(axis=1)
            inoculum_std = gc[inoculum].std(axis=1)
            ax.fill_between(xs,
                    [a - b for a,b, in zip(inoculum_average, inoculum_std)],
                    [a + b for a,b, in zip(inoculum_average, inoculum_std)],
                    color=lighten_color(cycle[i], amount=0.3), alpha=1, zorder=1)

    if time_units == 'hr':
        ax.set_xlabel('time [hr]', fontsize=fontsize)
        ax.axis([0, max(gc['Time'])/60, 0.005, 1.2])
    else:
        ax.set_xlabel('time [min]', fontsize=fontsize)
        ax.axis([0, max(gc['Time']), 0.005, 1.2])
    ax.set_ylabel('optical density', fontsize=fontsize)
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    legend = ax.legend(title=r'inoc. size (\# reps)',
                       fontsize=fontsize-4)
    legend.get_title().set_fontsize(str(fontsize-4))
    if date:
        ax.set_title('Experiment performed {}'.format(date), fontsize=fontsize)
    if if_plot:
        if date:
            plt.savefig('figs/plate_trajectories_no_background_{}.pdf'.format(date), bbox_inches='tight')
        else:
            plt.savefig('figs/plate_trajectories_no_background.pdf', bbox_inches='tight')

    return ax

def plot_std_fpt_vs_inoculum_normalized(fpts, growth_rates, verbose=False):
    """Plot the standard deviation of the first passage time distribution,
    in units of division time, across all experiments.
    Inputs: fpts ... dict, contains the first passage time distributions for
                     each replicate of each inoculum size for each experiment
                     Usage: std_fpt[exp][inoculum][replicate] = #
            growth_rates ... dict (same structure as data_dict), contains
                             growth rates for each replicate of each inoculum
                             of each experiment
    Outputs: ax ... axes, canvas for the plot of SD(FPT) vs inoculum size"""

    marker_list = ['s', '^', 'P', 'o', 'X', 'P', 'D', 'p']
    marker_list = ['s', '^', '^', '^', 'o', 'D', 'D', 'p']
    cycle = plt.cm.nipy_spectral(np.linspace(0,1,len(marker_list)+1))[::-1]
    cycle = [cycle[4], cycle[4], cycle[4],  cycle[4], cycle[4], cycle[4], cycle[6], cycle[6], cycle[6]]
    marker_size = [10, 8, 10, 8, 8, 9, 7, 7]
    marker_size = [10, 9, 10, 9, 9, 7, 7]
    mfc = ['white','white','white','white','white', cycle[6], cycle[6]]
    #labels = [r'bad {{\em E. coli}} growth (05-03-22)',
    #          r'{{\em E. coli}} growth, 37\textdegree C (05-19-22)',
    #          r'{{\em E. coli}} growth, 37\textdegree C (05-25-22)',
    #          r'{{\em E. coli}} growth, 37\textdegree C (06-01-22)',
    #          r'{{\em E. coli}} growth, 25\textdegree C (06-15-22)',
    #          r'{{\em S. aureus}} growth, 37\textdegree C (06-27-22)',
    #          r'{{\em S. aureus}} growth, 37\textdegree C (07-01-22)' ]
    labels = [r'bad {{\em E. coli}}',
              r'{{\em E. coli}}, 37\textdegree C',
              r'{{\em E. coli}}, 37\textdegree C',
              r'{{\em E. coli}}, 37\textdegree C',
              r'{{\em E. coli}}, 25\textdegree C',
              r'{{\em S. aureus}}, 37\textdegree C',
              r'{{\em S. aureus}}, 37\textdegree C' ]

    mu_a, mu_b, mu_c = get_sub_growth_rates(growth_rates)
    mus = [None, mu_a, mu_a, mu_a, mu_b, mu_c, mu_c]

    std_fpt = {exp: {} for exp in fpts}
    for exp in fpts:
        for inoculum in fpts[exp]:
            std_fpt[exp][inoculum] = np.std(fpts[exp][inoculum])
            if verbose: print(exp, inoculum, std_fpt[exp][inoculum])

    std_fpt_errs = {exp: {} for exp in fpts}
    inoculum_errs = {exp: {} for exp in fpts}
    for exp in fpts:
        for inoculum in fpts[exp]:
            res = stats.bootstrap((fpts[exp][inoculum],), np.std,
                                  confidence_level=0.68)
            ci_l, ci_u = res.confidence_interval
            mean_std = std_fpt[exp][inoculum]
            std_fpt_errs[exp][inoculum] = [ci_u - mean_std, mean_std - ci_l]

            inoculum_l, inoculum_u = stats.poisson.interval(0.68, inoculum)
            inoculum_errs[exp][inoculum] = (
                [(inoculum_u - inoculum)/np.sqrt(len(fpts[exp][inoculum])),
                 (inoculum - inoculum_l)/np.sqrt(len(fpts[exp][inoculum]))] )

    mean_growth_rates = {exp: np.mean(np.concatenate(
                         [growth_rates[exp][elem] for elem in
                          growth_rates[exp]])) for exp in fpts}

    fig, ax = plt.subplots(figsize=(4.8,4.8))
    for exp in fpts:
        mu = mus[exp]

        xs = sorted(list(std_fpt[exp].keys()))
        ys = [std_fpt[exp][inoculum] for inoculum in xs]
        ys = [elem*(mu/np.log(2)) for elem in ys]
        x_errs = np.array([inoculum_errs[exp][inoculum] for inoculum in xs]).T
        y_errs = np.array([std_fpt_errs[exp][inoculum] for inoculum in xs]).T
        y_errs = [elem*(mu/np.log(2)) for elem in y_errs]

        if exp in [1, 4, 5]:
            ax.errorbar(xs, ys, xerr=x_errs, yerr=y_errs,
                        fmt=marker_list[exp], color=cycle[exp+1],
                        markerfacecolor=mfc[exp], zorder=10,
                        markersize=marker_size[exp], elinewidth=1.2,
                        label=labels[exp], mew=1.5)
        else:
            ax.errorbar(xs, ys, xerr=x_errs, yerr=y_errs,
                        fmt=marker_list[exp], color=cycle[exp+1],
                        markerfacecolor=mfc[exp], zorder=10,
                        markersize=marker_size[exp], elinewidth=1.2,
                        mew=1.5)

        low_bound_xs = [xs[0]]
        low_bound_ys = [ys[0]]
        for k in range(len(xs)):
            if k == 0: continue
            if ys[k] < low_bound_ys[-1]:
                low_bound_xs.append(xs[k])
                low_bound_ys.append(ys[k])

        #ax.plot(low_bound_xs, low_bound_ys, '--', lw=1.0, color=cycle[exp+1])


    model_xs = list(range(1, 116))
    model_ys = [(1/np.log(2))*np.sqrt(sum([1/(n*n) for n in range(x,1000)])) for x in model_xs]
    ax.plot(model_xs, model_ys, color='r', lw=2, label='theory, simple birth process')

    ax.plot(np.linspace(50, 100, 2),
            [1.1 * x**(-1/2) for x in np.linspace(50, 100, 2)], color='k')
    ax.text(20, 0.09, r'$\sigma_t \sim n_0^{-1/2} $', color='k', fontsize=16)


    ax.set_xticks([1, 10, 10**2])
    ax.axis([1, 115, 0.08, 8])

    ax.set_xlabel(r'mean inoculum size $n_0$')
    ax.set_ylabel(r'temporal standard deviation $\sigma_t$ \\ \hspace*{3em} [\# division times]')

    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.plot(xs[:6], [40*x**(-1/2) for x in xs[:6]], color='r')
    plt.legend(fontsize=10, framealpha=1).set_zorder(13)


    std_fpt, inoculum_errs, std_fpt_errs, growth_rates = plot_each_std_fpt(fpts, growth_rates, if_plot=False)
    print(std_fpt[4])

    ax_inset = ax.inset_axes([1.9, 0.12, 10, 0.35], transform=ax.transData)
    ax_inset.set_xlabel(r'$n_0$', fontsize=12, labelpad=-10)
    ax_inset.set_ylabel(r'$\sigma_t$ [hr]', fontsize=12, labelpad=-12)

    #for exp in [1, 2, 3, 4, 5, 6]:
    for exp in [3]:
        xs = list(std_fpt[exp].keys())
        xs.sort()
        ys = [std_fpt[exp][x] for x in xs]
        x_errs = np.array([inoculum_errs[exp][x] for x in xs]).T
        y_errs = np.array([std_fpt_errs[exp][x] for x in xs]).T

        ax_inset.errorbar(xs, ys, xerr=x_errs, yerr=y_errs,
                    fmt=marker_list[exp], color=cycle[exp+1],
                    markerfacecolor=mfc[exp], zorder=10,
                    markersize=marker_size[exp]-1, elinewidth=1.2,
                    mew=1.5)
        ax_inset.set_xscale('log')
        ax_inset.set_yscale('log')
        ax_inset.axis([1, 120, 0.1, 1.5])

        exp_growth_rates = np.concatenate([growth_rates[exp][inoc] for inoc in
                                       growth_rates[exp]])
        mu = np.mean(exp_growth_rates)
        model_xs = list(range(1, 120))
        model_ys = [(1/mu)*np.sqrt(sum([1/(n*n) for n in range(x,1000)])) for x in model_xs]
        ax_inset.plot(model_xs, model_ys, color='r', lw=2)

    ax_inset.set_xticks([1, 100])
    ax_inset.tick_params(axis='both', which='major', labelsize=12)

    plt.savefig('figs/Fig4.pdf', bbox_inches='tight')
    return

def plot_each_std_fpt(fpts, growth_rates, verbose=False, if_plot=True):
    """Plot the standard deviation of the first passage time distribution for
    each experiment. Each experiment typically consists of six inoculum sizes.
    Inputs: fpts ... dict, contains the first passage time distributions for
                     each replicate of each inoculum size for each experiment
                     Usage: std_fpt[exp][inoculum][replicate] = #
            growth_rates ... dict (same structure as data_dict), contains
                             growth rates for each replicate of each inoculum
                             of each experiment
    Outputs: ax ... axes, canvas for the plot of SD(FPT) vs inoculum size"""

    std_fpt = {exp: {} for exp in fpts}
    for exp in fpts:
        for inoculum in fpts[exp]:
            std_fpt[exp][inoculum] = np.std(fpts[exp][inoculum])
            if verbose: print(exp, inoculum, std_fpt[exp][inoculum])

    std_fpt_errs = {exp: {} for exp in fpts}
    inoculum_errs = {exp: {} for exp in fpts}
    for exp in fpts:
        for inoculum in fpts[exp]:
            res = stats.bootstrap((fpts[exp][inoculum],), np.std,
                                  confidence_level=0.68)
            ci_l, ci_u = res.confidence_interval
            mean_std = std_fpt[exp][inoculum]
            #if np.round((ci_u - ci_l)/mean_std, 2) > 1:
                #print(fpts[exp][inoculum])
                #print(np.round((ci_u - ci_l)/mean_std, 2))
                #print()
            std_fpt_errs[exp][inoculum] = [ci_u - mean_std, mean_std - ci_l]

            inoculum_l, inoculum_u = stats.poisson.interval(0.68, inoculum)
            inoculum_errs[exp][inoculum] = (
                [(inoculum_u - inoculum)/np.sqrt(len(fpts[exp][inoculum])),
                 (inoculum - inoculum_l)/np.sqrt(len(fpts[exp][inoculum]))] )

    if not if_plot:
        return std_fpt, inoculum_errs, std_fpt_errs, growth_rates

    labels = [r'bad {{\em E. coli}}, (05-03-22)',
              r'{{\em E. coli}}, 37\textdegree C',
              r'{{\em E. coli}}, 37\textdegree C',
              r'{{\em E. coli}}, 37\textdegree C',
              r'{{\em E. coli}}, 25\textdegree C',
              r'{{\em S. aureus}}, 37\textdegree C',
              r'{{\em S. aureus}}, 37\textdegree C' ]

    for exp in fpts:
        xs = sorted(list(std_fpt[exp].keys()))
        ys = [std_fpt[exp][inoculum] for inoculum in xs]
        x_errs = np.array([inoculum_errs[exp][inoculum] for inoculum in xs]).T
        y_errs = np.array([std_fpt_errs[exp][inoculum] for inoculum in xs]).T

        if if_plot:
            fig, ax = plt.subplots(figsize=(4.8,4.8))

            ax.errorbar(xs, ys, xerr=x_errs, yerr=y_errs, fmt='s', color='k',
                        markerfacecolor='white', zorder=11, label=r'{} ($\pm 1 \sigma$)'
                        .format(labels[exp]), mew=2)

        exp_growth_rates = np.concatenate([growth_rates[exp][elem] for elem in
                                       growth_rates[exp]])
        mu = np.mean(exp_growth_rates)
        if if_plot:
            ax.text(1.2, 0.1, r'$\mu = $ {:.2f}/hr'.format(mu), color='k')
        # runs slow sometimes:
        #res = stats.bootstrap((exp_growth_rates,), np.mean, confidence_level=0.95)
        #ci_l, ci_u = res.confidence_interval

        model_xs = list(range(1, 116))
        model_ys = [(1/mu)*np.sqrt(sum([1/(n*n) for n in range(x,1000)])) for x in model_xs]
        if if_plot:
            ax.plot(model_xs, model_ys, color='r', lw=2, label='SBP model')

            ax.set_xticks([1, 10, 10**2])
            ax.axis([1, 115, 0.08, 4])

            ax.set_xlabel('mean inoculum size')
            ax.set_ylabel(r'temporal standard \\ \hspace*{1em} deviation [hours]')
                          #$\sqrt{(\langle t^2 \rangle - \langle t \rangle^2)_{n \, | \, n_0}}$

            ax.set_xscale('log')
            ax.set_yscale('log')
            #ax.plot(xs[:6], [40*x**(-1/2) for x in xs[:6]], color='r')
            plt.legend(fontsize=8)

            plt.savefig('figs/std_fpt_vs_inoculum_exp_{}.pdf'.format(exp), bbox_inches='tight')

def plot_distribution_of_initial_ODs():
    # to use this function, make sure that get_growth_curves returns the growth
    # curves
    exp_data_dict = load_exp_data_e_coli(load_data=True, if_plot=False,
                                         return_raw_data=True)

    ics = {t: [] for t in [0, 6, 12, 30, 60, 120, 240]}
    for t in ics:
        for exp in exp_data_dict:
            gc = exp_data_dict[exp]
            delta_t = gc['Time'][1] - gc['Time'][0]
            t_index = int(t/delta_t)
            inocula = sorted([elem for elem in set(gc.columns) if (type(elem) != str and
                                                                  elem > 0)])
            for inoc in inocula:
                num_replicates = len(gc[inoc].columns)
                for replicate in range(num_replicates):
                    ys = gc[inoc].iloc[:,replicate]
                    ics[t].append(ys[t_index])
        fig, ax = plt.subplots()

        threshold = 0.125
        print(min(ics[t]))
        print(max(ics[t]))
        ics_white = [elem for elem in ics[t] if elem < threshold]
        ics_red = [elem for elem in ics[t] if elem >= threshold]
        ax.hist(ics_white, bins=np.linspace(0.08, 0.46, 77), histtype='step')
        ax.hist(ics_red, bins=np.linspace(0.08, 0.46, 77),
                color='red')

        ax.axvline(x=threshold, color='r', lw=2)

        ax.set_xlabel(r'OD at t={} min'.format(t))
        ax.set_ylabel(r'frequency')
        ax.axis([0.08, 0.46, 0, 500])
        a = len(ics_red)
        b = len(ics_white) + len(ics_red)
        ax.text(0.2, 30, '{}\% ({}/{}) of trajectories omitted'
                .format(int(np.round(100*a/b)), a, b), fontsize=12)
        plt.savefig('figs/initial_OD_t_{}.pdf'.format(t), bbox_inches='tight')


############################################################
# SCRIPTS FOR GENERATING MAIN TEXT FIGURES
############################################################

def generate_fig_1():
    """Generates Figure 1 of the main text. 100 representative growth
       trajectories for the contaminant L. Monocytogenes, which has a division
       time of roughly 17 hours, with inoculum sizes of 1 and 100. This
       bacteria is assumed to grow in 1 L of milk. """

    mu = 24*np.log(2)/17 # units of 1/day
    mu = np.round(mu, 3)

    # For milk, shelf life threshold is dictated by the Pasteurized Milk
    # Ordinance, set to 20000 CFU/mL. Assuming bacteria grow in 1 L of milk:
    shelf_life_threshold = 2e7

    num_trajs = 100

    colors = ['black', 'grey']
    lss = ['-', '-']

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax_right = ax.twinx() # ax: abundance; ax_right: concentration

    for i,ic in enumerate([1, 100]):
        p = SimulateSBP(ic=ic, t_max=23, num_trajs=num_trajs, mu=mu)
        ts, trajs = p.get_SBP_trajs(load_data=True)

        # only plot 10 trajs
        for k,traj in enumerate(trajs[:5]):
            if i == 0:
                ax.plot(ts, traj, lw=0.5, color=colors[i], ls=lss[i], alpha=1,
                        zorder=2)
            if i == 1:
                ax.plot(ts, traj, lw=0.5, color=colors[i], ls=lss[i], alpha=1,
                        zorder=2)

        # calculate first passage times to shelf life threshold
        shelf_life_FPTs = []
        for k,traj in enumerate(trajs[:10]):
            shelf_life_FPTs.append(ts[np.argmax(traj >= shelf_life_threshold)])
            # if multiple maxima, np.argmax returns the first max value

        FPT_min = min(shelf_life_FPTs)
        FPT_max = max(shelf_life_FPTs)

        ax.plot([FPT_min, FPT_min],
                [0, shelf_life_threshold], color=colors[i], ls='-', lw=1.5)
        ax.plot([FPT_max, FPT_max],
                [0, shelf_life_threshold], color=colors[i], ls='-', lw=1.5)

        # decorate plot with "X day spread" text
        text_placements = [16.1, 7.9]
        fontsize=14
        if i == 0:
            ax.text(text_placements[i] + 1.2, 2*(6/1.5), r'${}$'.format(
                    np.round(FPT_max - FPT_min, 1)), color=colors[i],
                    fontsize=fontsize)
            ax.text(text_placements[i] + 0.8, 2, r'days'.format(
                    np.round(FPT_max - FPT_min, 1)), color=colors[i],
                    fontsize=fontsize)
        else:
            ax.text(text_placements[i], 8, r'${}$ days'.format(
                    np.round(FPT_max - FPT_min, 1)), color=colors[i],
                    fontsize=fontsize)
        #ax.text(text_placements[i] + 0.1, 1.5, 'spread', color=colors[i],
        #        fontsize=fontsize)

    # decorate plot with "reported shelf life variability" bar and text
    ax_right.plot([10, 10], [1.5e8, 6e8], ls='-', color='blue', lw=2)
    ax_right.plot([21, 21], [1.5e8, 6e8], ls='-', color='blue', lw=2)
    ax_right.plot([10, 21], [np.sqrt(1.5e8 * 6e8), np.sqrt(1.5e8 * 6e8)], ls='-', color='blue', lw=2)
    t = ax_right.text(10.6, 3.5e7, 'shelf-life variability', fontsize=14,
            color='blue', zorder=0)
    t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white', zorder=1))
    #t = ax_right.text(12, 6e6, 'variability: 9 days', fontsize=14,
    #        color='blue', zorder=0)
    #t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white', zorder=1))

    ax.set_xlabel('days after pasteurization')
    ax.set_ylabel(r'\# bacteria in 1L milk')
    ax.set_yscale('log')
    ax.set_xticks([0, 7, 14, 21])
    ax.set_yticks([1, 100, 1e4, 1e6, 1e8, 1e10, 1e12])
    ax.axis([0, 23, .7, 1e12])

    ax_right.set_yscale('log')
    #ax_right.set_ylabel(r'concentration [CFU/mL]')
    ax_right.set_yticks([])
    ax_right.set_yticklabels([])
    ax_right.axis([0, 23, .7/1000, 1e9])

    ax_right.axhline(y=20000, color='blue', lw=1.5)
    ax_right.text(1, 4500, 'shelf-life threshold', color='blue', fontsize=14)
    ax_right.axhline(y=1e6, color='red', lw=1.5)
    ax_right.text(1, 2.5e5, 'spoiled milk', color='red', fontsize=14)

    plt.savefig('figs/Fig1.pdf', bbox_inches='tight')

def generate_fig_3():
    """Generates Figure 2 of the main text. Plots
    replicate growth trajectories for E. coli (panel a), S. aureus (b),
    duckweed (c), along with the temporal variation as a function of
    threshold (panel d). """

    threshold = 0.03
    exp_data_dict = load_exp_data_e_coli(load_data=True, if_plot=False)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(ncols=2, nrows=2, figsize=(6, 4.8))

    labels = [r'bad {{\em E. coli}}',
              r'{{\em E. coli}}, 37\textdegree C',
              r'{{\em E. coli}}, 37\textdegree C',
              r'\hspace*{-2em} {\em E. coli}, 37$^{\circ}$C \\ inoc.\ size (\# reps)',
              r'{{\em E. coli}}, 25\textdegree C',
              r'{{\em S. aureus}}, 37\textdegree C',
              r'\hspace*{-2.5em} {{\em S. aureus}}, 37\textdegree C \\ inoc.\ size (\# reps)' ]

    for j,(gc,inocs,ax,title) in enumerate(
            [[exp_data_dict[3], [1.8, 16.1, 80.7], ax0, labels[3]],
             [exp_data_dict[6], [1.6, 4.2, 30.0], ax1, labels[6]]]):
        #inocula = sorted([elem for elem in set(gc.columns) if (type(elem) != str and
        #                                                      elem > 0)])
        #print(exp, inocula)
        ax = plot_growth_trajectories(gc, plotted_inocula=inocs, ax=ax,
                                      fontsize=12, time_units='hr',
                                      plot_std=False, if_plot=False)
        ax.axis([0, 18, 0.01, 1.2])
        legend = ax.legend(title=title, fontsize=6, loc='upper left')
        legend.get_title().set_fontsize('6')

        inocula = sorted([elem for elem in set(gc.columns) if (type(elem) != str and
                                                               elem > 0)])
        #cycle = plt.cm.nipy_spectral(np.linspace(0,1,len(inocula)+1))[::-1]
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        threshold_xs = np.logspace(np.log10(0.005), 0, 51)

        letters = ['a)', 'b)']
        ax.text(-4.5, 1, letters[j], fontsize=14, fontweight='bold')


        for i,inoc in enumerate(inocula):
            if inoc not in inocs: continue
            threshold_ys = []
            for x in threshold_xs:
                print(x)
                fake_data_dict = {0: gc}
                fpts_dict = get_std_fpt(fake_data_dict, x)
                fpts = fpts_dict[0]
                threshold_ys.append(np.std(fpts[inoc]))

            threshold_labels = [r'{{\em E. coli}} ({})'.format(inoc),
                      r'{{\em S. aureus}} ({})'.format(inoc)]
            threshold_lss = ['-', '--']
            ax3.plot(threshold_xs, threshold_ys, lw=2, color=cycle[i],
                     label=threshold_labels[j], ls=threshold_lss[j])

    fontsize = 12
    ax3.tick_params(axis='both', which='major', labelsize=fontsize)
    ax3.set_xscale('log')
    ax3.set_xlabel('threshold optical density', fontsize=fontsize)
    ax3.set_ylabel(r'temporal standard \\ \hspace*{2em} deviation [hr]',
                  fontsize=fontsize)
    legend = ax3.legend(title='organism (inoculum size)', fontsize=6,
                        loc='upper center', ncol=2)
    legend.get_title().set_fontsize('6')
    ax3.axis([0.01, 0.5, 0, 4])

    ax3.text(0.0028, 4, 'd)', fontsize=14, fontweight='bold')


    growth_rates = get_growth_rates(exp_data_dict, threshold=threshold)

    for exp in growth_rates:
        if (exp != 3) and (exp != 6): continue
        exp_growth_rates = np.concatenate([growth_rates[exp][elem] for elem in
                                       growth_rates[exp]])
        if exp == 3:
            label = r'{{\em E. coli}}, 37\textdegree C'
            color = 'k'
            ls = '-'
        if exp == 6:
            label = r'{{\em S. aureus}}, 37\textdegree C'
            color = 'red'
            ls = (0, (5, 1))
        ax2.hist(exp_growth_rates,
                bins=np.linspace(0, 3, 16), ls=ls, color=color, lw=2,
                histtype='step', label=label)
        legend = ax2.legend(fontsize=6, loc='upper left')
        #legend.get_title().set_fontsize('6')
        print('a', np.mean(np.sort(exp_growth_rates)))
    ax2.set_xlabel(r'per-capita growth rate $\mu$ [1/hr]', fontsize=fontsize)
    ax2.set_ylabel(r'frequency', fontsize=fontsize)
    ax2.axis([0, 3, 0, 80])

    ax2.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.text(-.96, 80, 'c)', fontsize=14, fontweight='bold')


    plt.tight_layout(w_pad=0, h_pad=0)
    plt.savefig('figs/Fig3.pdf', bbox_inches='tight')


    #fpts = get_std_fpt(exp_data_dict, threshold=threshold)
    #growth_rates = get_growth_rates(exp_data_dict, threshold=threshold,
    #                                if_plot=False)

    #e_coli_inocula, std_fpt_e_coli, e_coli_labels = get_experimental_temporal_variability('e_coli')

def generate_fig_4():
    """Empirically calculates the standard deviation of the first passage time
    distribution at a particular abundance threshold for three experimental
    systems.
    Inputs: experiment ... 'e_coli', 's_aureus', or 'duckweed' selects the
                           experiment
    Outputs: inocula ... length N list, list of the inoculum sizes that were
                         experimentally tested. N is the number of inoculum
                         sizes.
             fpts ... dict, contains the first passage time distributions for
                      each replicate of each inoculum size for each experiment
                      Usage: std_fpt[exp][inoculum][replicate] = # """
    # E coli
    threshold = 0.03
    exp_data_dict = load_exp_data_e_coli(load_data=True, if_plot=False)

    fpts = get_std_fpt(exp_data_dict, threshold=threshold)

    growth_rates = get_growth_rates(exp_data_dict, threshold=threshold,
                                    if_plot=False)
    plot_each_std_fpt(fpts, growth_rates, if_plot=False)
    plot_std_fpt_vs_inoculum_normalized(fpts, growth_rates)

    return 1,2,3

if __name__ == "__main__":
    generate_fig_1()
    generate_fig_3()
    generate_fig_4()
