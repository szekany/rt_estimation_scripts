# For some reason Theano is unhappy when I run the GP, need to disable future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import requests
import pymc3 as pm
import pandas as pd
import numpy as np
import theano
import theano.tensor as tt
import matplotlib

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker

from datetime import date
from datetime import datetime
from load_state import load_state, load_mi
from get_pt_data import get_pt_data
# from IPython.display import clear_output
from MCMCmodel import MCMCModel
# %config InlineBackend.figure_format = 'retina'


def confirmed_to_onset(confirmed, p_delay):

    assert not confirmed.isna().any()
    
    # Reverse cases so that we convolve into the past
    convolved = np.convolve(confirmed[::-1].values, p_delay)

    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1],
                       periods=len(convolved))

    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr)
    
    return onset

def adjust_onset_for_right_censorship(onset, p_delay):
    cumulative_p_delay = p_delay.cumsum()
    
    # Calculate the additional ones needed so shapes match
    ones_needed = len(onset) - len(cumulative_p_delay)
    padding_shape = (0, ones_needed)
    
    # Add ones and flip back
    cumulative_p_delay = np.pad(
        cumulative_p_delay,
        padding_shape,
        constant_values=1)
    cumulative_p_delay = np.flip(cumulative_p_delay)
    
    # Adjusts observed onset values to expected terminal onset values
    adjusted = onset / cumulative_p_delay
    
    return adjusted, cumulative_p_delay


def df_from_model(model):
    
    r_t = model.trace['r_t']
    mean = np.mean(r_t, axis=0)
    median = np.median(r_t, axis=0)
    hpd_90 = pm.stats.hpd(r_t, hdi_prob=.9)
    hpd_50 = pm.stats.hpd(r_t, hdi_prob=.5)
    
    print('model.trace_index:')
    print(model.trace_index)
    print('model.region:')
    print(model.region)

    # model.region.to_csv('model.csv')

    # model = pd.read_csv('model.csv')

    d = pd.to_datetime('2020-03-20')
    mask = (pd.to_datetime(model.region.index) >= d)
    model.region = model.region.loc[mask]
    print(model.region)

    # idx = pd.MultiIndex.from_product([
    #         [model.region],
    #         model.trace_index
    #     ], names=['region', 'date'])
        
    df = pd.DataFrame(data=np.c_[mean, median, hpd_90, hpd_50], index=model.trace_index,
                 columns=['mean', 'median', 'lower_90', 'upper_90', 'lower_50','upper_50'])

    print(df)

    return df

def create_and_run_model(name, state, p_delay):
    confirmed = state.positive.diff().dropna()
    print(confirmed)
    onset = confirmed_to_onset(confirmed, p_delay)
    adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    model = MCMCModel(name, onset, cumulative_p_delay)
    return model.run()

def main():

    matplotlib.use("TkAgg")
    show_plots = False

    # states = load_state()
    states = load_mi()

    # get_pt_data()

    # Load the patient CSV
    mylist = []
    for chunk in pd.read_csv(
        'data/linelist.csv',
        parse_dates=False,
        usecols=[
            'date_confirmation',
            'date_onset_symptoms'],
        low_memory=False,
        chunksize=10000):
        
        mylist.append(chunk)

    patients = pd.concat(mylist, axis=0)
    del mylist


    patients.columns = ['Onset', 'Confirmed']

    # There's an errant reversed date
    patients = patients.replace('01.31.2020', '31.01.2020')

    # Only keep if both values are present
    patients = patients.dropna()

    # Must have strings that look like individual dates
    # "2020.03.09" is 10 chars long
    is_ten_char = lambda x: x.str.len().eq(10)
    patients = patients[is_ten_char(patients.Confirmed) & 
                        is_ten_char(patients.Onset)]


    # Convert both to datetimes
    patients.Confirmed = pd.to_datetime(
        patients.Confirmed, errors='coerce', format='%d.%m.%Y')
    patients.Onset = pd.to_datetime(
        patients.Onset, errors='coerce', format='%d.%m.%Y')


    # Only keep records where confirmed > onset
    patients = patients[patients.Confirmed >= patients.Onset]

    ax = patients.plot.scatter(
        title='Onset vs. Confirmed Dates - COVID19',
        x='Onset',
        y='Confirmed',
        alpha=.1,
        lw=0,
        s=10,
        figsize=(6,6))

    formatter = mdates.DateFormatter('%m/%d')
    locator = mdates.WeekdayLocator(interval=2)

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(formatter)
        axis.set_major_locator(locator)

    if show_plots == True:
        plt.show()

    # Calculate the delta in days between onset and confirmation
    delay = (patients.Confirmed - patients.Onset).dt.days

    # Convert samples to an empirical distribution
    p_delay = delay.value_counts().sort_index()
    new_range = np.arange(0, p_delay.index.max()+1)
    p_delay = p_delay.reindex(new_range, fill_value=0)
    p_delay /= p_delay.sum()

    # Show our work
    fig, axes = plt.subplots(ncols=2, figsize=(9,3))
    p_delay.plot(title='P(Delay)', ax=axes[0])
    p_delay.cumsum().plot(title='P(Delay <= x)', ax=axes[1])
    for ax in axes:
        ax.set_xlabel('days')

    if show_plots == True:
        plt.show()

    if show_plots == True:

        state = 'CA'
        confirmed = states.xs(state).positive.diff().dropna()
        confirmed.tail()


        onset = confirmed_to_onset(confirmed, p_delay)

        adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)

        fig, ax = plt.subplots(figsize=(5,3))

        confirmed.plot(
            ax=ax,
            label='Confirmed',
            title=state,
            c='k',
            alpha=.25,
            lw=1)

        onset.plot(
            ax=ax,
            label='Onset',
            c='k',
            lw=1)

        adjusted.plot(
            ax=ax,
            label='Adjusted Onset',
            c='k',
            linestyle='--',
            lw=1)

        ax.legend();

        plt.show()

    models = {}

    for state, grp in states.groupby('state'):
        
        print(state)
        
        # if state in models:
        #     print(f'Skipping {state}, already in cache')
        #     continue
        if state == 'MI':
        
            models[state] = create_and_run_model(grp.droplevel(0), states.xs(state), p_delay)

    # Check to see if there were divergences
    n_diverging = lambda x: x.trace['diverging'].nonzero()[0].size
    divergences = pd.Series([n_diverging(m) for m in models.values()], index=models.keys())
    has_divergences = divergences.gt(0)

    print('Diverging states:')
    print(divergences[has_divergences])

    # Rerun states with divergences
    for state, n_divergences in divergences[has_divergences].items():
        models[state].run()

    results = None

    for state, model in models.items():

        df = df_from_model(model)

        if results is None:
            results = df
        else:
            results = pd.concat([results, df], axis=0)

    results.to_csv('data/rt_2020_06_13.csv')

if __name__ == "__main__":
    main()