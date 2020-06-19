import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import requests
import pymc3 as pm
import pandas as pd
import numpy as np
import theano
import theano.tensor as tt

from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker

from datetime import date
from datetime import datetime

def load_mi():
	file = 'wa_cases.csv'
	states = pd.read_csv(file,
	                     parse_dates=['date'],
	                     index_col=['state', 'date']).sort_index()
	return states


def load_state():
	url = 'https://covidtracking.com/api/v1/states/daily.csv'
	states = pd.read_csv(url,
	                     parse_dates=['date'],
	                     index_col=['state', 'date']).sort_index()

	# Note: GU/AS/VI do not have enough data for this model to run
	# Note: PR had -384 change recently in total count so unable to model
	states = states.drop(['MP', 'GU', 'AS', 'PR', 'VI'])

	# Errors in Covidtracking.com
	states.loc[('WA','2020-04-21'), 'positive'] = 12512
	states.loc[('WA','2020-04-22'), 'positive'] = 12753 
	states.loc[('WA','2020-04-23'), 'positive'] = 12753 + 190
	states.loc[('VA', '2020-04-22'), 'positive'] = 10266
	states.loc[('VA', '2020-04-23'), 'positive'] = 10988
	states.loc[('PA', '2020-04-22'), 'positive'] = 35684
	states.loc[('PA', '2020-04-23'), 'positive'] = 37053
	states.loc[('MA', '2020-04-20'), 'positive'] = 39643
	states.loc[('CT', '2020-04-18'), 'positive'] = 17550
	states.loc[('CT', '2020-04-19'), 'positive'] = 17962
	states.loc[('HI', '2020-04-22'), 'positive'] = 586
	states.loc[('RI', '2020-03-07'), 'positive'] = 3

	# Make sure that all the states have current data
	today = datetime.combine(date.today(), datetime.min.time())
	last_updated = states.reset_index('date').groupby('state')['date'].max()
	is_current = last_updated < today

	try:
	    assert is_current.sum() == 0
	except AssertionError:
	    print("Not all states have updated")
	    print(last_updated[is_current])

	# Ensure all case diffs are greater than zero
	for state, grp in states.groupby('state'):
	    new_cases = grp.positive.diff().dropna()
	    is_positive = new_cases.ge(0)
	    
	    try:
	        assert is_positive.all()
	    except AssertionError:
	        print(f"Warning: {state} has date with negative case counts")
	        print(new_cases[~is_positive])
	        
	# Let's make sure that states have added cases
	idx = pd.IndexSlice
	assert not states.loc[idx[:, '2020-04-22':'2020-04-23'], 'positive'].groupby('state').diff().dropna().eq(0).any()

	return states
