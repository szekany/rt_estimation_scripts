# import os
# import requests
# import pymc3 as pm
import pandas as pd
import numpy as np
# import theano
# import theano.tensor as tt
import matplotlib

from matplotlib import pyplot as plt
# from matplotlib import dates as mdates
# from matplotlib import ticker

from datetime import date
from datetime import datetime


def main():

    results = pd.read_csv('data/rt_2020_06_13.csv')

    uncert_low = results['lower_90']
    uncert_high = results['upper_90']

    print(uncert_low)

    ax = results.plot.scatter(
        title='Estimated Reproductive Rate - Washtenaw County, MI',
        x='date',
        y='mean',
        alpha=.8,
        lw=1,
        s=10,
        figsize=(10,6))
	

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_ylim(0, 2)
    ax.margins(x=0)

    ax.axhline(y=1, color='black', linestyle='-')
    # ax.axvline(x='2020-04-10', color='black', linestyle='-')
    # ax.text(10.1,0,'blah',rotation=0)

    # ax.set_xlim('4/15/2020', '6/13/2020')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=70)

    plt.tight_layout()
    ax.set_ylabel("Estimated Reproductive Rate (R_t)", fontweight="bold")
    ax.set_xlabel("Date", fontweight="bold")

    ax.fill_between(results['date'], uncert_low, uncert_high, color='g', alpha=0.2)


    # ax = results.plot.scatter(
    #     title='Onset vs. Confirmed Dates - COVID19',
    #     x='date',
    #     y='mean',
    #     alpha=.1,
    #     lw=0,
    #     s=10,
    #     figsize=(6,6))
    # plt.show()

    plt.savefig('6-14-20.png')

if __name__ == "__main__":
	main()