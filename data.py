"""
Author: Alex (Tai-Jung) Chen

Data preparation codes.
"""
import pandas as pd


def nij():
    file_name = "datasets/Raw_data/Global_MI_multi_cat_2.pkl"
    df = pd.read_pickle(file_name)
    print()


def spam():
    pass


def hmeq():
    file_name = "datasets/Raw_data/hmeq.csv"
    df = pd.read_csv(file_name)
    print()


def covid_pre():
    file_name = "datasets/DF.csv"
    df = pd.read_csv(file_name)
    df.drop(columns="ID", inplace=True)
    high_risk_cond = df['Risk level'] > 2
    low_risk_cond = df['Risk level'] <= 2

    df.loc[high_risk_cond, 'Risk level'] = 1
    df.loc[low_risk_cond, 'Risk level'] = 0
    print()

    # save to csv
    # df.to_csv('covid.csv', index=False)

    # turn into pickles
    # df['Risk level'].to_pickle("covid_Y.pkl")
    # df.drop(columns='Risk level').to_pickle("covid_X.pkl")


if __name__ == '__main__':
    nij()
    # hmeq()
