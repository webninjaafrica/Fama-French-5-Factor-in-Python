
###############################
# Fama French 5-Factor Model (2X3)-2021
#improved by Kelvin Magochi:
#website: https://wwww.webninjaafrica.com
#################################

import matplotlib
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import time
import functools
import seaborn as sns
from typing import Dict, Any, List, Tuple
import pandas as pd
import math
import os
from collections import defaultdict
from io import StringIO



class ff5_factor
    def __init__(self,self,file: str) -> Dict[str, pd.DataFrame]:
            self.lkma= {'1D': 1, '1M': 20, '1Y': 250, '3Y': 250 * 3, '5Y': 250 * 5, '10Y': 250 * 10}
            self.lkfa = {'Market Factor (MER)': 'MER',
                      'Size Factor (SMB)': 'SMB',
                      'Value Factor (HML)': 'HML',
                      'Profitability Factor (RMW)': 'RMW',
                      'Investment Factor (CMA)': 'CMA'}
            self.f_max_width = 22
            self.datagrid=file

    def initialize_datagrids():
            
            with open(self.datagrid) as fh:
                name = None
                data = defaultdict(list)
                for line in fh.readlines():
                    if line.startswith("#data-begin#"):
                        name = line[line.rfind("#") + 1:].rstrip('\n')
                    elif line.startswith("#data-end#"):
                        name = None
                    elif name is not None:
                        data[name].append(line.rstrip('\n'))
                d2 = dict()
                for k, v in data.items():
                    p = pd.read_csv(StringIO('\n'.join(data[k])), index_col=0, parse_dates=[0], low_memory=False)
                    p = p.replace(-99.99, math.nan).replace(999) / 100
                    d2[k] = p
            return d2

    def load_portfolios(self):
            portfolios = dict()
            i = 1
            for file in os.listdir('data-portfolios'):
                fileport =self,initialize_datagrids(os.path.join('data-portfolios', file))
                for k2, v2 in fileport.items():
                    for k3, v3 in v2.items():
                        if 'ignore' not in k2.lower() and 'value' in k2.lower():
                            portfolios['%i. ' % i + '=>'.join([file[:-4], k2, k3])] = v3
                            i += 1
            for file in os.listdir('data-stocks'):
                x = pd.read_csv(os.path.join('data-stocks', file), index_col=0, parse_dates=[0], low_memory=False)['Close']
                x = (x - x.shift(1)) / x.shift(1)
                portfolios['%i. ' % i + file[:-4]] = x

            return portfolios

    def colorify(self,array, cmap_name, alpha=None) -> List:
    cm = plt.cm.get_cmap(cmap_name)
    p = list()
    for item, position in zip(array, np.linspace(0, 1, len(array))):
        c = cm(position)
        c = tuple([*c[0:3], alpha or 1.0])
        p.append((item, c))
    return p


    def eliminate_outliears(self,series: pd.Series, zscore: float) -> pd.Series:
        return series[abs(sp.stats.zscore(series)) < zscore]


    def roll_ma(self,data: pd.DataFrame, ma: str) -> pd.DataFrame:
        return data.rolling(window=self.lkma[ma]).mean().dropna()


    def inversify_factor(self,factor):
        ret=[k for k, v in self.lkfa.items() if v == factor][0]
        return ret

    def draw_factor_canvas(self,ff: str, ma: str) -> None:
        f: str = self.lkfa[ff]
        ###p=(1960, 1980), (1980, 2007), (2007, 2020)
        periods = []
        plt.figure(figsize=(9, 5))
        plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f bp'))
        for s, e in periods:
            df_factor: pd.DataFrame = df_factors[f]
            df = df_factor[str(s):str(e)]
            df = df.rolling(window=self.lkma[ma]).mean().dropna()
            x =self.eliminate_outliears(df, 5) * 1e4
            sns.distplot(x, hist=True, rug=False, label="%s-%s" % (s, e), bins=30)
            plt.axvline(0.0, color='black', linestyle='dotted')
            plt.title(f"{ff}\n{ma} rolling average")
        plt.legend()


    def plot_factor_timeseries(self,ff: str, ma: str) -> None:
        f: str = self.lkfa[ff]
        df =self.roll_ma(df_factors[f], ma)
        plt.figure(figsize=(9, 5))
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f bp'))
        color =self.colorify(factors_list, 'tab10')[factors_list.index(f)][1]
        plt.plot(df * 1e4, color=color, linewidth=1)
        plt.axhline(0.0, color='black', linestyle='dotted')
        plt.title(f"{ff}\n{ma} rolling the average")


    def fit_model(self,factor_names: List[str], ret_actual: pd.DataFrame) -> Tuple[pd.Series, float]:
        X = df_factors[factor_names].reindex(ret_actual.index).dropna()
        Y = (ret_actual - df_factors.RF).dropna()
        reg = LinearRegression()
        reg.fit(X, Y)
        ret_predict = pd.Series(reg.predict(X), index=X.index) + df_factors.RF
        score = reg.score(X, Y)
        return ret_predict, score


    def get_actual_returns(self,pname: str, daterange: Tuple[Any, Any]) -> pd.DataFrame:
            date0, date1 = str(daterange[0]), str(daterange[1])
            d=portfolios[pname].loc[date0:date1]
            return d.reindex(df_factors.index).dropna()


    def fit_portfolio_returns(self,pname: str,f_mer: bool,f_smb: bool, f_hml: bool,f_rmw: bool,f_cma: bool,ma: str,daterange: Tuple[Any, Any]):
            sel_factors = list()
            if f_mer: sel_factors.append('MER')
            if f_smb: sel_factors.append('SMB')
            if f_hml: sel_factors.append('HML')
            if f_rmw: sel_factors.append('RMW')
            if f_cma: sel_factors.append('CMA')
            if len(sel_factors) == 0:
                print("OOPS! You did not select or apply a factor(s)..!")
                return
            ret_actual =self.get_actual_returns(pname, daterange)
            ret_predict, score = fit_model(sel_factors, ret_actual)
            ix = ret_actual.index.intersection(df_factors.index)
            plt.figure(figsize=(self.f_max_width, 4))
            plt.title("Actual vs. Predicted Returns ($R^2$ = %3.2f%%)" % float(100 * score))
            plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f bp'))
            plt.plot(roll_ma(ret_actual, ma) * 1e4, label='Actual return', color='black', linewidth=0.5)
            plt.plot(roll_ma(ret_predict, ma) * 1e4, label='Predicted return', color='red', linewidth=0.5)
            plt.axhline(0.0, color='black', linestyle='dotted')
            plt.legend(), plt.show()
            plt.figure(figsize=(self.f_max_width, 3))
            for iFactor, (factor, color) in enumerate(self.colorify(factors_list, 'tab10')):
                if factor in sel_factors:
                    plt.subplot(1, len(factors_list), iFactor + 1)
                    X = df_factors[factor].loc[ix]
                    Y = ret_actual.loc[ix]
                    plt.axhline(0.0, color='black', linestyle='dotted')
                    plt.axvline(0.0, color='black', linestyle='dotted')
                    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f bp'))
                    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f bp'))
                    plt.scatter(X * 1e4, Y * 1e4, color=color, s=1), plt.title(
                        f"Actual ret. vs. %s" % inversify_factor(factor))
            plt.subplots_adjust(wspace=0.3)
            plt.show()


    @functools.lru_cache(self,maxsize=128,typed=False)
    def estimate_R2_hist(pname, daterange):
        ret_actual =self.get_actual_returns(pname, daterange)
        _factors, _scores = [], []
        factor_scenarios = ['MER', 'SMB', 'HML', 'RMW', 'CMA', 'MER+SMB', 'MER+HML', 'MER+SMB+HML', 'MER+SMB+HML+RMW',
                            'MER+SMB+HML+CMA', 'MER+SMB+HML+RMW+CMA']
        for factor_names in factor_scenarios:
            factor_names = factor_names.split('+')
            X = df_factors[factor_names].reindex(ret_actual.index).dropna()
            Y = (ret_actual - df_factors.RF).dropna()
            reg = LinearRegression()
            reg.fit(X, Y)
            score = reg.score(X, Y)
            factor_names = "\n".join(factor_names)
            _factors.append(factor_names)
            _scores.append(score)
        return _factors, _scores


    def draw_R2_hist(self,pname,daterange):
        _factors, _scores = self.estimate_R2_hist(pname, daterange)
        plt.figure(figsize=(8, 3))
        plt.title(f'Explanatory power ($R^2$) of factor combinations ({daterange[0]}-{daterange[1]})')
        plt.ylim(0, 1), plt.ylabel("$R^2$")
        plt.bar(_factors, _scores,
                color=5 * ['red'] + 2 * ['lightgreen'] + 1 * ['green'] + 2 * ['lightblue'] + 1 * ['blue'])
        plt.show()


    @functools.lru_cache(maxsize=128, typed=False)
    def estimate_r2(self,pname):
                years = list(range(1960, 2021))
                factor_scenarios = ['MER', 'SMB', 'HML', 'RMW', 'CMA', 'MER+SMB+HML', 'MER+SMB+HML+RMW+CMA']
                from collections import defaultdict
                XXX = list()
                YYY = defaultdict(list)
                for iYear, year in enumerate(years):
                    daterange = str(year - 10), str(year)
                    ret_actual = self.get_actual_returns(pname, daterange)
                    if len(ret_actual) < 5:
                        continue
                    XXX.append(pd.Timestamp(daterange[1]))
                    for factor_names in factor_scenarios:
                        factor_names = factor_names.split('+')
                        X = df_factors[factor_names].reindex(ret_actual.index).dropna()
                        Y = (ret_actual - df_factors.RF).dropna()
                        reg = LinearRegression()
                        reg.fit(X, Y)
                        score = reg.score(X, Y)
                        factor_names = "+".join(factor_names)
                        YYY[factor_names].append(score)
                return XXX, YYY


    def plot_r2(self,pname, f_mer, f_smb, f_hml, f_rmw, f_cma, daterange):
                        XXX, YYY = estimate_r2(pname)
                        plt.figure(figsize=(8, 3))

                        plt.gca().xaxis.grid(linestyle='dotted', color='black')
                        plt.title(f'Explanatory power ($R^2$) timeline')
                        sel_factors = list()
                        if f_mer: sel_factors.append('MER')
                        if f_smb: sel_factors.append('SMB')
                        if f_hml: sel_factors.append('HML')
                        if f_rmw: sel_factors.append('RMW')
                        if f_cma: sel_factors.append('CMA')
                        for f in sel_factors:
                            plt.plot(XXX, YYY[f], label=self.inversify_factor(f))
                        for f in 'MER+SMB+HML'.split(','):
                            plt.plot(XXX, YYY[f], label='3 Factor Model', color='black')
                        for f in 'MER+SMB+HML+RMW+CMA'.split(','):
                            plt.plot(XXX, YYY[f], label='5 Factor Model', linestyle='--', color='black')
                        # plt.xlim(str(daterange[0]), str(daterange[1]))
                        plt.ylabel('$R^2$'), plt.legend(loc=(1.02, 0))
                        plt.show()
 #######################################################################################







#### LOADING portfolios::
datagrid_path='data/compustat merged dataset.csv'
pf=ff5_factor()
portfolios =pf.load_portfolios(datagrid_path)
factors =pf.initialize_datagrids()['']
factors_list=[c for c in factors.columns if c != 'RF']
