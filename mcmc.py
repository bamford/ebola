#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
#from nnest.nested import NestedSampler
from scipy.stats import poisson, norm, beta, gamma, lognorm
from scipy.special import logsumexp
#import ptemcee
import emcee
import pickle
from datetime import datetime
from tqdm import tqdm
import argparse


class Ebola(object):

    def __init__(self, N, country, plot=False, onlyfirst=None):
        df = pd.read_csv('data/previous-case-counts-%s.csv' % country)
        df['WHO report date'] = pd.to_datetime(df['WHO report date'], format="%d/%m/%Y")
        df['delta_time_days'] = (df['WHO report date'] - df['WHO report date'].min()).dt.days
        df = df.sort_values('delta_time_days')
        df = df.groupby('delta_time_days').mean().reset_index()
        self.df = df
        self.N = N
        self.onlyfirst = onlyfirst
        self.country = country
        self.plot = plot
        # Differential case counts
        self.delta_cases = df['Total Cases'].values[1:] - df['Total Cases'].values[:-1]
        # Differential death counts
        self.delta_deaths = df['Total Deaths'].values[1:] - df['Total Deaths'].values[:-1]

    def rate_func(self, beta, k, tau, sigma, gamma, f):
        N = self.N
        def rate(t, y):
            S, E, I, R, C, D = y
            beta_t = beta * np.exp(-k * (t - tau))
            dydt = [
                -beta_t * S * I / N,
                beta_t * S * I / N - sigma * E,
                sigma * E - gamma * I,
                (1 - f) * gamma * I,
                sigma * E,
                f * gamma * I
            ]
            return dydt
        return rate

    def solve(self, beta, k, tau, sigma, gamma, f, offset):
        y0 = [self.N - 1, 0, 1, 0, 1, 0]
        # Offset initial time by constant
        t = self.df['delta_time_days'].values + offset
        t = t[t > 0]
        t = np.insert(t, 0, 0, axis=0)
        rate = self.rate_func(beta, k, tau, sigma, gamma, f)
        #sol = solve_ivp(fun=rate, t_span=[0, t.max()], y0=y0,
        #                t_eval=t)
        #sol = sol.y.T[1:]
        sol, info = odeint(rate, y0, t, tfirst=True, full_output=True)
        if info['message'] != 'Integration successful.':
            return None
        sol = sol[1:]
        if self.plot:
            self.makeplot()
        return sol

    def makeplot(self, samples=None, ax=None):
        if samples is not None:
            model_cases = np.zeros((len(samples), len(self.df['delta_time_days'])))
            model_deaths = np.zeros_like(model_cases)
                                         
            for i, theta in enumerate(samples):
                # compute ode model solution
                theta_ode = theta[:-6]
                sol = self.solve(*theta_ode)
                if sol is not None:
                    model_cases[i] = sol[:, 4]
                    model_deaths[i] = sol[:, 5]

            delta_model_cases = np.diff(model_cases, axis=-1)
            delta_model_deaths = np.diff(model_deaths, axis=-1)

            t = self.df['delta_time_days']
            delta_t = np.diff(t)

            rate_model_cases = delta_model_cases / delta_t
            rate_model_deaths = delta_model_deaths / delta_t
                
            c05, c16, c50, c84, c95 = np.percentile(model_cases, [5, 16, 50, 84, 95], axis=0)
            d05, d16, d50, d84, d95 = np.percentile(model_deaths, [5, 16, 50, 84, 95], axis=0)

            dc05, dc16, dc50, dc84, dc95 = np.percentile(delta_model_cases, [5, 16, 50, 84, 95], axis=0)
            dd05, dd16, dd50, dd84, dd95 = np.percentile(delta_model_deaths, [5, 16, 50, 84, 95], axis=0)
            rc05, rc16, rc50, rc84, rc95 = np.percentile(rate_model_cases, [5, 16, 50, 84, 95], axis=0)
            rd05, rd16, rd50, rd84, rd95 = np.percentile(rate_model_deaths, [5, 16, 50, 84, 95], axis=0)

        rate_cases = self.delta_cases / delta_t
        rate_deaths = self.delta_deaths / delta_t

        if ax is None:
            f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 10))
        else:
            ax0, ax1, ax2 = ax

        sample_examples = np.random.choice(len(samples), 5)

        if samples is not None:
            ax0.plot(t, c50, linestyle='solid', marker='None',
                     color='red', lw=3)
            ax0.fill_between(t, c05, c95, color='red', alpha=0.2)
            ax0.fill_between(t, c16, c84, color='red', alpha=0.2)
            ax1.plot(t[1:], dc50, linestyle='solid', marker='None',
                     color='red', lw=3)
            ax1.fill_between(t[1:], dc05, dc95, color='red', alpha=0.2)
            ax1.fill_between(t[1:], dc16, dc84, color='red', alpha=0.2)
            ax2.plot(t[1:], rc50, linestyle='solid', marker='None',
                     color='red', lw=3)
            ax2.fill_between(t[1:], rc05, rc95, color='red', alpha=0.2)
            ax2.fill_between(t[1:], rc16, rc84, color='red', alpha=0.2)
            for j in sample_examples:
                ax0.plot(t, model_cases[j], linestyle='solid',
                         marker='None', color='red')
                ax1.plot(t[1:], delta_model_cases[j], linestyle='solid',
                         marker='None', color='red')
                ax2.plot(t[1:], rate_model_cases[j], linestyle='solid',
                         marker='None', color='red')
        ax0.plot(t, self.df['Total Cases'], color='red', mfc='None', marker='o', linestyle='None')
        ax1.plot(t[1:], self.delta_cases, color='red', mfc='None', marker='o', linestyle='None')
        ax2.plot(t[1:], rate_cases, color='red', mfc='None', marker='o', linestyle='None')
        if samples is not None:
            ax0.plot(t, d50, linestyle='solid', marker='None',
                     color='blue', lw=3)
            ax0.fill_between(t, d05, d95, color='blue', alpha=0.2)
            ax0.fill_between(t, d16, d84, color='blue', alpha=0.2)
            ax1.plot(t[1:], dd50, linestyle='solid', marker='None',
                     color='blue', lw=3)
            ax1.fill_between(t[1:], dd05, dd95, color='blue', alpha=0.2)
            ax1.fill_between(t[1:], dd16, dd84, color='blue', alpha=0.2)
            ax2.plot(t[1:], rd50, linestyle='solid', marker='None',
                     color='blue', lw=3)
            ax2.fill_between(t[1:], rd05, rd95, color='blue', alpha=0.2)
            ax2.fill_between(t[1:], rd16, rd84, color='blue', alpha=0.2)
            for j in sample_examples:
                ax0.plot(t, model_deaths[j], linestyle='solid',
                         marker='None', color='blue')
                ax1.plot(t[1:], delta_model_deaths[j], linestyle='solid',
                         marker='None', color='blue')
                ax2.plot(t[1:], rate_model_deaths[j], linestyle='solid',
                         marker='None', color='blue')
        ax0.plot(t, self.df['Total Deaths'], color='blue', mfc='None', marker='o', linestyle='None')
        ax1.plot(t[1:], self.delta_deaths, color='blue', mfc='None', marker='o', linestyle='None')
        ax2.plot(t[1:], rate_deaths, color='blue', mfc='None', marker='o', linestyle='None')
        
        if self.onlyfirst is not None:
            ax0.axvline(self.onlyfirst, linestyle=':')
            ax1.axvline(self.onlyfirst, linestyle=':')
            ax2.axvline(self.onlyfirst, linestyle=':')

    def log_prior(self, theta):
        b, k, tau, sigma, g, f, offset = theta[:-6]
        scatter_cases, scatter_cases_outlier, prob_cases_outlier = theta[-6:-3]
        scatter_deaths, scatter_deaths_outlier, prob_deaths_outlier = theta[-3:]
        logPs = []
        # individual priors
        logPs.append(beta.logpdf(b, 1.1, 2))
        logPs.append(gamma.logpdf(k, 1.1, 0))
        logPs.append(lognorm.logpdf(tau, 1, 0, 10))
        logPs.append(beta.logpdf(sigma, 1.15, 2))
        logPs.append(beta.logpdf(g, 1.15, 2))
        logPs.append(beta.logpdf(f, 2, 2))
        logPs.append(lognorm.logpdf(offset, 1.5, 0, 100))
        logPs.append(lognorm.logpdf(scatter_cases, 2, 0, 10))
        logPs.append(lognorm.logpdf(scatter_cases_outlier, 0.5, 0, 100))
        logPs.append(lognorm.logpdf(scatter_deaths, 2, 0, 10))
        logPs.append(lognorm.logpdf(scatter_deaths_outlier, 0.5, 0, 100))
        logPs.append(beta.logpdf(prob_cases_outlier, 1, 100))
        logPs.append(beta.logpdf(prob_deaths_outlier, 1, 100))
        # combined priors
        logPs.append(lognorm.logpdf(scatter_cases_outlier/scatter_cases, 1.5, 1, 1000))
        logPs.append(lognorm.logpdf(scatter_deaths_outlier/scatter_deaths, 1.5, 1, 1000))
        return np.sum(logPs)

    def log_like(self, theta):
        logP = self.log_prior(theta)
        if np.isinf(logP):
            return -np.infty
        # compute ode model solution
        theta_ode = theta[:-6]
        sol = self.solve(*theta_ode)
        if sol is None:
            return -np.infty
        #sol = sol[1:]
        model_cases = sol[:, 4]
        model_deaths = sol[:, 5]
        delta_model_cases = np.diff(model_cases)
        delta_model_deaths = np.diff(model_deaths)
        np.putmask(delta_model_cases, delta_model_cases <= 0, 1e-9)
        np.putmask(delta_model_deaths, delta_model_deaths <= 0, 1e-9)
        # compute loglike
        scatter_cases, scatter_cases_outlier, prob_cases_outlier = theta[-6:-3]
        scatter_deaths, scatter_deaths_outlier, prob_deaths_outlier = theta[-3:]
        # avoid NaNs in logarithm
        prob_cases_outlier = np.clip(prob_cases_outlier, 1e-99, 1-1e-99)
        scatter_cases = np.clip(scatter_cases, 1e-9, 1e9)
        scatter_cases_outlier = np.clip(scatter_cases_outlier, 1e-9, 1e9)
        prob_deaths_outlier = np.clip(prob_deaths_outlier, 1e-99, 1-1e-99)
        scatter_deaths = np.clip(scatter_deaths, 1e-9, 1e9)
        scatter_deaths_outlier = np.clip(scatter_deaths_outlier, 1e-9, 1e9)
        
        if self.onlyfirst is not None:
            cases = self.df['Total Cases'][:self.onlyfirst]
            deaths = self.df['Total Deaths'][:self.onlyfirst]
            model_cases = model_cases[:self.onlyfirst]
            model_deaths = model_deaths[:self.onlyfirst]
        else:
            cases = self.df['Total Cases']
            deaths = self.df['Total Deaths']

        # model logL as sum of two distributions
        # 2: a Normal scatter 
        # 3: a Normal outlier scatter
        logLs = []
        # for cases
        logLs.append(np.log(1 - prob_cases_outlier) +
                     norm.logpdf(cases, model_cases, scatter_cases))
        logLs.append(np.log(prob_cases_outlier) +
                     norm.logpdf(cases, model_cases, scatter_cases_outlier))
        # for deaths
        logLs.append(np.log(1 - prob_deaths_outlier) +
                     norm.logpdf(deaths, model_deaths, scatter_deaths))
        logLs.append(np.log(prob_deaths_outlier) +
                     norm.logpdf(deaths, model_deaths, scatter_deaths_outlier))
        # using logsumexp helps maintain numerical precision
        logL = np.sum(logsumexp(logLs, axis=0))
        if np.isnan(logL):
            print(theta)
            return -np.infty
        return logL


    def log_posterior(self, theta):
        log_like = self.log_like(theta)
        log_prior = self.log_prior(theta)
        return log_prior + log_like

    def neg_log_posterior(self, theta):
        return -self.log_posterior(theta)


def main(args):

    e = Ebola(args.N, args.country, plot=False)

    # set up ptemcee
    np.random.seed(666)  # reproducible

    par = ('beta', 'k', 'tau', 'sigma', 'gamma', 'f', 'offset',
           'scatter_cases', 'scatter_cases_outlier', 'prob_cases_outlier',
           'scatter_deaths', 'scatter_deaths_outlier', 'prob_deaths_outlier')
    ndim = len(par)  # number of parameters in the model
    nwalkers = 50  # number of MCMC walkers
    ntemp = 5  # number of parallel-tempered chains
    nburn = 10000  # "burn-in" period to let chains stabilize
    nsamp = 50000  # number of MCMC steps to take after burn-in

    p0 = np.array([0.3, 0.005, 5, 0.15, 0.15, 0.5, 10, 1.0, 10.0, 0.01, 1.0, 10.0, 0.01])
    initial_theta = np.random.normal(p0, 0.1 * np.abs(p0), (ntemp, nwalkers, ndim))

    #sampler = ptemcee.Sampler(nwalkers, ndim, e.log_like, e.log_prior, ntemps=ntemp, threads=8)
    sampler = emcee.Sampler(nwalkers, ndim, e.log_posterior, threads=4)

    timestamp = datetime.now().isoformat(timespec='minutes')
    theta = initial_theta
    nthin = 10
    nupdate = 10
    nsave = 100
    for i in tqdm(range((nburn + nsamp)//nupdate)):
        sampler.run_mcmc(theta, nupdate, thin=nthin)
        theta = None
        if i % nsave == 0:
            with open('chain-{}'.format(timestamp), 'wb') as f:
                pickle.dump(sampler.chain, f)

    with open('chain-{}'.format(timestamp), 'wb') as f:
        pickle.dump(sampler.chain, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=7,
                        help="Dimensionality")
    parser.add_argument('--train_iters', type=int, default=50,
                        help="number of train iters")
    parser.add_argument("--mcmc_steps", type=int, default=0)
    parser.add_argument("--num_live_points", type=int, default=100)
    parser.add_argument('--switch', type=float, default=-1)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('-use_gpu', action='store_true')
    parser.add_argument('--flow', type=str, default='nvp')
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--noise', type=float, default=-1)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument('--run_num', type=str, default='')
    parser.add_argument('--num_slow', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--country', type=str, default='guinea')
    parser.add_argument('--N', type=int, default=1000000)

    args = parser.parse_args()
    main(args)
