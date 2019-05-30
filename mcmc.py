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
from multiprocessing import Pool
import os

os.environ["OMP_NUM_THREADS"] = "1"

scatter_cases_outlier = scatter_deaths_outlier = 1000


class Ebola(object):

    def __init__(self, N, country, weekly=False, plot=False, onlyfirst=None):
        df = pd.read_csv('data/previous-case-counts-%s.csv' % country)
        df = df.drop(columns=['Unnamed: 0'])
        df['WHO report date'] = pd.to_datetime(df['WHO report date'], format="%d/%m/%Y")
        df = df.set_index('WHO report date')
        if weekly:
            df = df.resample('W').mean()
        else:
            df = df.resample('D').mean()
        df = df.dropna()
        df['delta_time'] = (df.index - df.index.min()).days
        if weekly:
             df['delta_time'] = df['delta_time'] // 7 + 1
        df = df.sort_values('delta_time')
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
        t = self.df['delta_time'].values + offset
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

    def makeplot(self, samples=None, ax=None, scatter=False, outliers=False):
        if samples is not None:
            model_cases = np.zeros((len(samples), len(self.df['delta_time'])))
            model_deaths = np.zeros_like(model_cases)

            for i, theta in enumerate(samples):
                # compute ode model solution
                theta_ode = theta[:-4]
                sol = self.solve(*theta_ode)
                if sol is not None:
                    model_cases[i] = sol[:, 4]
                    model_deaths[i] = sol[:, 5]

            delta_model_cases = np.diff(model_cases, axis=-1)
            delta_model_deaths = np.diff(model_deaths, axis=-1)

            if scatter or outliers:
                if scatter:
                    scatter_cases = samples[:, -4]
                    scatter_deaths = samples[:, -2]
                else:
                    scatter_cases = 0
                    scatter_deaths = 0
                noise_cases = np.random.normal(0, scatter_cases,
                                               (delta_model_cases.shape[1],
                                                len(samples))).T
                delta_model_cases += noise_cases
                noise_deaths = np.random.normal(0, scatter_deaths,
                                                (delta_model_deaths.shape[1],
                                                 len(samples))).T
                delta_model_deaths += noise_deaths
                if outliers:
                    prob_cases_outlier = samples[:, -3]
                    prob_deaths_outlier = samples[:, -1].T
                    noise_cases = np.random.normal(0, scatter_cases_outlier,
                                                   (delta_model_cases.shape[1],
                                                    len(samples))).T
                    noise_cases *= np.random.binomial(1, prob_cases_outlier,
                                                      (delta_model_cases.shape[1],
                                                      len(samples))).T
                    delta_model_cases += noise_cases
                    noise_deaths = np.random.normal(0, scatter_deaths_outlier,
                                                    (delta_model_deaths.shape[1],
                                                     len(samples))).T
                    noise_deaths *= np.random.binomial(1, prob_deaths_outlier,
                                                       (delta_model_deaths.shape[1],
                                                        len(samples))).T
                    delta_model_deaths += noise_deaths
                model_cases = np.cumsum(delta_model_cases, axis=-1)
                model_cases = np.insert(model_cases, 0, 0, axis=-1)
                model_deaths = np.cumsum(delta_model_deaths, axis=-1)
                model_deaths = np.insert(model_deaths, 0, 0, axis=-1)

            t = self.df['delta_time']
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

        #sample_examples = np.random.choice(len(samples), 5)
        sample_examples = []

        if samples is not None:
            ax0.plot(t, c50, linestyle='solid', marker='None',
                     color='red', lw=3)
            ax0.fill_between(t, c05, c95, color='red', alpha=0.2)
            ax0.fill_between(t, c16, c84, color='red', alpha=0.2)
            ax1.plot(t[1:], dc50, linestyle='solid', marker='None',
                     color='red', lw=3)
            ax1.fill_between(t[1:], dc05, dc95, color='red', alpha=0.2)
            ax1.fill_between(t[1:], dc16, dc84, color='red', alpha=0.2)
            for j in sample_examples:
                ax0.plot(t, model_cases[j], linestyle='solid',
                         marker='None', color='red')
                ax1.plot(t[1:], delta_model_cases[j], linestyle='solid',
                         marker='None', color='red')
        ax0.plot(t, self.df['Total Cases'], color='red', mfc='None', marker='o', linestyle='None')
        ax1.plot(t[1:], self.delta_cases, color='red', mfc='None', marker='o', linestyle='None')
        if samples is not None:
            ax0.plot(t, d50, linestyle='solid', marker='None',
                     color='blue', lw=3)
            ax0.fill_between(t, d05, d95, color='blue', alpha=0.2)
            ax0.fill_between(t, d16, d84, color='blue', alpha=0.2)
            ax2.plot(t[1:], rd50, linestyle='solid', marker='None',
                     color='blue', lw=3)
            ax2.fill_between(t[1:], dd05, dd95, color='blue', alpha=0.2)
            ax2.fill_between(t[1:], dd16, dd84, color='blue', alpha=0.2)
            for j in sample_examples:
                ax0.plot(t, model_deaths[j], linestyle='solid',
                         marker='None', color='blue')
                ax2.plot(t[1:], delta_model_deaths[j], linestyle='solid',
                         marker='None', color='blue')
        ax0.plot(t, self.df['Total Deaths'], color='blue', mfc='None', marker='o', linestyle='None')
        ax2.plot(t[1:], self.delta_deaths, color='blue', mfc='None', marker='o', linestyle='None')
        #ax0.set_ylim(0, self.df['Total Cases'].max() * 1.1)

        if self.onlyfirst is not None:
            ax0.axvline(self.onlyfirst, linestyle=':')
            ax1.axvline(self.onlyfirst, linestyle=':')
            ax2.axvline(self.onlyfirst, linestyle=':')

    def log_prior(self, theta):
        b, k, tau, sigma, g, f, offset = theta[:-4]
        scatter_cases, prob_cases_outlier = theta[-4:-2]
        scatter_deaths, prob_deaths_outlier = theta[-2:]
        # hard priors
        if prob_cases_outlier > 0.02 or prob_deaths_outlier > 0.02:
            return -np.infty
        logPs = []
        # individual priors
        logPs.append(beta.logpdf(b, 1.1, 2))
        logPs.append(gamma.logpdf(k, 1.1, 0))
        logPs.append(lognorm.logpdf(tau, 1, 0, 10))
        logPs.append(beta.logpdf(sigma, 1.15, 2))
        logPs.append(beta.logpdf(g, 1.15, 2))
        logPs.append(beta.logpdf(f, 2, 2))
        logPs.append(lognorm.logpdf(offset, 1.5, 0, 100))
        logPs.append(lognorm.logpdf(scatter_cases, 2, 0.01, 10))
        logPs.append(lognorm.logpdf(scatter_deaths, 2, 0.01, 10))
        logPs.append(beta.logpdf(prob_cases_outlier, 1, 100))
        logPs.append(beta.logpdf(prob_deaths_outlier, 1, 100))
        return np.sum(logPs)

    def log_like(self, theta):
        logP = self.log_prior(theta)
        if np.isinf(logP):
            return -np.infty
        # compute ode model solution
        theta_ode = theta[:-4]
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
        scatter_cases, prob_cases_outlier = theta[-4:-2]
        scatter_deaths, prob_deaths_outlier = theta[-2:]
        # avoid NaNs in logarithm
        prob_cases_outlier = np.clip(prob_cases_outlier, 1e-99, 1-1e-99)
        scatter_cases = np.clip(scatter_cases, 1e-9, 1e9)
        prob_deaths_outlier = np.clip(prob_deaths_outlier, 1e-99, 1-1e-99)
        scatter_deaths = np.clip(scatter_deaths, 1e-9, 1e9)

        if self.onlyfirst is not None:
            cases = self.delta_cases[:self.onlyfirst]
            deaths = self.delta_deaths[:self.onlyfirst]
            delta_model_cases = delta_model_cases[:self.onlyfirst]
            delta_model_deaths = delta_model_deaths[:self.onlyfirst]
        else:
            cases = self.delta_cases
            deaths = self.delta_deaths

        # model logL as sum of two distributions
        # 2: a Normal scatter
        # 3: a Normal outlier scatter
        # for cases
        logLs = []
        logLs.append(np.log(1 - prob_cases_outlier) +
                     norm.logpdf(cases, delta_model_cases, scatter_cases))
        logLs.append(np.log(prob_cases_outlier) +
                     norm.logpdf(cases, delta_model_cases, scatter_cases_outlier))
        # using logsumexp helps maintain numerical precision
        logL_cases = np.sum(logsumexp(logLs, axis=0))
        # for deaths
        logLs = []
        logLs.append(np.log(1 - prob_deaths_outlier) +
                     norm.logpdf(deaths, delta_model_deaths, scatter_deaths))
        logLs.append(np.log(prob_deaths_outlier) +
                     norm.logpdf(deaths, delta_model_deaths, scatter_deaths_outlier))
        # using logsumexp helps maintain numerical precision
        logL_deaths = np.sum(logsumexp(logLs, axis=0))
        # combine cases and deaths
        logL = logL_cases + logL_deaths
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

    weekly = not args.daily
    e = Ebola(args.N, args.country, weekly=weekly, plot=False)

    # set up emcee
    np.random.seed(666)  # reproducible

    par = ('beta', 'k', 'tau', 'sigma', 'gamma', 'f', 'offset',
           'scatter_cases', 'prob_cases_outlier',
           'scatter_deaths', 'prob_deaths_outlier')
    ndim = len(par)  # number of parameters in the model
    nwalkers = 500  # number of MCMC walkers
    nburn = 5000  # "burn-in" period to let chains stabilize
    nsamp = 10000  # number of MCMC steps to take after burn-in

    p0 = np.array([0.3, 0.005, 5, 0.15, 0.15, 0.5, 10, 1.0, 0.01, 1.0, 0.01])
    initial_theta = np.random.normal(p0, 0.1 * np.abs(p0), (nwalkers, ndim))

    timestamp = datetime.now().isoformat(timespec='minutes')
    theta = initial_theta
    nthin = 10
    nupdate = 10
    nsave = 100

    with Pool(1) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, e.log_posterior, pool=pool)
        for i in tqdm(range((nburn + nsamp)//nupdate)):
            sampler.run_mcmc(theta, nupdate, thin=nthin)
            theta = None
            if i % nsave == 0:
                with open('chain-{}-{}'.format(args.country, timestamp), 'wb') as f:
                    pickle.dump([sampler.chain, sampler.lnprobability], f)

    with open('chain-{}-{}'.format(args.country, timestamp), 'wb') as f:
        pickle.dump([sampler.chain, sampler.lnprobability], f)


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
    parser.add_argument('--daily', action='store_true')

    args = parser.parse_args()
    main(args)
