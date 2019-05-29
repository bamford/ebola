#!/usr/bin/env python
# coding: utf-8

from __future__ import  division, print_function

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from nnest.nested import NestedSampler
from scipy.stats import poisson, norm, beta, gamma, lognorm
from scipy.special import logsumexp

scatter_cases_outlier = scatter_deaths_outlier = 1000


class Ebola(object):

    def __init__(self, N, country, weekly=False, plot=False, onlyfirst=None):
        df = pd.read_csv('data/previous-case-counts-%s.csv' % country)
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

    def rate(self, y, t, beta, k, tau, sigma, gamma, f):
        S, E, I, R, C, D = y
        beta_t = beta * np.exp(-k * (t - tau))
        dydt = [
            -beta_t * S * I / self.N,
            beta_t * S * I / self.N - sigma * E,
            sigma * E - gamma * I,
            (1 - f) * gamma * I,
            sigma * E,
            f * gamma * I
        ]
        return dydt

    def solve(self, beta, k, tau, sigma, gamma, f, offset):
        y0 = [self.N - 1, 0, 1, 0, 1, 0]
        # Offset initial time by constant
        t = self.df['delta_time'].values + offset
        t[t < 0] = 0
        t = np.insert(t, 0, 0, axis=0)
        sol = odeint(self.rate, y0, t, args=(beta, k, tau, sigma, gamma, f))
        if self.plot:
            f, ax = plt.subplots()
            ax.set_title(self.country)
            ax.plot(self.df['delta_time'], sol[1:, 4], linestyle='solid', marker='None', color='red')
            ax.plot(self.df['delta_time'], self.df['Total Cases'], color='red', mfc='None', marker='o', linestyle='None')
            ax.plot(self.df['delta_time'], sol[1:, 5], linestyle='solid', marker='None', color='blue')
            ax.plot(self.df['delta_time'], self.df['Total Deaths'], color='blue', mfc='None', marker='o', linestyle='None')
            plt.show()
        return sol

    def log_prior(self, theta):
        b, k, tau, sigma, g, f, offset = theta[:-4]
        scatter_cases, prob_cases_outlier = theta[-4:-2]
        scatter_deaths, prob_deaths_outlier = theta[-2:]
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
        logPs.append(lognorm.logpdf(scatter_deaths, 2, 0, 10))
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
        model_cases = sol[1:, 4]
        model_deaths = sol[1:, 5]
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
        logLs = []
        # for cases
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

    def __call__(self, theta):
        log_like = self.log_like(theta)
        log_prior = self.log_prior(theta)
        return log_prior + log_like


def main(args):

    weekly = not args.daily
    e = Ebola(args.N, args.country, weekly=weekly, plot=False)

    def loglike(z):
        return np.array([e(x) for x in z])

    def transform(x):
        return np.array([
            0.5 + 0.5 * x[:, 0],
            0.05 + 0.05 * x[:, 1],
            50 + 50 * x[:, 2],
            0.5 + 0.5 * x[:, 3],
            0.5 + 0.5 * x[:, 4],
            0.5 + 0.5 * x[:, 5],
            25 + 25 * x[:, 6],
            200 + 200 * x[:, 7],
            0.01 + 0.01 * x[:, 9],
            200 + 200 * x[:, 10],
            0.01 + 0.01 * x[:, 12]
        ]).T

    log_dir = os.path.join(args.log_dir, args.country)
    sampler = NestedSampler(args.x_dim, loglike, transform=transform, log_dir=log_dir, num_live_points=args.num_live_points,
                            hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_blocks=args.num_blocks, num_slow=args.num_slow,
                            use_gpu=args.use_gpu)
    sampler.run(train_iters=args.train_iters, mcmc_steps=args.mcmc_steps, volume_switch=args.switch, noise=args.noise)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--x_dim', type=int, default=13,
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
