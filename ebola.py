#!/usr/bin/env python
# coding: utf-8

from __future__ import  division, print_function

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from nnest.nested import NestedSampler
from scipy.stats import beta, gamma, lognorm
from scipy.stats import multivariate_normal as mvnorm
from ebola_data import process_data

from IPython import embed


class Ebola(object):

    def __init__(self, N, country):
        df, dfs, days, cases, cov_cases, deaths, cov_deaths = process_data(country)
        self.N = N
        self.country = country
        self.df = df
        self.dfs = dfs
        self.days = days
        self.cases = cases
        self.cov_cases = cov_cases
        self.deaths = deaths
        self.cov_deaths = cov_deaths

    def makeplot(self, samples=None, ax=None, scatter=False):
        if samples is not None:
            model_cases = np.zeros((len(samples), len(self.days)))
            model_deaths = np.zeros_like(model_cases)

            for i, theta in enumerate(samples):
                # compute ode model solution
                sol = self.solve(*theta)
                if sol is not None:
                    model_cases[i] = sol[:, 4]
                    model_deaths[i] = sol[:, 5]

            delta_model_cases = np.diff(model_cases, axis=-1)
            delta_model_deaths = np.diff(model_deaths, axis=-1)

            if scatter:
                noise_cases = mvnorm(np.zeros(len(self.days)),
                                     self.cov_cases,
                                     allow_singular=True)
                delta_model_cases += noise_cases.rvs(len(samples))[:, 1:]
                model_cases = np.cumsum(delta_model_cases, axis=-1)
                model_cases = np.insert(model_cases, 0, 0, axis=-1)
                noise_deaths = mvnorm(np.zeros(len(self.days)),
                                      self.cov_deaths,
                                      allow_singular=True)
                delta_model_deaths += noise_deaths.rvs(len(samples))[:, 1:]
                model_deaths = np.cumsum(delta_model_deaths, axis=-1)
                model_deaths = np.insert(model_deaths, 0, 0, axis=-1)

            t = self.days
            delta_t = np.diff(t)
            rate_model_cases = delta_model_cases / delta_t
            rate_model_deaths = delta_model_deaths / delta_t

            c05, c16, c50, c84, c95 = np.percentile(model_cases, [5, 16, 50, 84, 95], axis=0)
            d05, d16, d50, d84, d95 = np.percentile(model_deaths, [5, 16, 50, 84, 95], axis=0)
            rc05, rc16, rc50, rc84, rc95 = np.percentile(rate_model_cases, [5, 16, 50, 84, 95], axis=0)
            rd05, rd16, rd50, rd84, rd95 = np.percentile(rate_model_deaths, [5, 16, 50, 84, 95], axis=0)

        if ax is None:
            f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 10))
        else:
            ax0, ax1, ax2 = ax

        if samples is not None:
            ax0.plot(t, c50, linestyle='solid', marker='None',
                     color='red', lw=3)
            ax0.fill_between(t, c05, c95, color='red', alpha=0.2)
            ax0.fill_between(t, c16, c84, color='red', alpha=0.2)
            ax1.plot(t[1:], rc50, linestyle='solid', marker='None',
                     color='red', lw=3)
            ax1.fill_between(t[1:], rc05, rc95, color='red', alpha=0.2)
            ax1.fill_between(t[1:], rc16, rc84, color='red', alpha=0.2)
        ax0.plot(self.df['Day'], self.df['Total Cases'], color='red', mfc='None', marker='o', linestyle='None')
        ax1.plot(self.df['Day'], self.df['Rate Cases'], color='red', mfc='None', marker='o', linestyle='None')
        ax1.plot(t, self.cases, color='red', mfc='None', marker='None', linestyle='--')
        if samples is not None:
            ax0.plot(t, d50, linestyle='solid', marker='None',
                     color='blue', lw=3)
            ax0.fill_between(t, d05, d95, color='blue', alpha=0.2)
            ax0.fill_between(t, d16, d84, color='blue', alpha=0.2)
            ax2.plot(t[1:], rd50, linestyle='solid', marker='None',
                     color='blue', lw=3)
            ax2.fill_between(t[1:], rd05, rd95, color='blue', alpha=0.2)
            ax2.fill_between(t[1:], rd16, rd84, color='blue', alpha=0.2)
        ax0.plot(self.df['Day'], self.df['Total Deaths'], color='blue', mfc='None', marker='o', linestyle='None')
        ax2.plot(self.df['Day'], self.df['Rate Deaths'], color='blue', mfc='None', marker='o', linestyle='None')
        ax2.plot(t, self.deaths, color='blue', mfc='None', marker='None', linestyle='--')

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
        t = self.days + offset
        t[t < 0] = 0
        #t = np.insert(t, 0, 0, axis=0)
        sol = odeint(self.rate, y0, t, args=(beta, k, tau, sigma, gamma, f))
        return sol

    def log_prior(self, theta):
        b, k, tau, sigma, g, f, offset = theta
        logPs = []
        # individual priors
        logPs.append(beta.logpdf(b, 1.1, 2))
        logPs.append(gamma.logpdf(k, 1.1, 0))
        logPs.append(lognorm.logpdf(tau, 1, 0, 10))
        logPs.append(beta.logpdf(sigma, 1.15, 2))
        logPs.append(beta.logpdf(g, 1.15, 2))
        logPs.append(beta.logpdf(f, 2, 2))
        logPs.append(lognorm.logpdf(offset, 1.5, 0, 100))
        return np.sum(logPs)

    def log_like(self, theta):
        logP = self.log_prior(theta)
        if np.isinf(logP):
            return -np.infty
        # compute ode model solution
        sol = self.solve(*theta)
        if sol is None:
            return -np.infty
        model_cases = sol[:, 4]
        model_deaths = sol[:, 5]
        delta_model_cases = np.insert(np.diff(model_cases), 0, 0)
        delta_model_deaths = np.insert(np.diff(model_deaths), 0, 0)
        np.putmask(delta_model_cases, delta_model_cases <= 0, 1e-9)
        np.putmask(delta_model_deaths, delta_model_deaths <= 0, 1e-9)
        # compute loglike
        logL_cases = mvnorm.logpdf(delta_model_cases,
                                   self.cases, self.cov_cases,
                                   allow_singular=True)
        logL_cases = logL_cases.sum()
        logL_deaths = mvnorm.logpdf(delta_model_deaths,
                                    self.deaths, self.cov_deaths,
                                    allow_singular=True)
        logL_deaths = logL_deaths.sum()
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

    e = Ebola(args.N, args.country)

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
            25 + 25 * x[:, 6]
        ]).T

    log_dir = os.path.join(args.log_dir, args.country)
    sampler = NestedSampler(args.x_dim, loglike, transform=transform,
                            log_dir=log_dir, num_live_points=args.num_live_points,
                            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                            num_blocks=args.num_blocks, num_slow=args.num_slow,
                            use_gpu=args.use_gpu)
    sampler.run(train_iters=args.train_iters, mcmc_steps=args.mcmc_steps,
                volume_switch=args.switch, noise=args.noise)


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
