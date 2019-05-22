import matplotlib.pyplot as plt

import pickle

with open('sampler', 'rb') as f:
    sampler = pickle.load(f)


# plot the chains to visually assess convergence
plt.figure(figsize=[20,10])
for i, p in enumerate(par):
    plt.subplot((ndim+1)//2, 2, i+1)
    for w in range(nwalkers):
            plt.plot(numpy.arange(sampler.chain.shape[2]), sampler.chain[2,w,:,i], 'r-', alpha=0.1)
            plt.plot(numpy.arange(sampler.chain.shape[2]), sampler.chain[0,w,:,i], 'g-', alpha=0.1)
    plt.ylabel(p)
    aymin, aymax = plt.ylim()
    plt.vlines(nburn, aymin, aymax, linestyle=':')
    plt.ylim(aymin, aymax)


# In[ ]:


# plot the chains to visually assess auto correlation time at equilibrium
plt.figure(figsize=[20,10])
for i, p in enumerate(par):
    plt.subplot((ndim+1)//2, 2, i+1)
    for w in range(0,nwalkers,10):
            plt.plot(numpy.arange(100), sampler.chain[1,w,nburn:nburn+100,i], 'r-')
            plt.plot(numpy.arange(100), sampler.chain[0,w,nburn:nburn+100,i], 'g-')
    plt.ylabel(p)
    aymin = np.min(sampler.chain[0,:,nburn:,i])
    aymax = np.max(sampler.chain[0,:,nburn:nburn+100,i])
    plt.ylim(aymin, aymax)
    plt.tight_layout()


# In[ ]:


samples = get_samples(sampler, nburn)


# In[ ]:


# examine parameter histograms
plt.figure(figsize=[20,10])
for i, p in enumerate(par):
    plt.subplot((ndim+1)//2, 2, i+1)
    n, b, patches = plt.hist(samples[:,i], bins=100, color='b', histtype='stepfilled', log=True)
    plt.xlabel(p)
    plt.tight_layout()


# In[ ]:


import corner
# create mega plot for parallel-tempered method
corner.corner(samples, labels=par)
ax = plt.subplot(2, 2, 2)
selection = np.random.choice(len(samples), 1000, replace=False)
e.makeplot(samples=samples[selection], ax=ax)
plt.subplots_adjust(wspace=0.15, hspace=0.15);


# In[ ]:


e.makeplot(samples=[])


# In[ ]:
