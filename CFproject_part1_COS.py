#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import norm



### Characteristic function of Black-Scholes

def charfunc_bs(u, tau, r, sigma):
    i = 1j
    return np.exp((r - 0.5*sigma**2)*i*u*tau - 0.5*sigma**2*u**2 *tau)



### COS method for Bermudan Put option (main part)
#
def COS_Bermudan_Put(
    S0, r, sigma, K, exercise_times, N=256, L=10, M_proj_factor=2):
 
    exercise_times = np.array(exercise_times) # possible exercise dates
    M = len(exercise_times)  
    T = exercise_times[-1 ]   

    a = -L*np.sqrt(T)
    b = L*np.sqrt(T)

    k = np.arange(N).reshape((N, 1))
    u = k * np.pi/(b - a)

    # Projection grid for approximating integrals
    M_proj = M_proj_factor*N
    x = np.linspace(a, b, M_proj) 
    dx = (b - a)/(M_proj-1)

    V = np.maximum(K*(1.0-np.exp(x)), 0.0)

    # backward induction of dynamic programming
    for j in range(M-2, -1, -1):
        
        dt = exercise_times[j+1] - exercise_times[j]

        # compute COS expansion coefficients 
        cos_matrix = np.cos(k*(np.pi*(x-a)/(b - a)))  
        integrand = V.reshape((1, M_proj))*cos_matrix  
        U = (2.0/(b - a))*np.sum( integrand, axis=1).reshape((N, 1))*dx
        U[0] *= 0.5 

        # Continuation value at grid x

        phi_vals = charfunc_bs(u, dt, r, sigma)  
        temp = phi_vals*U                         
        exp_matrix = np.exp(1j*(x-a).reshape((M_proj, 1))*u.reshape((1, N)))  
        cont = np.real(exp_matrix.dot(temp)).flatten()
        cont *= np.exp(-r*dt)

        exercise_val = K* (1.0 - np.exp(x))
        exercise_val[exercise_val<0] = 0.0  

        # Update option value 
        V = np.maximum(cont, exercise_val)


    t1 = exercise_times[0]
    
    # compute COS coefficients for V at first exercise 
    cos_matrix = np.cos(k*(np.pi*(x-a)/(b-a)))
    integrand = V.reshape((1, M_proj))*cos_matrix
    U = (2.0/(b-a))*np.sum(integrand, axis=1).reshape((N, 1))*dx
    U[0] *= 0.5

    phi_vals = charfunc_bs(u, t1, r, sigma)
    temp = phi_vals*U
    x0 = np.log(S0/K)
    exp_vec = np.exp(1j*u.flatten()*(x0 - a))
    continuation_at_t0 = np.real(exp_vec.dot(temp).flatten())
    continuation_at_t0 *= np.exp(-r*t1)

    return float(continuation_at_t0)



#  Binomial tree pricing

def Bermudan_Binomial_Put(S0, K, r, sigma, T, exercise_times, steps):

    dt = T/steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d)/(u-d)

    exercise_steps = sorted(set(int(np.round(t / dt)) for t in exercise_times))
    exercise_steps = [min(es, steps) for es in exercise_steps]

    ST = np.array([S0*(u**j) * (d**(steps-j)) for j in range(steps + 1)])
    V = np.maximum(K - ST, 0.0)

    #  backward induction
    for i in range(steps - 1, -1, -1):
        ST_i = np.array([S0*(u**j) * (d**(i - j)) for j in range(i + 1)])
        V = np.exp(-r*dt)*( p*V[1: i+2] + (1-p) * V[0: i+1] )
        if i in exercise_steps:
            V = np.maximum(V, np.maximum(K - ST_i,  0.0) )
    return V[0]

# Black-Scholes European Put price
def bs_put_price(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return price


### Numerical Implementation

S0 = 100
K = 110
r = 0.05
sigma = 0.2
T = 1.0

# computing European put price with Black-Scholes
european_put = bs_put_price(S0, K, r, sigma, T)
print(f'Black-Scholes European Put price: {european_put:.4f} \n')

exercise_times = [0.1, 0.25, 0.3, 0.5, 0.75, 0.9, 1.0]

# Compare COS Bermudan vs Binomial for increasing COS N
Ns = [32, 48, 64, 82, 128, 200, 256, 512]
cos_prices = []
for N in Ns:
    price_cos = COS_Bermudan_Put(S0, r, sigma, K, exercise_times, N=N, L=10)
    cos_prices.append(price_cos)

# binomial tree price
tree_steps = 1000  
price_tree = Bermudan_Binomial_Put(S0, K, r, sigma, T, exercise_times, tree_steps)

print(f"Bermudan Put Prices (COS) vs Binomial Tree for Excercise Dates {exercise_times}:")
print("\n Steps in COS (N) |     COS Price    | Binomial Price (1000 steps)")
for N, pc in zip(Ns, cos_prices):
    print( f"{N:>15} | {pc:>14.6f}  | {price_tree:>23.6f}")

# compare to European put (Black-Scholes)

print("\nComparison with European Put (Black-Scholes):")
print(" N  | Bermudan (COS) | European Put | Early‐Exercise Premium")
for N, bp  in  zip(Ns, cos_prices):
    premium =  bp - european_put
    print(f"{N:>3} | {bp:>15.6f} | {european_put:>13.6f} | {premium:>22.6f}")
        
# Analyzing Convergence of COS for different excercise dates
exercise_sets = [    
    [1.0],
    [0.5, 1.0],                       
    [0.25, 0.5, 0.75, 1.0],              
    [0.1, 0.25, 0.3, 0.5, 0.75, 0.9, 1.0]
]


fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.flatten()

 # dictionary to store the price at N=512 for each exercise set
results_at_N512 = {}

for idx, ex in enumerate(exercise_sets):
    cos_prices =  []
    for N in Ns:
        price_cos = COS_Bermudan_Put(S0, r, sigma, K, ex, N=N, L=10)
        cos_prices.append(price_cos)

    results_at_N512[tuple(ex)] = cos_prices[-1]

    # binomial price for each exercise set
    price_tree_ex = Bermudan_Binomial_Put(S0, K, r, sigma, T, ex, tree_steps)

    ax = axes[idx]
    ax.plot(Ns, cos_prices, marker="o", label="COS Bermudan")
    ax.hlines(price_tree_ex, Ns[0], Ns[-1],
              colors="r", linestyles="--", label="Binomial Tree")
    ax.set_xlabel("Number of COS terms (N)")
    ax.set_ylabel("Bermudan Put Price")
    label = ", ".join(f"{t:.2f}" for t in ex)
    ax.set_title(f"Exercise dates: [{label}]")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

## Price and Premium for different excercise dates
print("\nConvergence w.r.t. number of exercise dates (price at N=512):")
for ex, price in results_at_N512.items():
    label = ", ".join(f"{t:.2f}" for t in ex)
    print(f"Exercise dates: [{label}] -  Price: {price:.6f}, Early-Excercise Premium:{price-european_put:.6f}")



# In[ ]:




