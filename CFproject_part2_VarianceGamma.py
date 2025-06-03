#!/usr/bin/env python
# coding: utf-8

# In[1]:


### This is the extension of the previous COS pricing to the Variance Gamma model

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import norm


## Characteristic functions

def charfunc_bs(u, tau, r, sigma):
    i = 1j
    return np.exp((r - 0.5*sigma**2)*i*u*tau - 0.5*sigma**2*u**2 *tau)



def charfunc_vg(u, T, r, sigma, theta, nu):
    # risk-neutral drift adjustment
    omega = (1/nu)*np.log(1 - theta*nu - 0.5*sigma**2 *nu)
    
    # CF
    iu = 1j*u
    phi = np.exp(
        iu*(r + omega)*T
        - T/nu*np.log(1 - iu*theta*nu + 0.5*sigma**2*nu*u**2)
    )
    return phi




### COS method for Bermudan Put for given characteristic function

def COS_Bermudan_Put_CF(
    S0, r, K, exercise_times, cf, cf_params, N=256, L=10, M_proj_factor=2):

    exercise_times = np.array(exercise_times)
    M = len(exercise_times)
    T = exercise_times[-1]

    a = -L*np.sqrt(T)
    b = L*np.sqrt(T)

    k = np.arange(N).reshape((N, 1))
    u = k*np.pi/(b - a)

    # projection grid
    M_proj = M_proj_factor * N
    x = np.linspace(a, b, M_proj)
    dx = (b - a) / (M_proj - 1)

    V = np.maximum(K * (1.0 - np.exp(x)), 0.0)

    # backward induction
    for j in range(M-2, -1, -1):
        dt = exercise_times[j+1] - exercise_times[j]

        cos_matrix = np.cos(k*(np.pi*(x-a)/(b - a)))  
        integrand = V.reshape((1, M_proj))*cos_matrix       
        U = (2.0/(b - a)) * np.sum(integrand, axis=1).reshape((N, 1))*dx
        U[0] *= 0.5

        # continuation value
        phi_vals = cf(u, dt, r, *cf_params) 
        temp = phi_vals*U               
        exp_matrix = np.exp(1j*(x - a).reshape((M_proj, 1))*u.reshape((1, N)))  
        cont = np.real(exp_matrix.dot(temp)).flatten()
        cont *= np.exp(-r*dt)

        exercise_val = K*(1.0 - np.exp(x))
        exercise_val[exercise_val < 0] = 0.0

        V = np.maximum(cont, exercise_val)

    # price at t=0 
    t1 = exercise_times[0]
    cos_matrix = np.cos(k*(np.pi*(x-a)/(b - a)))
    integrand = V.reshape((1, M_proj))*cos_matrix
    U = (2.0 / (b - a))*np.sum(integrand, axis=1).reshape((N, 1))*dx
    U[0] *= 0.5

    phi_vals = cf(u, t1, r, *cf_params)
    temp = phi_vals * U
    x0 = np.log(S0/K)
    exp_vec = np.exp(1j*u.flatten()*(x0 - a))
    continuation_at_t0 = np.real(exp_vec.dot(temp).flatten())
    continuation_at_t0 *= np.exp(-r*t1)

    return float(continuation_at_t0)


# Black-Scholes European Put price
def bs_put_price(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return price



### Compare Bermudan-COS under BS vs VG

S0 = 100.0
K = 110.0
r = 0.05
sigma_bs = 0.2       #  BS volatility
# VG parameters
sigma_vg = 0.12      
theta_vg = -0.1      
nu_vg = 0.2          

T = 1.0
exercise_times = [0.1, 0.25, 0.3, 0.5, 0.75, 0.9, 1.0]

# European put under BS
european_put = bs_put_price(S0, K, r, sigma_bs, T)
print(f"Black–Scholes European Put price: {european_put:.6f}\n")

# Bermudan under BS 
Ns = [32, 48, 64, 82, 128, 200, 256, 512]
cos_bs_prices = []
for N in Ns:
    price_bs = COS_Bermudan_Put_CF(S0, r, K, exercise_times,
                                   cf=charfunc_bs,
                                   cf_params=(sigma_bs,),
                                   N=N, L=10, M_proj_factor = 2)
    cos_bs_prices.append(price_bs)

#  Bermudan under VG 
cos_vg_prices = []
for N in Ns:
    price_vg = COS_Bermudan_Put_CF(S0, r, K, exercise_times,
                                    cf=charfunc_vg,
                                    cf_params=(sigma_vg, theta_vg, nu_vg),
                                    N=N, L=10, M_proj_factor=2)
    cos_vg_prices.append(price_vg)



print("Bermudan Put Prices BS vs VG (COS):")
print("   N   |    COS-BS      |    COS-VG     ")
for N, p_bs, p_vg in zip(Ns, cos_bs_prices, cos_vg_prices):
    print(f"{N:>5}  | {p_bs:>12.6f}  | {p_vg:>12.6f}")

plt.figure(figsize=(8, 5))
plt.plot(Ns, cos_bs_prices, marker='o', label="COS Bermudan (BS)")
plt.plot(Ns, cos_vg_prices, marker='s', label="COS Bermudan (VG)")
plt.xlabel("Number of COS Terms (N)")
plt.ylabel("Bermudan Put Price")
plt.title("COS Convergence: Bermudan Put under BS vs VG")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

premium_bs = cos_bs_prices[-1] - european_put
premium_vg = cos_vg_prices[-1] - european_put


print("\nSummary (N=512):")
print(" Model  | Bermudan-COS Price | Early‐exercise Premium ")
print(f"  BS    |     {cos_bs_prices[-1]:>8.6f}      |      {premium_bs:>8.6f}")
print(f"  VG    |     {cos_vg_prices[-1]:>8.6f}      |      {premium_vg:>8.6f}")


# In[ ]:




