# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 13:09:33 2025
@author: masahr001
"""

########################1 imports
import os, json, datetime
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

########################2 compartments
class Compartment:
    def __init__(self, name: str, volume: float, pH0: float):
        self.name   = name
        self.volume = volume            # [L]
        self.H_conc = 10**(-pH0)        # [H+] initial (mol·L⁻¹)

########################3 modules

def webb_rate(irr: float, alpha: float, Vmax_psii: float) -> float:
    return Vmax_psii * (1 - np.exp(-alpha * irr / Vmax_psii))

def temp_mod(temp_C: float) -> float:
    return 1.0

def light_profile(t: float,
                  mode: str,
                  intensity: float,
                  on_t: float,
                  off_t: float,
                  ramp_t: float) -> float:
    if mode=='night': return 0.0
    if mode=='day':   return intensity
    period = on_t + off_t
    pos    = t % period
    plateau = on_t - 2*ramp_t
    if plateau <= 0:
        print("⚠ ramp_t too long → no plateau")
        return 0.0
    if pos <= off_t:
        return 0.0
    elif pos <= off_t + ramp_t:
        return intensity * ((pos - off_t)/ramp_t)
    elif pos <= off_t + ramp_t + plateau:
        return intensity
    else:
        dt = pos - (off_t + ramp_t + plateau)
        return intensity * (1 - dt/ramp_t)

_leak_warned = False
def leak(H_l: float, H_s: float,
         Vmax_leak: float, k_leak: float) -> float:
    global _leak_warned
    ΔH = H_l - H_s
    if ΔH <= 0:
        if not _leak_warned:
            print(f"⚠ ΔH={ΔH:.2e} ≤ 0 → leak=0")
            _leak_warned = True
        return 0.0
    return Vmax_leak * (1 - np.exp(-k_leak * ΔH))

def pq_and_nadph_and_cet(e_cyt: float,
                         PQH:   float,
                         PQ_tot: float,
                         H_s:    float,
                         temp_C: float,
                         Vmax_PQ_ref: float,
                         Ea_PQ:       float,
                         Km_PQ:       float,
                         Km_H:        float,
                         fCET_min:    float,
                         fCET_max:    float,
                         fCET_k:      float,
                         fCET_mid:    float) -> tuple:
    """
    Returns (dPQH, dNADPH, CET_e)
    """
    # Arrhenius scaling for PQ Vmax
    R, T_ref = 8.314, 298.15
    T = temp_C + 273.15
    Vmax_PQ = Vmax_PQ_ref * np.exp(-Ea_PQ/R * (1/T - 1/T_ref))

    # CET fraction ∈ [fCET_min, fCET_max]
    logistic = 1/(1 + np.exp(-fCET_k*(temp_C - fCET_mid)))
    frac_CET = fCET_min + (fCET_max - fCET_min)*logistic

    CET_e = frac_CET * e_cyt
    LEF_e = e_cyt - CET_e

    # NADPH: 2 e- → 1 NADPH
    dNADPH = LEF_e / 2.0

    # PQ reduction: 2 e- + 2 H+ → PQH2
    PQ     = PQ_tot - PQH
    sat_PQ = PQ    / (Km_PQ + PQ)
    sat_H  = (H_s**2)/(Km_H**2 + H_s**2)
    e_red  = e_cyt + CET_e
    rate_red = Vmax_PQ * sat_PQ * sat_H * (e_red/2)/ (Vmax_PQ or 1)

    return rate_red, dNADPH, CET_e

def b6f_pump(H_l: float, H_s: float,
             e_flux: float, n_H: float,
             gamma: float, H_half: float) -> float:
    ΔH = H_l - H_s
    gate = 1/(1 + np.exp(-gamma*(ΔH - H_half)))
    return n_H * e_flux * gate

def atp_synthase(H_l: float, H_s: float,
                 Jmax_ATP: float, K_ATP: float, n_ATP: float) -> float:
    ΔH = H_l - H_s
    J_H = Jmax_ATP * (ΔH**n_ATP)/(K_ATP**n_ATP + ΔH**n_ATP)
    return J_H/4.0  # 4 H+ → 1 ATP

########################4 ODE system

def photosyn_odes(t, y, p):
    H_l, H_s, PQH, NADPH, ATP = y
    vol_l, vol_s = p['lumen'].volume, p['stroma'].volume

    I     = light_profile(t, p['light_mode'], p['light_intensity'],
                          p['on_time'], p['off_time'], p['ramp_time'])
    Vpsii = webb_rate(I, p['alpha'], p['Vmax']) * temp_mod(p['temp_C'])
    e_cyt = Vpsii  # electrons delivered to PSI via PC after b6f

    # PQ, NADPH & CET
    rate_red_PQ, dNADPH, CET_e = pq_and_nadph_and_cet(
        e_cyt,
        PQH, p['PQ_tot'], H_s, p['temp_C'],
        p['Vmax_PQ_ref'], p['Ea_PQ'],
        p['Km_PQ'],     p['Km_H'],
        p['fCET_min'],  p['fCET_max'],
        p['fCET_k'],    p['fCET_mid']
    )

    # b6f pumping & PQH oxidation
    e_b6f      = e_cyt + CET_e
    J_b6f      = b6f_pump(H_l, H_s, e_b6f,
                          p['b6f_nH'], p['b6f_gamma'], p['b6f_half'])
    rate_ox_PQ = e_b6f / 2.0  # 2 e- oxidize 1 PQH2

    # leak & ATP
    J_leak = leak(H_l, H_s, p['Vmax_leak'], p['k_leak'])
    dATP   = atp_synthase(H_l, H_s,
                          p['Jmax_ATP'], p['K_ATP'], p['n_ATP'])

    # H+ balances (M·s⁻¹)
    dH_l = 2*Vpsii - J_leak + J_b6f - 4*dATP
    dH_s = (J_leak - J_b6f + 4*dATP)*(vol_l/vol_s) \
           - 2*rate_red_PQ*(vol_l/vol_s)

    # PQH & NADPH & ATP ODEs
    dPQH   = rate_red_PQ - rate_ox_PQ
    # NADP+: tracked implicitly via total NADP pool
    dATP   = dATP

    return [dH_l, dH_s, dPQH, dNADPH, dATP]

def run_photosyn_model(duration_s, steps, p):
    t_eval = np.linspace(0, duration_s, steps)
    y0 = [
      p['lumen'].H_conc,
      p['stroma'].H_conc,
      p['PQH0'],
      p['NADPH0'],
      p['ATP0']
    ]
    sol = solve_ivp(lambda tt,yy: photosyn_odes(tt,yy,p),
                    [0,duration_s], y0, t_eval=t_eval)
    return sol

########################5 plotting (two‐column layout)
def plot_all(sol, p):
    t = sol.t
    H_l, H_s, PQH, NADPH, ATP = sol.y

    # diagnostics
    I    = [light_profile(tt, p['light_mode'], p['light_intensity'],
                          p['on_time'], p['off_time'], p['ramp_time'])
            for tt in t]
    PSII = webb_rate(np.array(I), p['alpha'], p['Vmax'])
    Leak = [leak(h1,h2,p['Vmax_leak'],p['k_leak']) for h1,h2 in zip(H_l,H_s)]
    PQ   = p['PQ_tot'] - PQH
    ratio= PQ/(PQH+1e-16)
    eps=1e-12
    H_lp = np.maximum(H_l,eps); H_sp= np.maximum(H_s,eps)
    pH_l = -np.log10(H_lp);    pH_s=-np.log10(H_sp)
    dH_l = np.gradient(H_l, t)
    dH_s = np.gradient(H_s, t)

    fig, axs = plt.subplots(4, 2, figsize=(12,16), sharex=True)
    axs[0,0].plot(t,I)      ; axs[0,0].set_ylabel('Irradiance (µmol·m⁻²·s⁻¹)')
    axs[0,1].plot(t,PSII)   ; axs[0,1].set_ylabel('PSII Rate (mol e⁻·L⁻¹·s⁻¹)')

    axs[1,0].plot(t,Leak)   ; axs[1,0].set_ylabel('Leak Flux (mol H⁺·L⁻¹·s⁻¹)')
    axs[1,1].plot(t,ratio)  ; axs[1,1].set_ylabel('PQ/PQH₂ (–)')

    axs[2,0].plot(t,pH_l,label='lumen'); axs[2,0].plot(t,pH_s,label='stroma')
    axs[2,0].set_ylabel('pH'); axs[2,0].legend()
    axs[2,1].plot(t,dH_l,label='dHₗ/dt'); axs[2,1].plot(t,dH_s,label='dHₛ/dt')
    axs[2,1].set_ylabel('d[H⁺]/dt (M·s⁻¹)'); axs[2,1].legend()

    axs[3,0].plot(t,ATP)    ; axs[3,0].set_ylabel('ATP (mol·L⁻¹)')
    axs[3,1].plot(t,NADPH)  ; axs[3,1].set_ylabel('NADPH (mol·L⁻¹)')

    for ax in axs[3]:
        ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

########################6 main
if __name__=="__main__":
    lumen  = Compartment('lumen',  1e-5, 7.0)  
    stroma = Compartment('stroma', 1e-5, 7.0)

    params = {
      'lumen':          lumen,
      'stroma':         stroma,
      'alpha':          0.5,
      'Vmax':           30.0,
      'light_mode':     'onoff',
      'light_intensity':200.0,
      'on_time':        20.0,
      'off_time':       20.0,
      'ramp_time':      3.0,
      'Vmax_leak':      1e6,
      'k_leak':         1e-7,
      'PQ_tot':         1.4e-3,
      'PQH0':           0.7e-3,
      'Vmax_PQ_ref':    1.0,
      'Ea_PQ':          60e3,
      'Km_PQ':          1e-4,
      'Km_H':           1e-6,
      'fCET_min':       0.10,
      'fCET_max':       0.50,
      'fCET_k':         0.10,
      'fCET_mid':       20.0,
      'b6f_nH':         2.0,
      'b6f_gamma':      1e1,
      'b6f_half':       1e-5,
      'Jmax_ATP':       1000,
      'K_ATP':          1e-6,
      'n_ATP':          4.0,
      'NADPH0':         0.2e-3,
      'ATP0':           0.1e-3,
      'temp_C':         25.0
    }

    sol = run_photosyn_model(90, 900, params)
    plot_all(sol, params)
