# photosyn_model_full.py
#ODE system for PSII/PSI kinetics, ferredoxin, NADP+/NADPH,
###NADPH, ATP, and O2 dynamics usng a modular Reaction framework.

import json
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Reaction:
    def __init__(
        self,
        name: str,
        rate_fn: Callable[[Dict[str, float], Dict[str, float]], float],
        stoich: Dict[str, int],
    ):
        self.name = name
        self.rate_fn = rate_fn
        self.stoich = stoich

    def flux(self, concentrations: Dict[str, float], params: Dict[str, float]) -> float:
        """Calulcate reaction flux based on curent state and parameters."""
        return self.rate_fn(concentrations, params)


#pH-dependent modifiers (palceholder for now!!!!)
def pH_factor_psi(pH: float) -> float:
    return 1.0


def pH_factor_psii(pH: float) -> float:
    return 1.0


#Placeholder metabolic rates (temp functions)
def rate_benson_calvin(nadph: float, atp: float, o2: float) -> float:
    retun 0.0


def rate_photorespiration(nadph: float, co2: float, o2: float) -> float:
    return 0.0


# -- ODE stuff --

def rate_p700(conc, params):
    pH = -np.log10(conc['H⁺'])  # compute pH from H+ conc
    excitation = pH_factor_psi(pH) * params['k_P700']
    term1 = excitation * (params['P700_total'] - conc['P700⁺']) * conc['Fd']
    term2 = params['k_pc'] * (params['Pc_total'] - conc['Pc']) * conc['P700⁺']
    return term1 - term2


def rate_p680(conc, params):
    pH = -np.log10(conc['H⁺'])
    excitation = pH_factor_psii(pH) * params['k_P680']
    term1 = excitation * (params['P680_total'] - conc['P680⁺']) * conc['Qs'] * conc['H⁺']
    term2 = params['k_water'] * conc['P680⁺']
    return term1 - term2


def rate_fd(conc, params):
    pH = -np.log10(conc['H⁺'])
    loss = pH_factor_psi(pH) * params['k_P700'] * conc['Fd'] * (params['P700_total'] - conc['P700⁺'])
    gain = (
        params['k_FN'] * conc['NADP⁺']  # electron transfer from NADP+
        + params['k_O2_Fd'] * conc['O₂']
    ) * (params['Fd_total'] - conc['Fd'])
    return -loss + gain


def rate_nadp(conc, params):
    term1 = -0.5 * params['k_FN'] * (params['Fd_total'] - conc['Fd']) * conc['NADP⁺']
    term2 = rate_benson_calvin(conc['NADPH'], conc['ATP'], conc['O₂'])  # CBC consume
    return term1 + term2


def rate_nadph(conc, params):
    term1 = params['k_NH'] * conc['H⁺'] * (
        params['N_total'] - conc['NADP⁺'] - conc['NADPH']  # total NAD pool
    )
    term2 = rate_benson_calvin(conc['NADPH'], conc['ATP'], conc['O₂'])
    return term1 - term2


def rate_atp(conc, params):
    j_integral = conc.get('J_ATP_int', 0.0)  # intergral of J_ATP(r)*r dr
    synth = (
        params['omega1'] * params['k_ATP'] / params['membrane_m']
        * (2 * np.pi / (params['b_radius']**2 - params['a_radius']**2))
        * j_integral
    )
    consume = params['omega2'] * rate_benson_calvin(
        conc['NADPH'], conc['ATP'], conc['O₂']
    )
    return synth - consume


def rate_o2(conc, params):
    produce = 0.25 * params['k_water'] * conc['P680⁺']  # water splitting
    consume = 0.5 * (
        params['k_O2_Fd'] * (params['Fd_total'] - conc['Fd'])
        + rate_photorespiration(
            conc['NADPH'], params['CO₂'], conc['O₂']
        )
    ) * conc['O₂']
    return produce - consume


###########all reactions with their labels
labels_and_rates = [
    ('P700⁺', rate_p700),
    ('P680⁺', rate_p680),
    ('Fd', rate_fd),
    ('NADP⁺', rate_nadp),
    ('NADPH', rate_nadph),
    ('ATP', rate_atp),
    ('O₂', rate_o2),
]


def odes(t, y, params):
    state = dict(zip([lbl for lbl, _ in labels_and_rates], y))
    # fixed pools and parameters
    state.update({
        'H⁺': params['H_conc'],
        'Pc': params['Pc'],  # plastocyanin pool
        'Qs': params['Qs'],  # plastoquinone pool
        'J_ATP_int': 0.0,
    })

    return [rate_fn(state, params) for _, rate_fn in labels_and_rates]


def run_simulation(duration: float, steps: int, params: Dict[str, float]):
    """Simulate model and return solver result."""
    t_points = np.linspace(0, duration, steps)
    initial = [
        params['P700_total'], 0.0,
        params['Fd_total'], params['N_total'],
        0.0, params['ATP_init'], params['O2_init'],
    ]
    res = solve_ivp(
        fun=lambda t, y: odes(t, y, params),
        t_span=(0, duration),
        y0=initial,
        t_eval=t_points,
    )
    return res


if __name__ == '__main__':
    # Smple parameters for quick test
    params = {
        'k_P700': 1e3,  'P700_total': 1.0,
        'k_pc': 1e2,   'Pc_total': 0.5,
        'k_P680': 1e3, 'P680_total': 1.0,
        'k_water': 0.1,
        'k_FN': 1e2,   'k_O2_Fd': 1e1,
        'N_total': 1.0,'k_NH': 1e2,
        'omega1': 1.0, 'omega2': 1.0,
        'membrane_m': 1.0, 'a_radius': 0.5,
        'b_radius': 1.0, 'k_ATP': 1e3,
        'H_conc': 1e-7, 'Qs': 0.5,
        'ATP_init': 0.0, 'O2_init': 0.0,
    }

    sim = run_simulation(100, 1000, params)
    # Plot the bDynamics
    for idx, lbl in enumerate([lbl for lbl, _ in labels_and_rates]):
        plt.plot(sim.t, sim.y[idx], label=lbl)
    plt.xlabel('Time')  # x-axis label
    plt.ylabel('Concentration')
    plt.title('Photosynthesis Model Dyamics')
    plt.legend()
    plt.tight_layout()
    plt.show()
