import numpy as np
from numba import njit


@njit
def apply_H_pbc(state: int, L: int, J: float, K: float) -> np.ndarray:
    """
    Apply the XZX Hamiltonian to a state with pbc.
    Returns a NumPy array instead of a list.
    """
    max_output_size = 2 * L  # Estimate the maximum size of output, adjust as necessary
    state_out = np.empty(max_output_size, dtype=np.int64)
    coeff_out = np.empty(max_output_size, dtype=np.float64)

    idx = 0  # Index for filling the output array

    for i in range(L):
        # Application of sigma_z^i
        coeff = -K * (2 * (state & 2**i) / 2**i - 1)

        # Application of sigma_x^(i-1) and sigma_x^(i+1)
        m = state ^ (2**((i - 1) % L) + 2**((i + 1) % L))

        #output[idx] = [coeff, m]
        state_out[idx] = m
        coeff_out[idx] = coeff
        idx += 1

        if J != 0:
            # Application of sigma_y^i and sigma_y^(i+1)
            coeff = -J * (2 * (state & 2**i) / 2**i - 1) * (2 * (state & 2**((i + 1) % L)) / 2**((i + 1) % L) - 1)
            m = state ^ (2**i + 2**((i + 1) % L))

            #output[idx] = [coeff, m]
            state_out[idx] = m
            coeff_out[idx] = coeff
            idx += 1

    return coeff_out[:idx],state_out[:idx]  # Return the populated part of the array

@njit
def traslate_state(state: int, L: int) -> int:
    """
    Translate the state to the right with periodic boundary conditions.
    """
    return state >> 1 | (state & 1) << (L - 1)

@njit
def build_basisK(L: int, k: int) -> np.ndarray:
    """
    Build the basis of states with wave number k using NumPy arrays.
    """
    periods = []
    ec = []

    for n in range(2**L):
        nn = n
        for l in range(1, L + 1):
            nn = traslate_state(nn, L)
            if nn == n:
                break
        if l in periods:
            ec[periods.index(l)].append(n)
        else:
            periods.append(l)
            ec.append([n])

    basisK = []

    for i in range(len(periods)):
        if k * periods[i] % L == 0:
            for j in range(len(ec[i])):
                basisK.append(get_RS(ec[i][j], L)[0])

    return np.unique(np.array(basisK))

@njit
def check_period(state: int, L: int) -> int:
    """
    Check the period of the state.
    """
    nn = state
    for l in range(1, L + 1):
        nn = traslate_state(nn, L)
        if nn == state:
            break
    return l

@njit
def get_RS(n: int, L: int) -> np.ndarray:
    """
    Find the representative state and the translation distance from it.
    Returns a NumPy array instead of a list.
    """
    min_state = n
    d = 0
    for l in range(1, L):
        n = traslate_state(n, L)
        if n < min_state:
            min_state = n
            d = l
    return np.array([min_state, d], dtype=np.int64)

@njit
def build_HK(L: int, k: int, J=0, K=1) -> np.ndarray:
    """
    Build the Hamiltonian matrix for the subspace with wave number k.
    """
    basis = build_basisK(L, k)
    h = np.zeros((len(basis), len(basis)), dtype=np.complex128)

    for n in basis:
        i = np.where(basis == n)[0][0]
        yn = np.sqrt(check_period(n, L)) / L
        coeff_out, state_out = apply_H_pbc(n, L, J=J, K=K)

        for coeff,m in zip(coeff_out,state_out):
            mm, d = get_RS(m, L)
            if mm in basis:
                j = np.where(basis == mm)[0][0]
                ym = np.sqrt(check_period(mm, L)) / L
                wk = np.exp(2 * np.pi * 1j * k / L)
                h[j, i] += yn / ym * wk**d * coeff

    return h
