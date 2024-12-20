import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.sparse import block_diag as block_diag_sparse
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix


def apply_H_pbc(state, L):
    """
    Apply the XZX Hamiltonian to a state with pbc.
    
    Args:
        state (int): The input state.
        L (int): The lenght of the system.
        
    Returns:
        list: The list of [coefficient, state].
    """
    output = []
    
    for i in range(L):
        # Application of simga_z^i
        # 1 if the bit is 1 and -1 if the bit is 0
        coeff = 2 * (state & 2**i)/2**i - 1

        # Application of sigma_x^(i-1) and simga_x^(i+1)
        # Flip of the i-th and (i+1)-th bits
        m = state ^ (2**((i-1)%L) + 2**((i+1)%L))
        
        output.append([coeff, m])

    return output


def apply_H_obc(state, L):
    """
    Apply the XZX Hamiltonian to a state with obc.
    
    Args:
        state (int): The input state.
        L (int): The lenght of the system.
        
    Returns:
        list: The list of [coefficient, state].
    """
    output = []

    # First term
    #coeff = 2 * (state & 1) - 1.
    #m = state ^ 2
    #output.append([coeff, m])

    for i in range(1,L-1):

        # Application of simga_z^i
        # 1 if the bit is 1 and -1 if the bit is 0
        coeff = 2 * (state & 2**i)/2**i - 1

        # Application of sigma_x^(i-1) and simga_x^(i+1)
        # Flip of the i-th and (i+1)-th bits
        m = state ^ (2**(i-1) + 2**(i+1))

        output.append([coeff, m])

    # Last term
    #coeff = 2 * (state & 2**(L-1))/2**(L-1) - 1
    #m = state ^ 2**(L-2)
    #output.append([coeff, m])
    
    return output


def apply_H(state, L, pbc=True):
    """
    Apply the XZX Hamiltonian to a state with the wanted periodic conditions.
    
    Args:
        state (int): The input state.
        L (int): The lenght of the system.
        pbc (bool): The periodic boundary condition. True for pbc (default), False for obc.
        
    Returns:
        list: The list of [coefficient, state].
    """
    if pbc :
        return apply_H_pbc(state, L)
    else:
        return apply_H_obc(state, L)


def traslate_state(state, L):
    """
    Translate the state to the right with periodic boundary conditions.
    
    Args:
        state (int): The input state.
        L (int): The lenght of the system.

    Returns:
        int: The translated state.
    """
    return state >> 1 | (state & 1) << (L - 1)


def build_basisK(L,k):
    """
    Build the basis of states with wave number k.

    Args:
        L (int): The lenght of the system.
        k (int): The wave number.

    Returns:
        list: The basis of states wave number k.
    """
    basisK = []

    periods = []
    ec = []

    for n in range(2**L):
        nn = n
        for l in range(1,L+1):
            nn = traslate_state(nn,L)
            if nn == n:
                break
        if l in periods:
            ec[periods.index(l)].append(n)
        else:
            periods.append(l)
            ec.append([n])
        
    for i in range(len(periods)):
        if k*periods[i] % L == 0:
            for j in range(len(ec[i])):
                basisK.append(get_RS(ec[i][j],L)[0])
    
    return list(dict.fromkeys(basisK))


def check_period(state, L):
    """
    Check the period of the state.
    
    Args:
        state (int): The input state.
        L (int): The lenght of the system.
        
    Returns:
        int: The period of the state.
    """

    nn = state
    for l in range(1,L+1):
        nn = traslate_state(nn,L)
        if nn == state:
            break
    return l


def get_RS(n,L):
    """
    Find the rappresentative state and the the translation distance from it.

    Args:
        n (int): The input state.
        L (int): The lenght of the system.
    
    Returns:
        list: The list of [rappresentative state, translation distance].
    """

    min_state = n
    d = 0
    for l in range(1,L):
        n = traslate_state(n, L)
        if n < min_state:
            min_state = n
            d = l
    return [min_state, d]


def build_HK(L,k):
    """
    Build the Hamiltonian matrix for the subspace with wave number k.

    Args:
        L (int): The lenght of the system.
        k (int): The wave number.

    Returns:
        np.ndarray: The Hamiltonian matrix of the subspace.
    """
    
    basis = build_basisK(L,k)

    h = np.zeros((len(basis),len(basis)),dtype=complex)

    for n in basis:
        i = basis.index(n)
        yn = np.sqrt(check_period(n,L))/L
        output = apply_H(n, L)
        for m in output:
            mm,d = get_RS(m[1],L)
            if mm in basis:
                j = basis.index(mm)
                ym = np.sqrt(check_period(mm,L))/L
                wk = np.exp(2*np.pi*1j*k/L)
                h[j,i] += yn/ym * wk**d * m[0]
    return h


def build_fullH(L, pbc=True, sparse=False, block=False):
    """
        Build the full Hamiltonian matrix.

        Args:
            L (int): The lenght of the system.
            pbc (bool): The periodic boundary condition. True for pbc, False for obc (default True).
            sparse (bool): True to get a sparse matrix, False for a dense one (default False).
            block (bool): True to get calculate the Hamiltonian as sums of blocks using translational symmetry, False for the full one (default False). PBC are required and the result is always a sparse matrix.

        Returns:
            np.ndarray or csr_matrix: The Hamiltonian matrix.
    """
    
    if not block:
        if sparse:
            h = dok_matrix((2**L,2**L))
        else:   
            h = np.zeros((2**L,2**L))

        for n in range(2**L):
            output = apply_H(n, L, pbc)
            for m in output:
                h[n,m[1]] += m[0]

        return h
    
    else:
        if not pbc:
            raise ValueError("Block Hamiltonian can be calculated only with periodic boundary conditions")
        if not sparse:
            raise ValueError("Block Hamiltonian can be calculated only as a sparse matrix")
        return build_blockH(L)


def build_blockH(L):
    """
    Build the block Hamiltonian matrix with translational symmetry.

    Args:
        L (int): The lenght of the system.

    Returns:
        csr_matrix: The Hamiltonian matrix.
    """

    zero = build_HK(L,0)
    h = block_diag(zero,build_HK(L,1))
    for k in range(2,L):
        h=block_diag(h,build_HK(L,k))

    return h
