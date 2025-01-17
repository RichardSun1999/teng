from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
import numpy as np
import scipy as sp
from scipy import sparse
import jax
from jax import lax
import jax.numpy as jnp
# from jax.experimental import sparse as jexp_sparse
from .abstract_p_operator import AbstractPOperator

Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Union[jnp.ndarray, np.ndarray]
PyTreeDef = Any


class BurgersOperator(AbstractPOperator):

    def __init__(self, nb_dims: int, diffusion_coefs: Array, check_validity=True):
        """
        nb_dims: number of dimensions
        drift_coefs (mu_i): array of shape (nb_dims)
        diffusion_coefs (D_ij): array of shape (nb_dims, nb_dims)
        check_validity: whether to check the validity of the diffusion matrix
        computes the local operator O p (x) / p (x) with O = -\sum mu_i \partial_i + \sum D_ij \partial_i \partial_j
        """
        super().__init__()
        self.nb_dims = nb_dims
        self.diffusion_coefs = diffusion_coefs
        if check_validity:
            assert np.allclose(self.diffusion_coefs, self.diffusion_coefs.T), "diffusion matrix must be symmetric"
            assert np.all(np.linalg.eigvals(self.diffusion_coefs) > -1e-7), "diffusion matrix must be positive definite"
            assert self.diffusion_coefs.shape == (self.nb_dims, self.nb_dims), "diffusion coefs must be of shape (nb_dims, nb_dims)"

    def local_operator(self, var_state: Any, samples: Array, values: Array=None, compile: bool=True) -> Array:
        """
        computes the local operator O p (x) / p (x) with O = -\sum mu_i \partial_i + \sum D_ij \partial_i \partial_j
        """
        if compile:
            return self.local_operator_compiled(var_state, samples, values)
        else:
            assert False, "not implemented"

    def local_operator_pure(self, var_state_pure: Any, samples: Any, values: Array, state: PyTreeDef) -> Array:
        """
        a pure function version of computing the local energy.
        will be pmapped and compiled.
        however, compilation unrolls the for loop. unclear how much gain using this function
        we are writing the (possibly) less efficient version for now
        Not that we are computing O u(x)
        """
        u_func = lambda state, samples: var_state_pure.evaluate(state, samples[None, ...]).squeeze(0)
        jac_func = jax.jacrev(u_func, argnums=1)
        hes_func = jax.jacfwd(jac_func, argnums=1)
        jac_func = jax.vmap(jac_func, in_axes=(None, 0), out_axes=0)
        hes_func = jax.vmap(hes_func, in_axes=(None, 0), out_axes=0)
        jac = jac_func(state, samples).reshape(samples.shape[0], samples.shape[-1]) # combine system dimensions
        hes = hes_func(state, samples).reshape(samples.shape[0], samples.shape[-1], samples.shape[-1])  # combine system dimensions
        drift = (jac * values[..., None]).sum(axis=-1)
        diffusion = (hes * self.diffusion_coefs).sum(axis=(-1, -2))
        return jnp.clip(-drift + diffusion, a_min=-1e20, a_max=1e20)  # follow the convention of sign

    @partial(jax.pmap, in_axes=(None, None, 0, 0, None), static_broadcasted_argnums=(0, 1))
    def local_operator_pure_pmapped(self, var_state_pure: Any, samples: Any, values: Array, state: PyTreeDef) -> Array:
        return self.local_operator_pure(var_state_pure, samples, values, state)

    # @partial(jax.jit, static_argnums=(0, 1))
    # def local_energy_pure_jitted(self, var_state_pure: Any, samples: Any, log_psi: Array, state: PyTreeDef) -> Array:
    #     return self.local_energy_pure_unjitted(var_state_pure, samples, log_psi, state)

    def local_operator_compiled(self, var_state: Any, samples: Any, values: Array = None) -> Array:
        """
        wrapper of self.local_energy_pure
        """
        if values is None:
            values = var_state.log_psi(samples) # we don't actually need this for now
        return self.local_operator_pure_pmapped(var_state.pure_funcs, samples, values, var_state.get_state())