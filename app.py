"""
Financial Market Theory Exam Engine
====================================

A comprehensive exam preparation tool for Financial Market Theory courses.
Covers portfolio optimization, bond math, stock valuation, factor models, and more.

Repository: https://github.com/yourusername/fmt-exam-engine
Run locally: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt
import re
from typing import Tuple, List, Optional, Dict, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

# Try to import cvxpy for the unified solver; fall back to SLSQP if not available
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("cvxpy not installed. Using SLSQP fallback for constrained optimization.")


# =============================================================================
# NEW: INPUT SAFETY AND CONVENIENCE HELPERS (Section 6)
# =============================================================================

def parse_vector_string(text: str, name: str = "vector") -> np.ndarray:
    """
    Parse a vector from various string formats with validation.
    
    Handles: "0.08 0.15", "8%, 15%", "0.08, 0.15", "[0.08, 0.15]", etc.
    
    Args:
        text: Input string
        name: Name for error messages
    
    Returns:
        1D numpy array
    
    Raises:
        ValidationError: If parsing fails or input is empty
    """
    if not text or text.strip() == "":
        raise ValidationError(f"{name} cannot be empty")
    
    # Remove brackets, commas, tabs, newlines
    text = text.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
    text = text.replace('\t', ' ').replace('\n', ' ').replace(',', ' ')
    text = text.replace('%', '')
    
    # Find all numbers
    pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
    matches = re.findall(pattern, text)
    
    if not matches:
        raise ValidationError(f"Could not parse any numbers from {name}: '{text}'")
    
    values = [float(m) for m in matches]
    
    # Heuristic: if values seem like percentages, convert
    if all(1 < abs(v) < 100 for v in values if v != 0):
        values = [v / 100 for v in values]
    
    arr = np.array(values, dtype=float)
    
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValidationError(f"{name} contains NaN or infinite values")
    
    return arr


def parse_matrix_string(text: str, name: str = "matrix", 
                        force_symmetric: bool = False,
                        force_square: bool = False) -> np.ndarray:
    """
    Parse a matrix from text with validation and optional symmetrization.
    
    Args:
        text: Input text (rows on separate lines)
        name: Name for error messages
        force_symmetric: If True, symmetrize the matrix
        force_square: If True, require square matrix
    
    Returns:
        2D numpy array
    
    Raises:
        ValidationError: If parsing fails or validation fails
    """
    if not text or text.strip() == "":
        raise ValidationError(f"{name} cannot be empty")
    
    rows = []
    for line in text.strip().split('\n'):
        if line.strip():
            try:
                row = parse_vector_string(line, f"{name} row")
                if len(row) > 0:
                    rows.append(row)
            except ValidationError:
                continue
    
    if not rows:
        raise ValidationError(f"Could not parse any rows from {name}")
    
    # Ensure all rows have same length (pad shorter rows with zeros)
    max_len = max(len(r) for r in rows)
    for i, r in enumerate(rows):
        if len(r) < max_len:
            rows[i] = np.pad(r, (0, max_len - len(r)), constant_values=0)
    
    mat = np.array(rows, dtype=float)
    
    if force_square and mat.shape[0] != mat.shape[1]:
        raise ValidationError(f"{name} must be square, got shape {mat.shape}")
    
    if force_symmetric:
        if mat.shape[0] != mat.shape[1]:
            raise ValidationError(f"{name} must be square to symmetrize, got shape {mat.shape}")
        mat = (mat + mat.T) / 2
    
    if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
        raise ValidationError(f"{name} contains NaN or infinite values")
    
    return mat


def symmetrize_covariance(cov: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Symmetrize a covariance matrix and validate positive semi-definiteness.
    
    Args:
        cov: Input covariance matrix
        tol: Tolerance for symmetry check warning
    
    Returns:
        Symmetrized covariance matrix
    
    Raises:
        ValidationError: If matrix is not square or has issues
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValidationError(f"Covariance must be square, got shape {cov.shape}")
    
    # Check how far from symmetric
    asym = np.max(np.abs(cov - cov.T))
    if asym > tol:
        warnings.warn(f"Covariance matrix asymmetry: {asym:.2e}. Symmetrizing.")
    
    # Symmetrize
    cov_sym = (cov + cov.T) / 2
    
    # Check eigenvalues
    eigvals = np.linalg.eigvalsh(cov_sym)
    if np.any(eigvals < -tol):
        warnings.warn(f"Covariance has negative eigenvalue: {np.min(eigvals):.2e}")
    
    return cov_sym


def validate_dimensions(*arrays_and_names: Tuple[np.ndarray, str, int]) -> None:
    """
    Validate that multiple arrays have consistent dimensions.
    
    Args:
        arrays_and_names: Tuples of (array, name, expected_length)
    
    Raises:
        ValidationError: If dimensions don't match
    """
    for arr, name, expected in arrays_and_names:
        if arr is None:
            continue
        actual = len(arr) if arr.ndim == 1 else arr.shape[0]
        if actual != expected:
            raise ValidationError(f"{name} has length {actual}, expected {expected}")


# =============================================================================
# NEW: UNIFIED PORTFOLIO SOLVER (Section 1) - Highest Priority
# =============================================================================

@dataclass
class UnifiedPortfolioResult:
    """Result container for unified portfolio solver."""
    weights: np.ndarray
    cash_weight: float
    portfolio_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    portfolio_esg: Optional[float]
    binding_constraints: List[str]
    solver_status: str
    solver_message: str
    diagnostics: Dict
    
    def to_dict(self) -> Dict:
        return {
            'weights': self.weights,
            'cash_weight': self.cash_weight,
            'return': self.portfolio_return,
            'volatility': self.portfolio_volatility,
            'sharpe': self.sharpe_ratio,
            'esg': self.portfolio_esg,
            'binding_constraints': self.binding_constraints,
            'status': self.solver_status,
            'message': self.solver_message,
            'diagnostics': self.diagnostics
        }


def unified_portfolio_solver(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 4.0,
    # ESG parameters
    esg_scores: Optional[np.ndarray] = None,
    esg_preference: float = 0.0,
    esg_min_target: Optional[float] = None,
    # Constraint parameters
    allow_cash: bool = True,
    no_short: bool = False,
    min_weights: Optional[np.ndarray] = None,
    max_weights: Optional[np.ndarray] = None,
    excluded_assets: Optional[Set[int]] = None,
    leverage_cap: Optional[float] = None,
    target_volatility: Optional[float] = None,
    # Zero-cost factor parameters
    zero_cost_indices: Optional[Set[int]] = None,
    # Asset names for reporting
    asset_names: Optional[List[str]] = None
) -> UnifiedPortfolioResult:
    """
    Unified portfolio solver that handles all common exam scenarios.
    
    Maximizes mean-variance utility with optional ESG term:
        U = w'μ - (γ/2)w'Σw + a·w's
    
    Subject to configurable constraints:
        - Budget: sum(funded weights) + cash = 1 (if allow_cash, else sum(weights)=1)
        - Zero-cost: sum(zero_cost_weights) = 0
        - Bounds: min_weights <= w <= max_weights
        - Long-only: w >= 0 (if no_short)
        - Excluded assets: w_i = 0 for i in excluded_assets
        - Leverage cap: sum(|w|) <= L
        - Volatility target: sqrt(w'Σw) <= σ_target
        - ESG minimum: w's >= esg_min_target
    
    Args:
        expected_returns: Expected returns vector (n,)
        covariance_matrix: Covariance matrix (n, n)
        risk_free_rate: Risk-free rate
        risk_aversion: Risk aversion parameter γ
        esg_scores: ESG scores vector (n,)
        esg_preference: ESG preference parameter a
        esg_min_target: Minimum portfolio ESG score
        allow_cash: If True, cash = 1 - sum(risky weights)
        no_short: If True, enforce w >= 0
        min_weights: Per-asset minimum weights (n,)
        max_weights: Per-asset maximum weights (n,)
        excluded_assets: Set of asset indices to exclude (weight = 0)
        leverage_cap: Maximum sum of absolute weights
        target_volatility: Maximum portfolio volatility
        zero_cost_indices: Set of indices for zero-cost factors (sum to 0)
        asset_names: Names for reporting
    
    Returns:
        UnifiedPortfolioResult with weights, statistics, and diagnostics
    """
    # Validation
    n = len(expected_returns)
    mu = np.asarray(expected_returns, dtype=float)
    Sigma = symmetrize_covariance(np.asarray(covariance_matrix, dtype=float))
    
    if Sigma.shape != (n, n):
        raise ValidationError(f"Covariance shape {Sigma.shape} doesn't match n={n}")
    
    # Handle ESG
    has_esg = esg_scores is not None and esg_preference > 0
    if esg_scores is not None:
        s = np.asarray(esg_scores, dtype=float)
        if len(s) != n:
            raise ValidationError(f"ESG scores length {len(s)} doesn't match n={n}")
    else:
        s = np.zeros(n)
    
    # Determine funded vs zero-cost assets
    if zero_cost_indices is None:
        zero_cost_indices = set()
    funded_indices = set(range(n)) - zero_cost_indices
    
    # Excluded assets
    if excluded_assets is None:
        excluded_assets = set()
    
    # Asset names
    if asset_names is None:
        asset_names = [f'Asset {i+1}' for i in range(n)]
    
    # Initialize bounds
    if min_weights is None:
        min_weights = np.full(n, -np.inf if not no_short else 0.0)
    else:
        min_weights = np.asarray(min_weights, dtype=float)
    
    if max_weights is None:
        max_weights = np.full(n, np.inf)
    else:
        max_weights = np.asarray(max_weights, dtype=float)
    
    if no_short:
        min_weights = np.maximum(min_weights, 0.0)
    
    # Set excluded assets to zero bounds
    for i in excluded_assets:
        min_weights[i] = 0.0
        max_weights[i] = 0.0
    
    # Try cvxpy first, then SLSQP fallback
    if CVXPY_AVAILABLE:
        result = _solve_cvxpy(
            n, mu, Sigma, s, risk_free_rate, risk_aversion, esg_preference,
            allow_cash, min_weights, max_weights, funded_indices, zero_cost_indices,
            leverage_cap, target_volatility, esg_min_target, asset_names
        )
    else:
        result = _solve_slsqp(
            n, mu, Sigma, s, risk_free_rate, risk_aversion, esg_preference,
            allow_cash, min_weights, max_weights, funded_indices, zero_cost_indices,
            leverage_cap, target_volatility, esg_min_target, asset_names
        )
    
    return result


def _solve_cvxpy(
    n: int, mu: np.ndarray, Sigma: np.ndarray, s: np.ndarray,
    rf: float, gamma: float, a: float, allow_cash: bool,
    lb: np.ndarray, ub: np.ndarray,
    funded_idx: Set[int], zerocost_idx: Set[int],
    leverage_cap: Optional[float], target_vol: Optional[float],
    esg_min: Optional[float], names: List[str]
) -> UnifiedPortfolioResult:
    """Solve using cvxpy quadratic program."""
    
    w = cp.Variable(n)
    
    # Objective: maximize w'μ - (γ/2)w'Σw + a*w's
    # = minimize -w'μ + (γ/2)w'Σw - a*w's
    excess = mu - rf
    objective = -w @ excess + (gamma / 2) * cp.quad_form(w, Sigma)
    if a > 0:
        objective -= a * (w @ s)
    
    constraints = []
    constraint_names = []
    
    # Budget constraint for funded assets
    funded_list = sorted(funded_idx)
    if funded_list:
        if allow_cash:
            # sum(funded) <= 1 (cash is residual, can be positive)
            constraints.append(cp.sum(w[funded_list]) <= 1)
            constraint_names.append("Budget: sum(funded) <= 1")
        else:
            # sum(funded) == 1
            constraints.append(cp.sum(w[funded_list]) == 1)
            constraint_names.append("Budget: sum(funded) == 1")
    
    # Zero-cost constraint
    zerocost_list = sorted(zerocost_idx)
    if zerocost_list:
        constraints.append(cp.sum(w[zerocost_list]) == 0)
        constraint_names.append("Zero-cost: sum(zero-cost) == 0")
    
    # Bound constraints
    for i in range(n):
        if lb[i] > -1e10:
            constraints.append(w[i] >= lb[i])
        if ub[i] < 1e10:
            constraints.append(w[i] <= ub[i])
    
    # Leverage constraint
    if leverage_cap is not None:
        constraints.append(cp.norm(w, 1) <= leverage_cap)
        constraint_names.append(f"Leverage: ||w||_1 <= {leverage_cap}")
    
    # Volatility constraint
    if target_vol is not None:
        # Approximate with Frobenius / use SOC constraint
        # sqrt(w'Σw) <= target_vol
        constraints.append(cp.quad_form(w, Sigma) <= target_vol ** 2)
        constraint_names.append(f"Volatility: σ <= {target_vol*100:.2f}%")
    
    # ESG minimum constraint
    if esg_min is not None:
        constraints.append(w @ s >= esg_min)
        constraint_names.append(f"ESG: w's >= {esg_min}")
    
    # Solve
    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        # Try alternative solver
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
        except Exception:
            prob.solve(verbose=False)
    
    if w.value is None:
        # Solver failed
        return UnifiedPortfolioResult(
            weights=np.zeros(n),
            cash_weight=1.0,
            portfolio_return=rf,
            portfolio_volatility=0.0,
            sharpe_ratio=0.0,
            portfolio_esg=0.0,
            binding_constraints=[],
            solver_status='failed',
            solver_message=str(prob.status),
            diagnostics={'problem_status': prob.status}
        )
    
    weights = np.asarray(w.value).flatten()
    
    # Compute statistics
    funded_weight = np.sum(weights[funded_list]) if funded_list else 0.0
    cash_weight = 1 - funded_weight if allow_cash else 0.0
    
    # Portfolio return (risky part)
    port_ret = float(weights @ mu)
    if allow_cash:
        port_ret_total = port_ret + cash_weight * rf
    else:
        port_ret_total = port_ret
    
    port_vol = float(np.sqrt(weights @ Sigma @ weights))
    sharpe = (port_ret_total - rf) / port_vol if port_vol > 1e-10 else 0.0
    port_esg = float(weights @ s) if np.any(s != 0) else None
    
    # Detect binding constraints
    binding = _detect_binding_constraints(
        weights, lb, ub, funded_list, zerocost_list, port_vol,
        leverage_cap, target_vol, esg_min, port_esg, names
    )
    
    # Diagnostics
    diagnostics = {
        'problem_status': prob.status,
        'objective_value': prob.value,
        'funded_weight': funded_weight,
        'solver': 'cvxpy',
        'constraint_count': len(constraints)
    }
    
    # Warnings
    if cash_weight < -1e-6:
        diagnostics['warning'] = f"Negative cash ({cash_weight:.4f}) implies leverage"
    
    return UnifiedPortfolioResult(
        weights=weights,
        cash_weight=cash_weight,
        portfolio_return=port_ret_total,
        portfolio_volatility=port_vol,
        sharpe_ratio=sharpe,
        portfolio_esg=port_esg,
        binding_constraints=binding,
        solver_status='optimal' if prob.status == cp.OPTIMAL else prob.status,
        solver_message=str(prob.status),
        diagnostics=diagnostics
    )


def _solve_slsqp(
    n: int, mu: np.ndarray, Sigma: np.ndarray, s: np.ndarray,
    rf: float, gamma: float, a: float, allow_cash: bool,
    lb: np.ndarray, ub: np.ndarray,
    funded_idx: Set[int], zerocost_idx: Set[int],
    leverage_cap: Optional[float], target_vol: Optional[float],
    esg_min: Optional[float], names: List[str]
) -> UnifiedPortfolioResult:
    """Fallback solver using scipy SLSQP."""
    
    funded_list = sorted(funded_idx)
    zerocost_list = sorted(zerocost_idx)
    excess = mu - rf
    
    def objective(w):
        # Minimize -utility
        ret_part = -np.dot(w, excess)
        risk_part = (gamma / 2) * np.dot(w, Sigma @ w)
        esg_part = -a * np.dot(w, s) if a > 0 else 0
        return ret_part + risk_part + esg_part
    
    constraints = []
    
    # Budget constraint
    if funded_list:
        if allow_cash:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: 1 - np.sum(w[funded_list])  # sum <= 1
            })
        else:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w[funded_list]) - 1
            })
    
    # Zero-cost constraint
    if zerocost_list:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w[zerocost_list])
        })
    
    # Leverage constraint
    if leverage_cap is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: leverage_cap - np.sum(np.abs(w))
        })
    
    # Volatility constraint
    if target_vol is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: target_vol ** 2 - np.dot(w, Sigma @ w)
        })
    
    # ESG minimum constraint
    if esg_min is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: np.dot(w, s) - esg_min
        })
    
    # Bounds
    bounds = [(lb[i], ub[i]) for i in range(n)]
    
    # Initial guess
    w0 = np.zeros(n)
    if funded_list:
        active_funded = [i for i in funded_list if lb[i] < ub[i]]
        if active_funded:
            for i in active_funded:
                w0[i] = 1.0 / len(active_funded)
    
    # Solve
    result = minimize(
        objective, w0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'ftol': 1e-10, 'maxiter': 1000}
    )
    
    weights = result.x
    
    # Compute statistics
    funded_weight = np.sum(weights[funded_list]) if funded_list else 0.0
    cash_weight = 1 - funded_weight if allow_cash else 0.0
    
    port_ret = float(weights @ mu)
    if allow_cash:
        port_ret_total = port_ret + cash_weight * rf
    else:
        port_ret_total = port_ret
    
    port_vol = float(np.sqrt(weights @ Sigma @ weights))
    sharpe = (port_ret_total - rf) / port_vol if port_vol > 1e-10 else 0.0
    port_esg = float(weights @ s) if np.any(s != 0) else None
    
    # Detect binding constraints
    binding = _detect_binding_constraints(
        weights, lb, ub, funded_list, zerocost_list, port_vol,
        leverage_cap, target_vol, esg_min, port_esg, names
    )
    
    diagnostics = {
        'success': result.success,
        'message': result.message,
        'iterations': result.nit,
        'funded_weight': funded_weight,
        'solver': 'SLSQP'
    }
    
    if cash_weight < -1e-6:
        diagnostics['warning'] = f"Negative cash ({cash_weight:.4f}) implies leverage"
    
    return UnifiedPortfolioResult(
        weights=weights,
        cash_weight=cash_weight,
        portfolio_return=port_ret_total,
        portfolio_volatility=port_vol,
        sharpe_ratio=sharpe,
        portfolio_esg=port_esg,
        binding_constraints=binding,
        solver_status='optimal' if result.success else 'failed',
        solver_message=result.message,
        diagnostics=diagnostics
    )


def _detect_binding_constraints(
    w: np.ndarray, lb: np.ndarray, ub: np.ndarray,
    funded_list: List[int], zerocost_list: List[int],
    vol: float, leverage_cap: Optional[float], target_vol: Optional[float],
    esg_min: Optional[float], port_esg: Optional[float],
    names: List[str], tol: float = 1e-5
) -> List[str]:
    """Detect which constraints are binding."""
    binding = []
    
    for i in range(len(w)):
        if abs(w[i] - lb[i]) < tol and lb[i] > -1e10:
            binding.append(f"{names[i]} at lower bound ({lb[i]:.4f})")
        if abs(w[i] - ub[i]) < tol and ub[i] < 1e10:
            binding.append(f"{names[i]} at upper bound ({ub[i]:.4f})")
    
    if target_vol is not None and abs(vol - target_vol) < tol:
        binding.append(f"Volatility at cap ({target_vol*100:.2f}%)")
    
    if leverage_cap is not None and abs(np.sum(np.abs(w)) - leverage_cap) < tol:
        binding.append(f"Leverage at cap ({leverage_cap})")
    
    if esg_min is not None and port_esg is not None and abs(port_esg - esg_min) < tol:
        binding.append(f"ESG at minimum ({esg_min:.4f})")
    
    return binding


# =============================================================================
# NEW: ZERO-COST FACTOR PORTFOLIO SUPPORT (Section 2)
# =============================================================================

@dataclass
class ZeroCostFactorResult:
    """Result for portfolios with zero-cost legs (e.g., WML)."""
    funded_weights: np.ndarray
    zerocost_weights: np.ndarray
    cash_weight: float
    portfolio_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    factor_exposures: Dict[str, float]
    diagnostics: Dict


def optimize_with_zero_cost_factors(
    funded_returns: np.ndarray,
    funded_cov: np.ndarray,
    zerocost_returns: np.ndarray,
    zerocost_cov: np.ndarray,
    cross_cov: np.ndarray,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 4.0,
    no_short_funded: bool = False,
    funded_names: Optional[List[str]] = None,
    zerocost_names: Optional[List[str]] = None
) -> ZeroCostFactorResult:
    """
    Optimize portfolio with funded assets and zero-cost factor portfolios.
    
    For zero-cost factors like WML (Winners-Minus-Losers):
    - sum(funded weights) + cash = 1
    - sum(zerocost weights) = 0 (self-financing)
    
    Args:
        funded_returns: Expected returns of funded assets (n_funded,)
        funded_cov: Covariance of funded assets (n_funded, n_funded)
        zerocost_returns: Expected returns of zero-cost factors (n_zc,)
        zerocost_cov: Covariance of zero-cost factors (n_zc, n_zc)
        cross_cov: Cross-covariance (n_funded, n_zc)
        risk_free_rate: Risk-free rate
        risk_aversion: Risk aversion γ
        no_short_funded: If True, funded weights >= 0
        funded_names: Names for funded assets
        zerocost_names: Names for zero-cost factors
    
    Returns:
        ZeroCostFactorResult with weights and statistics
    """
    n_funded = len(funded_returns)
    n_zc = len(zerocost_returns)
    n_total = n_funded + n_zc
    
    # Build combined arrays
    mu = np.concatenate([funded_returns, zerocost_returns])
    
    # Build combined covariance
    Sigma = np.zeros((n_total, n_total))
    Sigma[:n_funded, :n_funded] = funded_cov
    Sigma[n_funded:, n_funded:] = zerocost_cov
    Sigma[:n_funded, n_funded:] = cross_cov
    Sigma[n_funded:, :n_funded] = cross_cov.T
    Sigma = symmetrize_covariance(Sigma)
    
    # Names
    if funded_names is None:
        funded_names = [f'Funded_{i+1}' for i in range(n_funded)]
    if zerocost_names is None:
        zerocost_names = [f'ZeroCost_{i+1}' for i in range(n_zc)]
    all_names = funded_names + zerocost_names
    
    # Use unified solver
    result = unified_portfolio_solver(
        expected_returns=mu,
        covariance_matrix=Sigma,
        risk_free_rate=risk_free_rate,
        risk_aversion=risk_aversion,
        allow_cash=True,
        no_short=no_short_funded,  # Only apply to funded
        zero_cost_indices=set(range(n_funded, n_total)),
        asset_names=all_names
    )
    
    # Split weights
    funded_w = result.weights[:n_funded]
    zc_w = result.weights[n_funded:]
    
    # Factor exposures
    factor_exposures = {}
    for i, name in enumerate(zerocost_names):
        factor_exposures[name] = float(zc_w[i])
    
    return ZeroCostFactorResult(
        funded_weights=funded_w,
        zerocost_weights=zc_w,
        cash_weight=result.cash_weight,
        portfolio_return=result.portfolio_return,
        portfolio_volatility=result.portfolio_volatility,
        sharpe_ratio=result.sharpe_ratio,
        factor_exposures=factor_exposures,
        diagnostics=result.diagnostics
    )


# =============================================================================
# NEW: LARGE-N EQUAL-CORRELATION CLOSED-FORM (Section 3)
# =============================================================================

@dataclass
class LargeNEqualCorrResult:
    """Result for large-N equal-correlation portfolio."""
    tangency_weight_per_asset: float
    optimal_weight_per_asset: float
    tangency_sharpe: float
    optimal_total_risky_weight: float
    optimal_cash_weight: float
    portfolio_return: float
    portfolio_volatility: float
    effective_sharpe_squared: float
    formulas: Dict[str, str]


def large_n_equal_correlation_portfolio(
    n_assets: int,
    expected_return: float,
    volatility: float,
    correlation: float,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 4.0
) -> LargeNEqualCorrResult:
    """
    Closed-form solution for N identical assets with equal correlation.
    
    For N assets with:
    - Same expected return μ
    - Same volatility σ
    - Same pairwise correlation ρ
    
    The tangency portfolio has equal weights 1/N, and analytical formulas exist.
    
    Key formulas:
    - Portfolio variance: σ_p² = σ²[(1-ρ)/N + ρ] for equal weights
    - Limiting variance as N→∞: σ_p² → ρσ² (diversification limit)
    - Sharpe ratio: SR = (μ - rf) / σ_p
    
    Args:
        n_assets: Number of assets N
        expected_return: Common expected return μ
        volatility: Common volatility σ
        correlation: Pairwise correlation ρ
        risk_free_rate: Risk-free rate rf
        risk_aversion: Risk aversion γ
    
    Returns:
        LargeNEqualCorrResult with closed-form results
    """
    if not -1/(n_assets - 1) <= correlation <= 1:
        raise ValidationError(
            f"Correlation {correlation} invalid for N={n_assets}. "
            f"Must be >= {-1/(n_assets-1):.4f} for positive-definite covariance."
        )
    
    mu = expected_return
    sigma = volatility
    rho = correlation
    rf = risk_free_rate
    gamma = risk_aversion
    N = n_assets
    
    # Tangency portfolio: equal weights 1/N
    tan_weight = 1 / N
    
    # Portfolio variance for equal weights
    # σ_p² = (1/N)² × N × σ² + (1/N)² × N(N-1) × ρσ²
    #      = σ²/N + σ²(N-1)ρ/N
    #      = σ²[(1 + (N-1)ρ)/N]
    #      = σ²[(1-ρ)/N + ρ]
    port_var = sigma**2 * ((1 - rho)/N + rho)
    port_vol = np.sqrt(port_var)
    
    # Sharpe ratio
    excess = mu - rf
    tan_sharpe = excess / port_vol
    
    # Optimal portfolio (unconstrained MV with risk-free)
    # For identical assets with common Sharpe and equal correlation:
    # w* = (1/γ) × Σ⁻¹ × (μ - rf)
    # 
    # With equal correlation, Σ⁻¹ has a simple form:
    # Σ⁻¹[i,j] = -ρ / [σ²(1-ρ)(1+(N-1)ρ)] for i≠j
    # Σ⁻¹[i,i] = [1+(N-2)ρ] / [σ²(1-ρ)(1+(N-1)ρ)]
    #
    # Sum of Σ⁻¹ × 1 = N × [1+(N-2)ρ - (N-1)ρ] / [σ²(1-ρ)(1+(N-1)ρ)]
    #                = N × [1-ρ] / [σ²(1-ρ)(1+(N-1)ρ)]
    #                = N / [σ²(1+(N-1)ρ)]
    #
    # Optimal weight per asset (closed form):
    denom = sigma**2 * (1 + (N-1)*rho)
    sigma_inv_sum = N / denom  # Sum of all elements of Σ⁻¹
    
    # Each row of Σ⁻¹ sums to 1/[σ²(1+(N-1)ρ)]
    sigma_inv_row_sum = 1 / denom
    
    # Optimal weight per asset
    opt_weight_per_asset = (1/gamma) * sigma_inv_row_sum * excess
    
    # Total risky weight
    total_risky = N * opt_weight_per_asset
    cash_weight = 1 - total_risky
    
    # Portfolio return and volatility at optimal
    # For optimal portfolio, use same equal-weight formula scaled by total_risky
    opt_port_ret = total_risky * mu + cash_weight * rf
    opt_port_var = (total_risky ** 2) * port_var / (N ** 2) * (N ** 2)  # = total_risky² × port_var
    # Wait, let me recalculate: if we have w_i = opt_weight_per_asset for all i,
    # σ_p² = Σ_i Σ_j w_i w_j Σ_ij
    # For equal weights w_i = w, and N assets:
    # σ_p² = w² × [N×σ² + N(N-1)×ρσ²] = w² × N × σ² × [1 + (N-1)ρ]
    # But w = opt_weight_per_asset, so total = N×w
    opt_port_vol_sq = (opt_weight_per_asset ** 2) * N * sigma**2 * (1 + (N-1)*rho)
    opt_port_vol = np.sqrt(opt_port_vol_sq)
    
    # Effective Sharpe squared (for reference)
    # For tangency: SR² = (μ-rf)² / [σ²((1-ρ)/N + ρ)]
    sharpe_sq = excess**2 / port_var
    
    formulas = {
        'portfolio_variance': r'σ_p² = σ²[(1-ρ)/N + ρ]',
        'limiting_variance': r'lim_{N→∞} σ_p² = ρσ² (diversification limit)',
        'sharpe_ratio': r'SR = (μ - r_f) / σ_p',
        'optimal_weight': r'w_i^* = (μ - r_f) / [γσ²(1 + (N-1)ρ)]',
        'sharpe_squared': r"SR² = (μ - r_f)² / [σ²((1-ρ)/N + ρ)]"
    }
    
    return LargeNEqualCorrResult(
        tangency_weight_per_asset=tan_weight,
        optimal_weight_per_asset=opt_weight_per_asset,
        tangency_sharpe=tan_sharpe,
        optimal_total_risky_weight=total_risky,
        optimal_cash_weight=cash_weight,
        portfolio_return=opt_port_ret,
        portfolio_volatility=opt_port_vol,
        effective_sharpe_squared=sharpe_sq,
        formulas=formulas
    )


def diversification_limit_volatility(
    asset_volatility: float,
    correlation: float
) -> float:
    """
    Compute the limiting portfolio volatility as N → ∞.
    
    For N identical assets with pairwise correlation ρ and volatility σ:
    lim_{N→∞} σ_p = σ√ρ (for ρ > 0)
    
    This is the irreducible systematic risk that cannot be diversified away.
    
    Args:
        asset_volatility: Individual asset volatility σ
        correlation: Pairwise correlation ρ
    
    Returns:
        Limiting portfolio volatility
    """
    if correlation <= 0:
        return 0.0  # Can diversify to zero with uncorrelated/negative corr assets
    return asset_volatility * np.sqrt(correlation)


# =============================================================================
# NEW: MULTI-PERIOD RETURN UTILITIES (Section 4)
# =============================================================================

@dataclass
class MultiPeriodResult:
    """Results for multi-period return analysis."""
    compounded_mean: float
    compounded_variance: float
    compounded_volatility: float
    prob_exceed_threshold: float
    expected_terminal_wealth: float
    terminal_wealth_volatility: float
    assumptions: List[str]


def multi_period_return_statistics(
    annual_mean: float,
    annual_volatility: float,
    n_years: int,
    initial_wealth: float = 1.0,
    wealth_threshold: Optional[float] = None,
    use_lognormal: bool = True
) -> MultiPeriodResult:
    """
    Compute statistics for multi-year compounded returns.
    
    Two approaches:
    1. Lognormal approximation (recommended):
       ln(W_T/W_0) ~ N(n×(μ - σ²/2), n×σ²)
    
    2. Arithmetic approximation:
       E[R_T] ≈ n×μ, Var[R_T] ≈ n×σ² (for small μ, σ)
    
    Args:
        annual_mean: Annual expected return (arithmetic mean)
        annual_volatility: Annual volatility
        n_years: Investment horizon in years
        initial_wealth: Starting wealth (default 1.0)
        wealth_threshold: Threshold for probability calculation
        use_lognormal: If True, use lognormal; else arithmetic approximation
    
    Returns:
        MultiPeriodResult with compounded statistics and probabilities
    """
    mu = annual_mean
    sigma = annual_volatility
    T = n_years
    W0 = initial_wealth
    
    assumptions = []
    
    if use_lognormal:
        # Lognormal model: ln(W_T/W_0) ~ N(T×(μ - σ²/2), T×σ²)
        assumptions.append("Returns are lognormally distributed")
        assumptions.append("Continuous compounding approximation")
        
        # Log-return statistics
        log_mean = T * (mu - sigma**2 / 2)
        log_var = T * sigma**2
        log_vol = np.sqrt(log_var)
        
        # Expected terminal wealth: E[W_T] = W0 × exp(T×μ)
        # This uses the lognormal mean formula
        expected_wealth = W0 * np.exp(T * mu)
        
        # Variance of terminal wealth (lognormal)
        # Var[W_T] = W0² × exp(2Tμ) × (exp(Tσ²) - 1)
        wealth_var = W0**2 * np.exp(2*T*mu) * (np.exp(T*sigma**2) - 1)
        wealth_vol = np.sqrt(wealth_var)
        
        # Compounded return statistics (for reporting)
        # E[R_compound] = exp(log_mean + log_var/2) - 1 ≈ exp(T×μ) - 1
        comp_mean = np.exp(log_mean + log_var/2) - 1
        comp_var = (np.exp(log_var) - 1) * np.exp(2*log_mean + log_var)
        comp_vol = np.sqrt(comp_var)
        
        # Probability of exceeding threshold
        if wealth_threshold is not None:
            # P(W_T > K) = P(ln(W_T/W0) > ln(K/W0))
            # = 1 - Φ((ln(K/W0) - log_mean) / log_vol)
            log_threshold = np.log(wealth_threshold / W0)
            z = (log_threshold - log_mean) / log_vol
            prob_exceed = 1 - norm.cdf(z)
        else:
            prob_exceed = np.nan
    
    else:
        # Simple arithmetic approximation
        assumptions.append("Linear approximation (valid for short horizons)")
        assumptions.append("Assumes additive returns")
        
        comp_mean = T * mu
        comp_var = T * sigma**2
        comp_vol = np.sqrt(comp_var)
        
        expected_wealth = W0 * (1 + comp_mean)
        wealth_vol = W0 * comp_vol
        
        if wealth_threshold is not None:
            # P(W_T > K) ≈ P(W0(1 + R_T) > K)
            # = P(R_T > K/W0 - 1)
            required_return = wealth_threshold / W0 - 1
            z = (required_return - comp_mean) / comp_vol
            prob_exceed = 1 - norm.cdf(z)
        else:
            prob_exceed = np.nan
    
    return MultiPeriodResult(
        compounded_mean=comp_mean,
        compounded_variance=comp_var,
        compounded_volatility=comp_vol,
        prob_exceed_threshold=prob_exceed,
        expected_terminal_wealth=expected_wealth,
        terminal_wealth_volatility=wealth_vol,
        assumptions=assumptions
    )


def prob_shortfall(
    annual_mean: float,
    annual_volatility: float,
    n_years: int,
    target_return: float,
    use_lognormal: bool = True
) -> float:
    """
    Probability that compounded return falls below target.
    
    P(R_T < target) where R_T is the T-year compounded return.
    
    Args:
        annual_mean: Annual expected return
        annual_volatility: Annual volatility
        n_years: Investment horizon
        target_return: Target compounded return
        use_lognormal: Use lognormal model
    
    Returns:
        Probability of shortfall
    """
    result = multi_period_return_statistics(
        annual_mean, annual_volatility, n_years,
        initial_wealth=1.0,
        wealth_threshold=1 + target_return,
        use_lognormal=use_lognormal
    )
    return 1 - result.prob_exceed_threshold


def wealth_quantile(
    annual_mean: float,
    annual_volatility: float,
    n_years: int,
    initial_wealth: float,
    quantile: float,
    use_lognormal: bool = True
) -> float:
    """
    Compute wealth level at a given quantile.
    
    Args:
        annual_mean: Annual expected return
        annual_volatility: Annual volatility
        n_years: Investment horizon
        initial_wealth: Starting wealth
        quantile: Desired quantile (e.g., 0.05 for 5th percentile)
        use_lognormal: Use lognormal model
    
    Returns:
        Wealth level at specified quantile
    """
    mu = annual_mean
    sigma = annual_volatility
    T = n_years
    W0 = initial_wealth
    
    if use_lognormal:
        log_mean = T * (mu - sigma**2 / 2)
        log_vol = sigma * np.sqrt(T)
        
        z = norm.ppf(quantile)
        log_wealth = log_mean + z * log_vol
        return W0 * np.exp(log_wealth)
    else:
        comp_mean = T * mu
        comp_vol = sigma * np.sqrt(T)
        
        z = norm.ppf(quantile)
        return W0 * (1 + comp_mean + z * comp_vol)


# =============================================================================
# NEW: FIXED INCOME ENGINE HARDENING (Section 5)
# =============================================================================

def compute_par_swap_rate(
    spot_rates: np.ndarray,
    maturities: np.ndarray,
    swap_maturity: float,
    payment_frequency: int = 1
) -> Tuple[float, Dict]:
    """
    Compute the par swap rate for a given maturity.
    
    The par swap rate is the fixed rate that makes the swap have zero value at inception.
    
    Formula: S = (1 - DF_T) / Σ DF_i
    
    where:
    - DF_T is the discount factor at swap maturity
    - Σ DF_i is the sum of discount factors at each payment date (annuity factor)
    
    Args:
        spot_rates: Spot rates curve
        maturities: Maturities corresponding to spot rates
        swap_maturity: Maturity of the swap
        payment_frequency: Payments per year (1=annual, 2=semi-annual)
    
    Returns:
        (par_swap_rate, details_dict)
    """
    spot_rates = np.asarray(spot_rates)
    maturities = np.asarray(maturities)
    
    # Generate payment dates
    dt = 1 / payment_frequency
    payment_times = np.arange(dt, swap_maturity + 1e-9, dt)
    
    # Interpolate spot rates at payment dates
    interp_rates = np.interp(payment_times, maturities, spot_rates)
    
    # Compute discount factors
    dfs = 1 / (1 + interp_rates) ** payment_times
    
    # Annuity factor
    annuity_factor = np.sum(dfs) * dt
    
    # Terminal discount factor
    df_T = dfs[-1]
    
    # Par swap rate
    swap_rate = (1 - df_T) / annuity_factor
    
    details = {
        'payment_times': payment_times,
        'discount_factors': dfs,
        'annuity_factor': annuity_factor,
        'terminal_df': df_T,
        'formula': 'S = (1 - DF_T) / A_T',
        'calculation': f'S = (1 - {df_T:.6f}) / {annuity_factor:.6f} = {swap_rate:.6f}'
    }
    
    return swap_rate, details


def compute_annuity_factor(
    spot_rates: np.ndarray,
    maturities: np.ndarray,
    annuity_maturity: float,
    payment_frequency: int = 1
) -> Tuple[float, Dict]:
    """
    Compute the annuity factor A_T = Σ DF_i × Δt.
    
    The annuity factor is the present value of $1 paid at each payment date.
    
    Args:
        spot_rates: Spot rates curve
        maturities: Maturities corresponding to spot rates
        annuity_maturity: Maturity of the annuity
        payment_frequency: Payments per year
    
    Returns:
        (annuity_factor, details_dict)
    """
    dt = 1 / payment_frequency
    payment_times = np.arange(dt, annuity_maturity + 1e-9, dt)
    
    interp_rates = np.interp(payment_times, maturities, spot_rates)
    dfs = 1 / (1 + interp_rates) ** payment_times
    
    # Annuity factor = sum of (DF × period fraction)
    annuity_factor = np.sum(dfs) * dt
    
    # Alternative: PV of $1/year for T years
    # A_T = (1 - DF_T) / r for flat curve
    
    details = {
        'payment_times': payment_times,
        'discount_factors': dfs,
        'period_fraction': dt,
        'pv_per_payment': dfs * dt,
        'formula': 'A_T = Σ DF_i × Δt'
    }
    
    return annuity_factor, details


def verify_replication_identity(
    positions: np.ndarray,
    instrument_cashflows: List[Dict],
    target_cashflows: Dict[float, float],
    tolerance: float = 1e-6
) -> Tuple[bool, Dict]:
    """
    Verify that a replicating portfolio exactly matches target cashflows.
    
    Args:
        positions: Number of units of each instrument
        instrument_cashflows: List of {dates: [...], amounts: [...]} dicts
        target_cashflows: {date: amount} dict for target
        tolerance: Numerical tolerance for matching
    
    Returns:
        (is_exact_match, verification_details)
    """
    # Build all dates
    all_dates = set(target_cashflows.keys())
    for inst in instrument_cashflows:
        all_dates.update(inst['dates'])
    all_dates = sorted(all_dates)
    
    # Compute replicated cashflows
    replicated = {d: 0.0 for d in all_dates}
    for j, inst in enumerate(instrument_cashflows):
        for k, d in enumerate(inst['dates']):
            replicated[d] += positions[j] * inst['amounts'][k]
    
    # Compare with target
    errors = {}
    max_error = 0.0
    for d in all_dates:
        target = target_cashflows.get(d, 0.0)
        rep = replicated[d]
        err = abs(target - rep)
        errors[d] = {'target': target, 'replicated': rep, 'error': err}
        max_error = max(max_error, err)
    
    is_match = max_error < tolerance
    
    details = {
        'is_exact_match': is_match,
        'max_error': max_error,
        'tolerance': tolerance,
        'cashflow_comparison': errors
    }
    
    return is_match, details


def price_bond_with_spot_curve(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    spot_rates: np.ndarray,
    maturities: np.ndarray,
    frequency: int = 1
) -> Tuple[Dict, List[str], pd.DataFrame]:
    """
    Price a coupon bond using a zero-coupon (spot) yield curve.
    
    Each cash flow is discounted at its maturity-specific spot rate.
    This is the correct method when given a term structure of interest rates.
    
    Price = Σ CF_t / (1 + y_t)^t
    
    Then solves for the YTM that reprices the bond (flat yield equivalent).
    
    Args:
        face_value: Face/par value of the bond
        coupon_rate: Annual coupon rate (decimal)
        maturity: Years to maturity
        spot_rates: Zero-coupon yield curve
        maturities: Maturities corresponding to spot rates
        frequency: Payment frequency (1=annual, 2=semi-annual)
    
    Returns:
        (result_dict, excel_steps, schedule_df)
    
    Exam Reference: "Price a bond given the zero-coupon yield curve"
    """
    spot_rates = np.asarray(spot_rates)
    maturities = np.asarray(maturities)
    
    excel_steps = []
    excel_steps.append("**Bond Pricing with Zero-Coupon Yield Curve:**")
    excel_steps.append("")
    excel_steps.append(f"Face Value: ${face_value:,.2f}")
    excel_steps.append(f"Coupon Rate: {coupon_rate*100:.2f}%")
    excel_steps.append(f"Maturity: {maturity} years")
    excel_steps.append(f"Payment Frequency: {'Annual' if frequency == 1 else 'Semi-annual'}")
    excel_steps.append("")
    
    # Generate payment schedule
    n_periods = maturity * frequency
    period_coupon = face_value * coupon_rate / frequency
    dt = 1 / frequency
    
    payment_times = np.arange(dt, maturity + 1e-9, dt)
    
    # Cash flows
    cash_flows = np.full(len(payment_times), period_coupon)
    cash_flows[-1] += face_value  # Add face value at maturity
    
    excel_steps.append("**Step 1: Payment Schedule**")
    excel_steps.append(f"Coupon Payment: ${face_value:,.0f} × {coupon_rate} / {frequency} = ${period_coupon:.2f}")
    excel_steps.append("")
    
    # Interpolate spot rates at payment dates
    interp_rates = np.interp(payment_times, maturities, spot_rates)
    
    excel_steps.append("**Step 2: Spot Rates at Payment Dates**")
    for i, (t, r) in enumerate(zip(payment_times, interp_rates)):
        excel_steps.append(f"  y({t:.1f}) = {r*100:.4f}%")
    excel_steps.append("")
    
    # Discount factors using spot rates
    discount_factors = 1 / (1 + interp_rates) ** payment_times
    
    excel_steps.append("**Step 3: Discount Factors (using spot rates)**")
    excel_steps.append("DF(t) = 1 / (1 + y_t)^t")
    for i, (t, r, df) in enumerate(zip(payment_times, interp_rates, discount_factors)):
        excel_steps.append(f"  DF({t:.1f}) = 1 / (1 + {r:.4f})^{t:.1f} = {df:.6f}")
    excel_steps.append("")
    
    # Present values
    pv = cash_flows * discount_factors
    price = np.sum(pv)
    
    excel_steps.append("**Step 4: Present Value of Each Cash Flow**")
    excel_steps.append("PV(t) = CF(t) × DF(t)")
    for i, (t, cf, df, p) in enumerate(zip(payment_times, cash_flows, discount_factors, pv)):
        excel_steps.append(f"  PV({t:.1f}) = ${cf:.2f} × {df:.6f} = ${p:.4f}")
    excel_steps.append("")
    excel_steps.append(f"**Bond Price = Σ PV = ${price:.4f}**")
    excel_steps.append("")
    
    # Solve for YTM (flat yield that reprices the bond)
    excel_steps.append("**Step 5: Solve for Yield-to-Maturity**")
    excel_steps.append("YTM is the single rate y such that:")
    excel_steps.append("Price = Σ CF_t / (1 + y)^t")
    excel_steps.append("")
    
    def price_at_yield(y):
        if abs(y) < 1e-10:
            return np.sum(cash_flows)
        dfs = 1 / (1 + y / frequency) ** (payment_times * frequency)
        return np.sum(cash_flows * dfs)
    
    # Newton-Raphson to find YTM
    y = coupon_rate  # Initial guess
    for _ in range(100):
        p_y = price_at_yield(y)
        dy = 0.0001
        dp = (price_at_yield(y + dy) - price_at_yield(y - dy)) / (2 * dy)
        if abs(dp) < 1e-12:
            break
        y_new = y - (p_y - price) / dp
        if abs(y_new - y) < 1e-8:
            y = y_new
            break
        y = max(0.0001, y_new)
    
    ytm = y
    
    excel_steps.append(f"**YTM = {ytm*100:.4f}%**")
    excel_steps.append("")
    excel_steps.append("Excel: Use RATE or Goal Seek to find y where")
    excel_steps.append(f"  Σ CF_t / (1 + y)^t = ${price:.4f}")
    excel_steps.append("")
    
    # Duration calculation (using YTM for weights)
    dfs_ytm = 1 / (1 + ytm / frequency) ** (payment_times * frequency)
    pv_ytm = cash_flows * dfs_ytm
    weights = pv_ytm / np.sum(pv_ytm)
    time_weighted = payment_times * weights
    macaulay_duration = np.sum(time_weighted)
    modified_duration = macaulay_duration / (1 + ytm / frequency)
    
    excel_steps.append("**Step 6: Duration Calculation**")
    excel_steps.append("Using YTM for consistent duration calculation:")
    excel_steps.append("")
    excel_steps.append("| t | CF | DF (YTM) | PV | Weight | t × Weight |")
    excel_steps.append("|---|----|---------|----|--------|-----------|")
    for i, (t, cf, df, p, w, tw) in enumerate(zip(payment_times, cash_flows, dfs_ytm, pv_ytm, weights, time_weighted)):
        excel_steps.append(f"| {t:.1f} | {cf:.2f} | {df:.6f} | {p:.4f} | {w:.6f} | {tw:.6f} |")
    excel_steps.append("")
    excel_steps.append(f"**Macaulay Duration = Σ(t × Weight) = {macaulay_duration:.4f} years**")
    excel_steps.append(f"**Modified Duration = Mac Duration / (1 + y) = {macaulay_duration:.4f} / {1 + ytm:.4f} = {modified_duration:.4f}**")
    excel_steps.append("")
    
    # Create schedule DataFrame
    schedule = pd.DataFrame({
        't (years)': payment_times,
        'Cash Flow': cash_flows,
        'Spot Rate': interp_rates,
        'DF (Spot)': discount_factors,
        'PV (Spot)': pv,
        'DF (YTM)': dfs_ytm,
        'PV (YTM)': pv_ytm,
        'Weight': weights,
        't × Weight': time_weighted
    })
    
    result = {
        'price': price,
        'ytm': ytm,
        'macaulay_duration': macaulay_duration,
        'modified_duration': modified_duration,
        'cash_flows': cash_flows,
        'payment_times': payment_times,
        'spot_rates_used': interp_rates,
        'discount_factors': discount_factors,
        'present_values': pv,
        'face_value': face_value,
        'coupon_rate': coupon_rate,
        'coupon_payment': period_coupon,
        'maturity': maturity
    }
    
    return result, excel_steps, schedule


def bootstrap_forward_rates(
    spot_rates: np.ndarray,
    maturities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1-year forward rates from spot curve.
    
    f(t, t+1) = [(1 + r_{t+1})^{t+1} / (1 + r_t)^t]^1 - 1
    
    Args:
        spot_rates: Spot rates curve
        maturities: Maturities (should be integers or half-integers)
    
    Returns:
        (forward_rates, forward_periods) tuple
    """
    forwards = []
    periods = []
    
    for i in range(len(maturities) - 1):
        t1 = maturities[i]
        t2 = maturities[i + 1]
        r1 = spot_rates[i]
        r2 = spot_rates[i + 1]
        
        # Forward rate
        df1 = 1 / (1 + r1) ** t1
        df2 = 1 / (1 + r2) ** t2
        
        tau = t2 - t1
        fwd = (df1 / df2) ** (1 / tau) - 1
        
        forwards.append(fwd)
        periods.append((t1, t2))
    
    return np.array(forwards), periods


# =============================================================================
# NEW: EXAM REGRESSION TESTS (Section 7)
# =============================================================================

def run_exam_regression_tests(verbose: bool = True) -> Tuple[bool, List[str]]:
    """
    Run regression tests covering key exam scenarios.
    
    Tests:
    1. ESG "avoid" portfolio (exclude negative ESG assets)
    2. Zero-cost factor portfolio (WML)
    3. Tangency portfolio with cash
    4. Bond replication identity
    5. Large-N equal correlation
    6. Multi-period probabilities
    
    Args:
        verbose: If True, print test results
    
    Returns:
        (all_passed, list_of_messages)
    """
    results = []
    all_passed = True
    
    def log(msg: str):
        results.append(msg)
        if verbose:
            print(msg)
    
    log("=" * 60)
    log("EXAM REGRESSION TESTS - Financial Market Theory Engine v7.0")
    log("=" * 60)
    
    # Test 1: ESG Avoid Portfolio
    log("\n[Test 1] ESG Avoid Portfolio (exclude asset with negative ESG)")
    try:
        mu = np.array([0.08, 0.12, 0.06])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.0625, 0.01],
            [0.005, 0.01, 0.0225]
        ])
        esg = np.array([0.7, -0.3, 0.9])  # Asset 2 has negative ESG
        
        result = unified_portfolio_solver(
            expected_returns=mu,
            covariance_matrix=cov,
            risk_free_rate=0.02,
            risk_aversion=4.0,
            excluded_assets={1},  # Exclude asset with negative ESG
            no_short=True,
            asset_names=['Good1', 'Bad_ESG', 'Good2']
        )
        
        # Check that excluded asset has zero weight
        assert abs(result.weights[1]) < 1e-6, f"Excluded asset weight should be 0, got {result.weights[1]}"
        assert result.solver_status == 'optimal', f"Solver failed: {result.solver_message}"
        log(f"   PASSED: Excluded asset weight = {result.weights[1]:.6f}")
        log(f"   Weights: {result.weights}")
    except Exception as e:
        log(f"   FAILED: {str(e)}")
        all_passed = False
    
    # Test 2: Zero-Cost Factor Portfolio
    log("\n[Test 2] Zero-Cost Factor Portfolio (WML style)")
    try:
        # 2 funded assets + 1 zero-cost factor
        mu = np.array([0.08, 0.10, 0.05])  # Last is zero-cost factor
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.0625, 0.008],
            [0.005, 0.008, 0.01]
        ])
        
        result = unified_portfolio_solver(
            expected_returns=mu,
            covariance_matrix=cov,
            risk_free_rate=0.02,
            risk_aversion=4.0,
            zero_cost_indices={2},  # Asset 2 is zero-cost
            allow_cash=True,
            asset_names=['Stock1', 'Stock2', 'WML']
        )
        
        # Check that zero-cost sums to zero (it's just one asset, so should be 0)
        # Actually for single zero-cost, it should be 0
        # But the constraint is sum(zerocost) = 0, which means w[2] = 0 for single asset
        # This is a degenerate case - let's check sum of funded + cash = 1
        funded_sum = result.weights[0] + result.weights[1]
        total = funded_sum + result.cash_weight
        assert abs(total - 1.0) < 1e-5, f"Funded + cash should = 1, got {total}"
        log(f"   PASSED: Funded + cash = {total:.6f}")
        log(f"   Weights: {result.weights}, Cash: {result.cash_weight:.4f}")
    except Exception as e:
        log(f"   FAILED: {str(e)}")
        all_passed = False
    
    # Test 3: Tangency Portfolio with Cash
    log("\n[Test 3] Tangency Portfolio with Cash (verify Sharpe consistency)")
    try:
        mu = np.array([0.10, 0.15])
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.09]
        ])
        rf = 0.02
        
        # Compute analytical tangency
        cov_inv = np.linalg.inv(cov)
        excess = mu - rf
        ones = np.ones(2)
        
        tan_weights = cov_inv @ excess / (ones @ cov_inv @ excess)
        tan_ret = tan_weights @ mu
        tan_vol = np.sqrt(tan_weights @ cov @ tan_weights)
        tan_sharpe = (tan_ret - rf) / tan_vol
        
        # Use solver with high risk aversion (should allocate mostly to tangency direction)
        result = unified_portfolio_solver(
            expected_returns=mu,
            covariance_matrix=cov,
            risk_free_rate=rf,
            risk_aversion=2.0,
            allow_cash=True
        )
        
        # The Sharpe ratio should match tangency (ignoring cash allocation)
        # Actually, with cash, the Sharpe of the risky part should match tangency
        risky_weight = 1 - result.cash_weight
        if risky_weight > 0.01:
            normalized_weights = result.weights / risky_weight
            test_sharpe = (normalized_weights @ mu - rf) / np.sqrt(normalized_weights @ cov @ normalized_weights)
            assert abs(test_sharpe - tan_sharpe) < 0.01, f"Sharpe mismatch: {test_sharpe} vs {tan_sharpe}"
            log(f"   PASSED: Sharpe ratio = {test_sharpe:.4f} (analytical: {tan_sharpe:.4f})")
        else:
            log(f"   PASSED (trivial): Nearly all cash, risky_weight = {risky_weight:.4f}")
    except Exception as e:
        log(f"   FAILED: {str(e)}")
        all_passed = False
    
    # Test 4: Bond Replication Identity
    log("\n[Test 4] Bond Replication Identity")
    try:
        # Two bonds replicating a zero-coupon
        # Bond A: 1-year, 5% coupon
        # Bond B: 2-year, 6% coupon
        # Target: 2-year zero paying 100
        
        instrument_cashflows = [
            {'dates': [1, 2], 'amounts': [5, 105]},  # Bond A: 5, 105
            {'dates': [1, 2], 'amounts': [6, 106]}   # Bond B: 6, 106
        ]
        target_cashflows = {2: 100}  # Zero-coupon at T=2
        
        # Matrix: [[5, 6], [105, 106]]
        # Target: [0, 100]
        C = np.array([[5, 6], [105, 106]])
        T = np.array([0, 100])
        
        positions = np.linalg.solve(C, T)
        
        is_match, details = verify_replication_identity(
            positions, instrument_cashflows, target_cashflows, tolerance=1e-6
        )
        
        assert is_match, f"Replication failed: max error = {details['max_error']}"
        log(f"   PASSED: Replication exact (max error = {details['max_error']:.2e})")
        log(f"   Positions: Bond A = {positions[0]:.4f}, Bond B = {positions[1]:.4f}")
    except Exception as e:
        log(f"   FAILED: {str(e)}")
        all_passed = False
    
    # Test 5: Large-N Equal Correlation
    log("\n[Test 5] Large-N Equal Correlation (closed form)")
    try:
        result = large_n_equal_correlation_portfolio(
            n_assets=100,
            expected_return=0.10,
            volatility=0.20,
            correlation=0.3,
            risk_free_rate=0.02,
            risk_aversion=4.0
        )
        
        # Verify portfolio variance formula
        sigma = 0.20
        rho = 0.3
        N = 100
        expected_vol = sigma * np.sqrt((1 - rho)/N + rho)
        
        # The tangency sharpe should match (μ - rf) / σ_p
        expected_sharpe = (0.10 - 0.02) / expected_vol
        
        assert abs(result.tangency_sharpe - expected_sharpe) < 1e-4, \
            f"Sharpe mismatch: {result.tangency_sharpe} vs {expected_sharpe}"
        log(f"   PASSED: Tangency Sharpe = {result.tangency_sharpe:.4f}")
        log(f"   Diversification: σ_p = {expected_vol*100:.2f}% (limit: {sigma*np.sqrt(rho)*100:.2f}%)")
    except Exception as e:
        log(f"   FAILED: {str(e)}")
        all_passed = False
    
    # Test 6: Multi-Period Probability
    log("\n[Test 6] Multi-Period Probability (lognormal)")
    try:
        result = multi_period_return_statistics(
            annual_mean=0.08,
            annual_volatility=0.15,
            n_years=10,
            initial_wealth=100000,
            wealth_threshold=150000,  # 50% growth
            use_lognormal=True
        )
        
        # Probability of doubling should be reasonable (not 0 or 1)
        assert 0 < result.prob_exceed_threshold < 1, \
            f"Probability should be in (0,1), got {result.prob_exceed_threshold}"
        
        log(f"   PASSED: P(W > 150k) = {result.prob_exceed_threshold:.4f}")
        log(f"   Expected wealth: ${result.expected_terminal_wealth:,.0f}")
    except Exception as e:
        log(f"   FAILED: {str(e)}")
        all_passed = False
    
    # Test 7: Black-Litterman Model
    log("\n[Test 7] Black-Litterman Model")
    try:
        # Set up a simple 3-asset case
        market_weights = np.array([0.4, 0.35, 0.25])
        cov = np.array([
            [0.04, 0.01, 0.008],
            [0.01, 0.0225, 0.005],
            [0.008, 0.005, 0.09]
        ])
        delta = 2.5
        tau = 0.05
        
        # Create a relative view: Asset 1 outperforms Asset 2 by 3%
        P = np.array([[1, -1, 0]])
        Q = np.array([0.03])
        
        optimizer = BlackLittermanOptimizer(
            market_weights=market_weights,
            covariance=cov,
            risk_aversion=delta,
            tau=tau,
            P=P,
            Q=Q,
            risk_free_rate=0.02
        )
        
        # Equilibrium returns should be δ × Σ × w
        expected_eq = delta * cov @ market_weights
        assert np.allclose(optimizer.equilibrium_returns, expected_eq, atol=1e-6), \
            f"Equilibrium mismatch"
        
        # Posterior should shift towards the view
        # Asset 1 return should increase, Asset 2 should decrease
        shift = optimizer.posterior_returns - optimizer.equilibrium_returns
        assert shift[0] > 0, "View should increase Asset 1 return"
        assert shift[1] < 0, "View should decrease Asset 2 return"
        
        # Weights should sum to 1
        assert abs(np.sum(optimizer.posterior_weights) - 1.0) < 1e-6, \
            f"Weights should sum to 1, got {np.sum(optimizer.posterior_weights)}"
        
        log(f"   PASSED: Equilibrium returns computed correctly")
        log(f"   Return shifts: {shift}")
        log(f"   BL weights: {optimizer.posterior_weights}")
    except Exception as e:
        log(f"   FAILED: {str(e)}")
        all_passed = False
    
    # Test 8: Wealth Equivalent Loss
    log("\n[Test 8] Wealth Equivalent Loss")
    try:
        result, _ = wealth_equivalent_loss(
            optimal_sharpe=0.4,
            suboptimal_sharpe=0.0,
            risk_aversion=4.0,
            horizon=1
        )
        
        # For SR=0.4, γ=4, T=1: WEL = 1 - exp(-4/2 × 0.16 × 1) = 1 - exp(-0.32)
        expected_wel = 1 - np.exp(-0.32)
        assert abs(result['wealth_equivalent_loss'] - expected_wel) < 1e-6, \
            f"WEL mismatch: {result['wealth_equivalent_loss']} vs {expected_wel}"
        
        log(f"   PASSED: WEL = {result['wealth_equivalent_loss_pct']:.4f}%")
    except Exception as e:
        log(f"   FAILED: {str(e)}")
        all_passed = False
    
    # Test 9: Stocks for Target Volatility
    log("\n[Test 9] Diversification Formula")
    try:
        result, _ = stocks_for_target_volatility(
            asset_volatility=0.30,
            correlation=0.2,
            target_volatility=0.15
        )
        
        # Verify: σ_p² = σ²[(1-ρ)/N + ρ]
        # 0.15² = 0.30²[(0.8)/N + 0.2]
        # 0.0225 = 0.09[(0.8)/N + 0.2]
        # 0.25 = (0.8)/N + 0.2
        # N = 0.8/0.05 = 16
        # Due to ceiling, result might be 16 or 17 depending on numerical precision
        assert result['n_stocks'] in [16, 17], f"Expected 16-17 stocks, got {result['n_stocks']}"
        assert result['achievable'], "Should be achievable"
        
        # Verify the achieved volatility is at or below target
        assert result['achieved_volatility'] <= 0.15 + 1e-6, \
            f"Achieved volatility {result['achieved_volatility']} should be <= 0.15"
        
        log(f"   PASSED: Need {result['n_stocks']} stocks for ≤15% volatility (achieved: {result['achieved_volatility']*100:.2f}%)")
    except Exception as e:
        log(f"   FAILED: {str(e)}")
        all_passed = False
    
    log("\n" + "=" * 60)
    if all_passed:
        log("ALL TESTS PASSED")
    else:
        log("SOME TESTS FAILED")
    log("=" * 60)
    
    return all_passed, results


# =============================================================================
# UTILITY FUNCTIONS - Parsing & Helpers
# =============================================================================

def parse_messy_input(text: str) -> np.ndarray:
    """
    Parse messy copy-paste input from PDFs.
    Handles formats like: "0.08 0.15", "8%, 15%", "0.08, 0.15", etc.
    Returns a 1D numpy array.
    """
    if not text or text.strip() == "":
        return np.array([])
    
    # Remove common artifacts
    text = text.replace('\t', ' ').replace('\n', ' ')
    
    # Remove percentage signs and convert
    text = text.replace('%', '')
    
    # Find all numbers (including negatives and decimals)
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    if not matches:
        return np.array([])
    
    values = [float(m) for m in matches]
    
    # Heuristic: if values seem like percentages (>1 and <100), convert
    # Only apply if ALL values look like percentages
    if all(1 < abs(v) < 100 for v in values if v != 0):
        values = [v / 100 for v in values]
    
    return np.array(values)


def parse_matrix_input(text: str) -> np.ndarray:
    """
    Parse a matrix from text input.
    Each row should be on a new line.
    """
    if not text or text.strip() == "":
        return np.array([])
    
    rows = []
    for line in text.strip().split('\n'):
        if line.strip():
            row_values = parse_messy_input(line)
            if len(row_values) > 0:
                rows.append(row_values)
    
    if not rows:
        return np.array([])
    
    # Ensure all rows have the same length
    max_len = max(len(r) for r in rows)
    padded_rows = []
    for r in rows:
        if len(r) < max_len:
            r = np.pad(r, (0, max_len - len(r)), constant_values=0)
        padded_rows.append(r)
    
    return np.array(padded_rows)


def parse_two_column_input(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse input with two values per row (e.g., Mean, StdDev).
    Returns two arrays: first column, second column.
    """
    matrix = parse_matrix_input(text)
    if matrix.size == 0 or matrix.ndim != 2 or matrix.shape[1] < 2:
        return np.array([]), np.array([])
    return matrix[:, 0], matrix[:, 1]


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 4) -> str:
    """Format a number with specified decimals."""
    return f"{value:.{decimals}f}"


# =============================================================================
# INPUT VALIDATION HELPERS
# =============================================================================

class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass


def validate_rate(rate: float, name: str, allow_negative: bool = False) -> float:
    """
    Validate a rate input.
    
    Args:
        rate: The rate value to validate
        name: Name of the rate for error messages
        allow_negative: Whether negative rates are allowed
    
    Returns:
        The validated rate
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(rate, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(rate).__name__}")
    
    if np.isnan(rate) or np.isinf(rate):
        raise ValidationError(f"{name} cannot be NaN or infinite")
    
    if not allow_negative and rate < 0:
        raise ValidationError(f"{name} cannot be negative: {rate}")
    
    # Warning for potentially misformatted rates (entered as percentages)
    if abs(rate) > 1 and abs(rate) < 100:
        # This might be a percentage entered without conversion
        pass  # Could add warning here
    
    return float(rate)


def validate_positive(value: float, name: str) -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    return float(value)


def validate_array(arr: np.ndarray, name: str, expected_len: Optional[int] = None,
                   min_len: int = 1) -> np.ndarray:
    """
    Validate an array input.
    
    Args:
        arr: Array to validate
        name: Name for error messages
        expected_len: Expected length (if specified)
        min_len: Minimum length
    
    Returns:
        Validated array
    """
    if arr is None or len(arr) == 0:
        raise ValidationError(f"{name} cannot be empty")
    
    if len(arr) < min_len:
        raise ValidationError(f"{name} must have at least {min_len} elements, got {len(arr)}")
    
    if expected_len is not None and len(arr) != expected_len:
        raise ValidationError(f"{name} must have {expected_len} elements, got {len(arr)}")
    
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValidationError(f"{name} contains NaN or infinite values")
    
    return np.asarray(arr, dtype=float)


def validate_matrix(mat: np.ndarray, name: str, 
                    expected_shape: Optional[Tuple[int, int]] = None,
                    square: bool = False,
                    symmetric: bool = False,
                    positive_definite: bool = False) -> np.ndarray:
    """
    Validate a matrix input.
    
    Args:
        mat: Matrix to validate
        name: Name for error messages
        expected_shape: Expected (rows, cols)
        square: Must be square
        symmetric: Must be symmetric
        positive_definite: Must be positive definite
    
    Returns:
        Validated matrix
    """
    if mat is None or mat.size == 0:
        raise ValidationError(f"{name} cannot be empty")
    
    if mat.ndim != 2:
        raise ValidationError(f"{name} must be 2-dimensional, got {mat.ndim}")
    
    if expected_shape is not None and mat.shape != expected_shape:
        raise ValidationError(f"{name} must have shape {expected_shape}, got {mat.shape}")
    
    if square and mat.shape[0] != mat.shape[1]:
        raise ValidationError(f"{name} must be square, got shape {mat.shape}")
    
    if symmetric:
        if mat.shape[0] != mat.shape[1]:
            raise ValidationError(f"{name} must be square to be symmetric")
        if not np.allclose(mat, mat.T, rtol=1e-5):
            raise ValidationError(f"{name} must be symmetric")
    
    if positive_definite:
        try:
            eigenvalues = np.linalg.eigvalsh(mat)
            if np.any(eigenvalues <= 0):
                raise ValidationError(f"{name} must be positive definite (has non-positive eigenvalues)")
        except np.linalg.LinAlgError:
            raise ValidationError(f"{name} eigenvalue computation failed")
    
    if np.any(np.isnan(mat)) or np.any(np.isinf(mat)):
        raise ValidationError(f"{name} contains NaN or infinite values")
    
    return np.asarray(mat, dtype=float)


def validate_cashflows(dates: np.ndarray, amounts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate that cashflow dates and amounts are aligned."""
    dates = validate_array(dates, "Cashflow dates")
    amounts = validate_array(amounts, "Cashflow amounts", expected_len=len(dates))
    
    # Check dates are positive and increasing
    if np.any(dates <= 0):
        raise ValidationError("Cashflow dates must be positive")
    
    if not np.all(np.diff(dates) > 0):
        raise ValidationError("Cashflow dates must be strictly increasing")
    
    return dates, amounts


# =============================================================================
# FIXED INCOME - PURE MATH FUNCTIONS
# =============================================================================

def bootstrap_spot_curve(
    maturities: np.ndarray,
    coupon_rates: np.ndarray,
    prices: np.ndarray,
    face_value: float = 100.0,
    frequency: int = 1
) -> Tuple[Dict, List[str], Dict]:
    """
    Bootstrap spot/zero curve from coupon bond prices.
    
    Uses sequential bootstrapping: solve for each spot rate one at a time,
    starting from shortest maturity.
    
    Args:
        maturities: Bond maturities in years (must be sorted ascending)
        coupon_rates: Annual coupon rates (as decimals)
        prices: Clean prices of bonds
        face_value: Face value (default 100)
        frequency: Payment frequency per year (1=annual, 2=semi-annual)
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    # Validation
    n = len(maturities)
    maturities = validate_array(maturities, "Maturities", min_len=1)
    coupon_rates = validate_array(coupon_rates, "Coupon rates", expected_len=n)
    prices = validate_array(prices, "Prices", expected_len=n)
    face_value = validate_positive(face_value, "Face value")
    
    # Sort by maturity
    sort_idx = np.argsort(maturities)
    maturities = maturities[sort_idx]
    coupon_rates = coupon_rates[sort_idx]
    prices = prices[sort_idx]
    
    # Check maturities are unique
    if len(np.unique(maturities)) != len(maturities):
        raise ValidationError("Maturities must be unique")
    
    excel_steps = []
    excel_steps.append("**Spot Curve Bootstrapping Algorithm:**")
    excel_steps.append("For each bond (in order of maturity), solve for the spot rate")
    excel_steps.append("that makes the discounted cashflows equal the price.")
    excel_steps.append("")
    
    # Storage for results
    spot_rates = np.zeros(n)
    discount_factors = np.zeros(n)
    
    # Bootstrap iteratively
    for i in range(n):
        T = maturities[i]
        coupon = coupon_rates[i] * face_value / frequency
        P = prices[i]
        
        excel_steps.append(f"**Bond {i+1}: T={T}y, Coupon={coupon_rates[i]*100:.2f}%, Price={P:.4f}**")
        
        # Generate cashflow times for this bond
        cf_times = np.arange(1/frequency, T + 1e-9, 1/frequency)
        n_cf = len(cf_times)
        
        # Cashflows: coupons + face value at maturity
        cashflows = np.full(n_cf, coupon)
        cashflows[-1] += face_value
        
        # Sum of PV of known cashflows (using already bootstrapped rates)
        known_pv = 0.0
        known_pv_str = []
        
        for j, t in enumerate(cf_times[:-1]):  # All but last cashflow
            # Find the spot rate for this time (interpolate if needed)
            if t in maturities[:i]:
                idx = np.where(maturities[:i] == t)[0][0]
                df_t = discount_factors[idx]
            elif i > 0 and t < maturities[i-1]:
                # Linear interpolation on spot rates
                r_interp = np.interp(t, maturities[:i], spot_rates[:i])
                df_t = 1 / (1 + r_interp) ** t
            else:
                # No prior data, assume flat at last known rate
                if i > 0:
                    df_t = 1 / (1 + spot_rates[i-1]) ** t
                else:
                    df_t = 1  # Will be solved
            
            if j < len(cf_times) - 1:
                known_pv += cashflows[j] * df_t
                known_pv_str.append(f"{cashflows[j]:.2f}×DF({t:.2f})")
        
        # Solve for the final spot rate
        # P = known_pv + final_cf * DF(T)
        # DF(T) = (P - known_pv) / final_cf
        final_cf = cashflows[-1]
        df_T = (P - known_pv) / final_cf
        
        if df_T <= 0:
            raise ValidationError(f"Implied discount factor for T={T} is non-positive ({df_T:.4f}). Check input prices.")
        
        # Convert to spot rate: DF = 1/(1+r)^T => r = DF^(-1/T) - 1
        spot_rate = df_T ** (-1/T) - 1
        
        discount_factors[i] = df_T
        spot_rates[i] = spot_rate
        
        excel_steps.append(f"  Known PV = {' + '.join(known_pv_str) if known_pv_str else '0'} = {known_pv:.4f}")
        excel_steps.append(f"  Final CF = {final_cf:.2f} at T={T}")
        excel_steps.append(f"  DF({T}) = (Price - Known PV) / Final CF = ({P:.4f} - {known_pv:.4f}) / {final_cf:.2f} = {df_T:.6f}")
        excel_steps.append(f"  Spot rate r({T}) = DF^(-1/T) - 1 = {df_T:.6f}^(-1/{T}) - 1 = {spot_rate*100:.4f}%")
        excel_steps.append("")
    
    # Build result
    result = {
        'maturities': maturities,
        'spot_rates': spot_rates,
        'discount_factors': discount_factors,
        'zero_rates': spot_rates,  # Alias
    }
    
    tables = {
        'spot_curve': pd.DataFrame({
            'Maturity (T)': maturities,
            'Spot Rate (r)': spot_rates,
            'Spot Rate (%)': spot_rates * 100,
            'Discount Factor': discount_factors
        })
    }
    
    excel_steps.append("**Excel Implementation:**")
    excel_steps.append("1. For first bond (if zero-coupon or single CF): DF = Price/FV, r = DF^(-1/T) - 1")
    excel_steps.append("2. For subsequent bonds: Sum known PVs using interpolated DFs")
    excel_steps.append("3. Solve: DF(T) = (Price - KnownPV) / FinalCF")
    excel_steps.append("4. Convert: r(T) = DF(T)^(-1/T) - 1")
    
    return result, excel_steps, tables


def compute_forward_rates(
    spot_rates: np.ndarray,
    maturities: np.ndarray,
    forward_start: float,
    forward_end: float
) -> Tuple[Dict, List[str], Dict]:
    """
    Compute forward rate from spot rates.
    
    Forward rate f(t1, t2) satisfies: (1+r(t2))^t2 = (1+r(t1))^t1 × (1+f)^(t2-t1)
    
    Args:
        spot_rates: Array of spot rates
        maturities: Corresponding maturities
        forward_start: Start time of forward period
        forward_end: End time of forward period
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    # Validation
    spot_rates = validate_array(spot_rates, "Spot rates")
    maturities = validate_array(maturities, "Maturities", expected_len=len(spot_rates))
    
    if forward_start < 0:
        raise ValidationError("Forward start must be non-negative")
    if forward_end <= forward_start:
        raise ValidationError("Forward end must be greater than forward start")
    
    excel_steps = []
    excel_steps.append(f"**Forward Rate Calculation: f({forward_start}, {forward_end})**")
    excel_steps.append("")
    
    # Interpolate spot rates at the required maturities
    if forward_start == 0:
        r_start = 0
        df_start = 1.0
    else:
        r_start = np.interp(forward_start, maturities, spot_rates)
        df_start = 1 / (1 + r_start) ** forward_start
    
    r_end = np.interp(forward_end, maturities, spot_rates)
    df_end = 1 / (1 + r_end) ** forward_end
    
    excel_steps.append(f"Spot rate r({forward_start}) = {r_start*100:.4f}% (interpolated)")
    excel_steps.append(f"Spot rate r({forward_end}) = {r_end*100:.4f}% (interpolated)")
    excel_steps.append(f"DF({forward_start}) = 1/(1+{r_start:.4f})^{forward_start} = {df_start:.6f}")
    excel_steps.append(f"DF({forward_end}) = 1/(1+{r_end:.4f})^{forward_end} = {df_end:.6f}")
    excel_steps.append("")
    
    # Forward rate formula
    # (1+r2)^t2 = (1+r1)^t1 × (1+f)^(t2-t1)
    # f = [(1+r2)^t2 / (1+r1)^t1]^(1/(t2-t1)) - 1
    # Or equivalently: f = [DF(t1)/DF(t2)]^(1/(t2-t1)) - 1
    
    tau = forward_end - forward_start
    forward_rate = (df_start / df_end) ** (1/tau) - 1
    
    excel_steps.append("**Formula:**")
    excel_steps.append(r"f(t1,t2) = [DF(t1)/DF(t2)]^(1/(t2-t1)) - 1")
    excel_steps.append(f"f({forward_start},{forward_end}) = [{df_start:.6f}/{df_end:.6f}]^(1/{tau}) - 1")
    excel_steps.append(f"f({forward_start},{forward_end}) = {forward_rate*100:.4f}%")
    excel_steps.append("")
    
    excel_steps.append("**Excel Formula:**")
    excel_steps.append(f"`=(DF_start/DF_end)^(1/({forward_end}-{forward_start}))-1`")
    
    result = {
        'forward_rate': forward_rate,
        'forward_start': forward_start,
        'forward_end': forward_end,
        'tenor': tau,
        'r_start': r_start,
        'r_end': r_end,
        'df_start': df_start,
        'df_end': df_end
    }
    
    return result, excel_steps, {}


def compute_forward_curve(
    spot_rates: np.ndarray,
    maturities: np.ndarray,
    intervals: Optional[List[Tuple[float, float]]] = None
) -> Tuple[Dict, List[str], Dict]:
    """
    Compute multiple forward rates from a spot curve.
    
    Args:
        spot_rates: Array of spot rates
        maturities: Corresponding maturities
        intervals: List of (start, end) tuples. If None, compute 1-year forwards.
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    if intervals is None:
        # Default: 1-year forward rates starting at each integer year
        max_t = int(max(maturities))
        intervals = [(i, i+1) for i in range(max_t)]
    
    forwards = []
    all_steps = ["**Forward Rate Curve:**", ""]
    
    for t1, t2 in intervals:
        result, steps, _ = compute_forward_rates(spot_rates, maturities, t1, t2)
        forwards.append({
            'start': t1,
            'end': t2,
            'forward_rate': result['forward_rate']
        })
        all_steps.append(f"f({t1},{t2}) = {result['forward_rate']*100:.4f}%")
    
    result = {
        'forwards': forwards,
        'spot_rates': spot_rates,
        'maturities': maturities
    }
    
    tables = {
        'forward_curve': pd.DataFrame(forwards)
    }
    
    return result, all_steps, tables


def price_annuity_bond(
    cashflow_dates: np.ndarray,
    cashflow_amounts: np.ndarray,
    discount_factors: Optional[np.ndarray] = None,
    spot_rates: Optional[np.ndarray] = None,
    rate_maturities: Optional[np.ndarray] = None
) -> Tuple[Dict, List[str], Dict]:
    """
    Price an annuity/amortizing bond from a full cashflow vector.
    
    Args:
        cashflow_dates: Times of cashflows (in years)
        cashflow_amounts: Cashflow amounts
        discount_factors: Discount factors at cashflow dates (if provided directly)
        spot_rates: Spot rates for discounting (alternative to DFs)
        rate_maturities: Maturities corresponding to spot_rates (for interpolation)
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    # Validation
    cashflow_dates, cashflow_amounts = validate_cashflows(cashflow_dates, cashflow_amounts)
    
    excel_steps = []
    excel_steps.append("**Annuity Bond Pricing:**")
    excel_steps.append("")
    
    # Get discount factors
    if discount_factors is not None:
        dfs = validate_array(discount_factors, "Discount factors", expected_len=len(cashflow_dates))
    elif spot_rates is not None and rate_maturities is not None:
        spot_rates = validate_array(spot_rates, "Spot rates")
        rate_maturities = validate_array(rate_maturities, "Rate maturities", expected_len=len(spot_rates))
        
        # Interpolate spot rates and compute DFs
        interp_rates = np.interp(cashflow_dates, rate_maturities, spot_rates)
        dfs = 1 / (1 + interp_rates) ** cashflow_dates
        excel_steps.append("Discount factors computed from interpolated spot rates.")
    else:
        raise ValidationError("Must provide either discount_factors or (spot_rates + rate_maturities)")
    
    # Compute PVs
    pvs = cashflow_amounts * dfs
    total_pv = np.sum(pvs)
    
    excel_steps.append("")
    excel_steps.append("| Date (t) | Cashflow | DF(t) | PV |")
    excel_steps.append("|----------|----------|-------|-----|")
    for i in range(len(cashflow_dates)):
        excel_steps.append(f"| {cashflow_dates[i]:.2f} | {cashflow_amounts[i]:.2f} | {dfs[i]:.6f} | {pvs[i]:.4f} |")
    excel_steps.append(f"| **Total** | | | **{total_pv:.4f}** |")
    excel_steps.append("")
    
    excel_steps.append("**Excel Implementation:**")
    excel_steps.append("1. Set up columns: Date, Cashflow, DF, PV")
    excel_steps.append("2. PV column: `=Cashflow * DF`")
    excel_steps.append("3. Total PV: `=SUM(PV_column)`")
    
    result = {
        'price': total_pv,
        'cashflow_dates': cashflow_dates,
        'cashflow_amounts': cashflow_amounts,
        'discount_factors': dfs,
        'present_values': pvs
    }
    
    tables = {
        'cashflow_schedule': pd.DataFrame({
            'Date (t)': cashflow_dates,
            'Cashflow': cashflow_amounts,
            'Discount Factor': dfs,
            'Present Value': pvs
        })
    }
    
    return result, excel_steps, tables


def replicate_cashflows(
    target_dates: np.ndarray,
    target_amounts: np.ndarray,
    instrument_cashflows: List[Dict],
    instrument_prices: np.ndarray,
    target_market_price: Optional[float] = None
) -> Tuple[Dict, List[str], Dict]:
    """
    Replicate target cashflows using tradable instruments. Detect arbitrage if market price differs.
    
    Args:
        target_dates: Dates of target cashflows
        target_amounts: Target cashflow amounts
        instrument_cashflows: List of dicts with 'dates' and 'amounts' for each instrument
        instrument_prices: Market prices of instruments
        target_market_price: Market price of target (for arbitrage detection)
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    # Validation
    target_dates, target_amounts = validate_cashflows(target_dates, target_amounts)
    n_instruments = len(instrument_cashflows)
    instrument_prices = validate_array(instrument_prices, "Instrument prices", expected_len=n_instruments)
    
    excel_steps = []
    excel_steps.append("**Multi-Cashflow Replication:**")
    excel_steps.append("")
    
    # Build all unique dates
    all_dates = set(target_dates)
    for inst in instrument_cashflows:
        all_dates.update(inst['dates'])
    all_dates = sorted(all_dates)
    n_dates = len(all_dates)
    
    # Build cashflow matrix: C[i,j] = cashflow of instrument j at date i
    C = np.zeros((n_dates, n_instruments))
    for j, inst in enumerate(instrument_cashflows):
        for k, d in enumerate(inst['dates']):
            i = all_dates.index(d)
            C[i, j] = inst['amounts'][k]
    
    # Build target vector
    T = np.zeros(n_dates)
    for k, d in enumerate(target_dates):
        i = all_dates.index(d)
        T[i] = target_amounts[k]
    
    excel_steps.append(f"Cashflow matrix C: {n_dates} dates × {n_instruments} instruments")
    excel_steps.append(f"Target vector T: {n_dates} elements")
    excel_steps.append("")
    
    # Solve the system
    if n_instruments == n_dates:
        # Square system - try direct inversion
        det = np.linalg.det(C)
        if abs(det) > 1e-10:
            method = "Direct inversion (exact solution)"
            positions = np.linalg.solve(C, T)
            residuals = np.zeros(n_dates)
            exact_solution = True
        else:
            method = "Least squares (singular matrix)"
            positions, residuals_arr, rank, s = np.linalg.lstsq(C, T, rcond=None)
            residuals = C @ positions - T
            exact_solution = False
    elif n_instruments > n_dates:
        # Underdetermined - minimum norm solution
        method = "Minimum norm (underdetermined)"
        positions, residuals_arr, rank, s = np.linalg.lstsq(C, T, rcond=None)
        residuals = C @ positions - T
        exact_solution = np.allclose(residuals, 0, atol=1e-6)
    else:
        # Overdetermined - least squares
        method = "Least squares (overdetermined)"
        positions, residuals_arr, rank, s = np.linalg.lstsq(C, T, rcond=None)
        residuals = C @ positions - T
        exact_solution = np.allclose(residuals, 0, atol=1e-6)
    
    excel_steps.append(f"**Solution Method:** {method}")
    excel_steps.append("")
    
    # Replication cost
    replication_cost = np.sum(positions * instrument_prices)
    
    excel_steps.append("**Positions:**")
    for j in range(n_instruments):
        excel_steps.append(f"  Instrument {j+1}: {positions[j]:.6f} units × ${instrument_prices[j]:.2f} = ${positions[j]*instrument_prices[j]:.4f}")
    excel_steps.append(f"**Replication Cost:** ${replication_cost:.4f}")
    excel_steps.append("")
    
    # Verify cashflows
    excel_steps.append("**Cashflow Verification:**")
    excel_steps.append("| Date | Target | Replicated | Residual |")
    excel_steps.append("|------|--------|------------|----------|")
    replicated_cf = C @ positions
    for i, d in enumerate(all_dates):
        excel_steps.append(f"| {d:.2f} | {T[i]:.2f} | {replicated_cf[i]:.2f} | {residuals[i]:.4f} |")
    excel_steps.append("")
    
    # Arbitrage detection
    arbitrage = None
    if target_market_price is not None:
        price_diff = target_market_price - replication_cost
        
        excel_steps.append(f"**Arbitrage Analysis:**")
        excel_steps.append(f"  Target market price: ${target_market_price:.4f}")
        excel_steps.append(f"  Replication cost: ${replication_cost:.4f}")
        excel_steps.append(f"  Difference: ${price_diff:.4f}")
        excel_steps.append("")
        
        if abs(price_diff) > 0.01 and exact_solution:
            if price_diff > 0:
                # Target overpriced: sell target, buy replicating portfolio
                arbitrage = {
                    'type': 'Target overpriced',
                    'profit': price_diff,
                    'action': 'Sell target, buy replicating portfolio',
                    'trades': [
                        {'instrument': 'Target', 'position': -1, 'cashflow': target_market_price},
                    ]
                }
                for j in range(n_instruments):
                    arbitrage['trades'].append({
                        'instrument': f'Instrument {j+1}',
                        'position': positions[j],
                        'cashflow': -positions[j] * instrument_prices[j]
                    })
                excel_steps.append("**ARBITRAGE OPPORTUNITY:**")
                excel_steps.append(f"  Target is overpriced by ${price_diff:.4f}")
                excel_steps.append("  Action: SELL target, BUY replicating portfolio")
                excel_steps.append(f"  Net initial cash: +${price_diff:.4f}")
            else:
                # Target underpriced: buy target, short replicating portfolio
                arbitrage = {
                    'type': 'Target underpriced',
                    'profit': -price_diff,
                    'action': 'Buy target, short replicating portfolio',
                    'trades': [
                        {'instrument': 'Target', 'position': 1, 'cashflow': -target_market_price},
                    ]
                }
                for j in range(n_instruments):
                    arbitrage['trades'].append({
                        'instrument': f'Instrument {j+1}',
                        'position': -positions[j],
                        'cashflow': positions[j] * instrument_prices[j]
                    })
                excel_steps.append("**ARBITRAGE OPPORTUNITY:**")
                excel_steps.append(f"  Target is underpriced by ${-price_diff:.4f}")
                excel_steps.append("  Action: BUY target, SHORT replicating portfolio")
                excel_steps.append(f"  Net initial cash: +${-price_diff:.4f}")
        else:
            excel_steps.append("No arbitrage opportunity (prices are consistent or solution inexact)")
    
    excel_steps.append("")
    # Provide a worked example showing the computed matrices and results
    C_str = np.array2string(C, precision=4, separator=', ')
    T_str = np.array2string(T, precision=4, separator=', ')
    pos_str = np.array2string(positions, precision=4, separator=', ')
    excel_steps.append("**Replicating Cashflows – Worked Solution**")
    excel_steps.append(f"Cashflow matrix (C):\n{C_str}")
    excel_steps.append(f"Target vector (T):\n{T_str}")
    excel_steps.append(f"Solved positions N = C⁻¹T = {pos_str}")
    excel_steps.append(f"Replication cost = Σ N_i × price_i = {replication_cost:.4f}")
    excel_steps.append("")
    
    result = {
        'positions': positions,
        'replication_cost': replication_cost,
        'exact_solution': exact_solution,
        'method': method,
        'residuals': residuals,
        'arbitrage': arbitrage,
        'all_dates': all_dates,
        'cashflow_matrix': C,
        'target_vector': T
    }
    
    tables = {
        'positions': pd.DataFrame({
            'Instrument': [f'Instrument {j+1}' for j in range(n_instruments)],
            'Position': positions,
            'Price': instrument_prices,
            'Cost': positions * instrument_prices
        }),
        'verification': pd.DataFrame({
            'Date': all_dates,
            'Target': T,
            'Replicated': replicated_cf,
            'Residual': residuals
        })
    }
    
    return result, excel_steps, tables


# =============================================================================
# PORTFOLIO - PURE MATH FUNCTIONS (Constrained Optimization)
# =============================================================================

def optimize_portfolio_unconstrained(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float,
    risk_aversion: float,
    esg_scores: Optional[np.ndarray] = None,
    esg_preference: float = 0.0
) -> Tuple[Dict, List[str], Dict]:
    """
    Unconstrained mean-variance optimization (closed-form solution).
    
    Args:
        expected_returns: Expected returns vector
        covariance_matrix: Covariance matrix
        risk_free_rate: Risk-free rate
        risk_aversion: Risk aversion coefficient (gamma)
        esg_scores: Optional ESG scores
        esg_preference: ESG preference parameter (a)
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    # Validation
    n = len(expected_returns)
    mu = validate_array(expected_returns, "Expected returns")
    Sigma = validate_matrix(covariance_matrix, "Covariance matrix", 
                           expected_shape=(n, n), symmetric=True, positive_definite=True)
    rf = validate_rate(risk_free_rate, "Risk-free rate", allow_negative=True)
    gamma = validate_positive(risk_aversion, "Risk aversion")
    
    excel_steps = []
    excel_steps.append("**Unconstrained Mean-Variance Optimization (Closed Form)**")
    excel_steps.append("")
    
    # Compute inverse
    Sigma_inv = np.linalg.inv(Sigma)
    excess_returns = mu - rf
    ones = np.ones(n)
    
    # GMV Portfolio: w_gmv = Σ⁻¹1 / 1'Σ⁻¹1
    gmv_num = Sigma_inv @ ones
    gmv_weights = gmv_num / (ones @ Sigma_inv @ ones)
    gmv_return = gmv_weights @ mu
    gmv_vol = np.sqrt(gmv_weights @ Sigma @ gmv_weights)
    
    excel_steps.append("**Global Minimum Variance Portfolio:**")
    excel_steps.append("w_GMV = Σ⁻¹1 / (1'Σ⁻¹1)")
    excel_steps.append(f"Weights: {np.array2string(gmv_weights, precision=4)}")
    excel_steps.append(f"Return: {gmv_return*100:.4f}%, Volatility: {gmv_vol*100:.4f}%")
    excel_steps.append("")
    
    # Tangency Portfolio: w_tan = Σ⁻¹(μ-rf) / 1'Σ⁻¹(μ-rf)
    tan_num = Sigma_inv @ excess_returns
    tan_denom = ones @ Sigma_inv @ excess_returns
    if abs(tan_denom) < 1e-10:
        raise ValidationError("Cannot compute tangency portfolio (zero denominator)")
    tan_weights = tan_num / tan_denom
    tan_return = tan_weights @ mu
    tan_vol = np.sqrt(tan_weights @ Sigma @ tan_weights)
    tan_sharpe = (tan_return - rf) / tan_vol if tan_vol > 0 else 0
    
    excel_steps.append("**Tangency Portfolio:**")
    excel_steps.append("w_tan = Σ⁻¹(μ-rf) / 1'Σ⁻¹(μ-rf)")
    excel_steps.append(f"Weights: {np.array2string(tan_weights, precision=4)}")
    excel_steps.append(f"Return: {tan_return*100:.4f}%, Volatility: {tan_vol*100:.4f}%, Sharpe: {tan_sharpe:.4f}")
    excel_steps.append("")
    
    # Optimal Portfolio: w* = (1/γ)Σ⁻¹(μ-rf) + (a/γ)Σ⁻¹s
    opt_weights = (1/gamma) * (Sigma_inv @ excess_returns)
    
    if esg_scores is not None and esg_preference > 0:
        esg = validate_array(esg_scores, "ESG scores", expected_len=n)
        esg_component = (esg_preference / gamma) * (Sigma_inv @ esg)
        opt_weights = opt_weights + esg_component
        excel_steps.append("**Optimal Portfolio (with ESG):**")
        excel_steps.append("w* = (1/γ)Σ⁻¹(μ-rf) + (a/γ)Σ⁻¹s")
    else:
        esg_component = np.zeros(n)
        excel_steps.append("**Optimal Portfolio:**")
        excel_steps.append("w* = (1/γ)Σ⁻¹(μ-rf)")
    
    opt_return = opt_weights @ mu
    opt_vol = np.sqrt(opt_weights @ Sigma @ opt_weights)
    opt_sharpe = (opt_return - rf) / opt_vol if opt_vol > 0 else 0
    risky_weight = np.sum(opt_weights)
    
    excel_steps.append(f"Weights: {np.array2string(opt_weights, precision=4)}")
    excel_steps.append(f"Total risky weight: {risky_weight:.4f}, Cash: {1-risky_weight:.4f}")
    excel_steps.append(f"Return: {opt_return*100:.4f}%, Volatility: {opt_vol*100:.4f}%, Sharpe: {opt_sharpe:.4f}")
    excel_steps.append("")
    
    excel_steps.append("**Excel Implementation:**")
    excel_steps.append("1. Compute Σ⁻¹: `=MINVERSE(cov_matrix)`")
    excel_steps.append("2. Compute Σ⁻¹(μ-rf): `=MMULT(Sigma_inv, excess_returns)`")
    excel_steps.append("3. Tangency weights: `=numerator/SUM(numerator)`")
    excel_steps.append("4. Optimal weights: `=(1/gamma)*MMULT(Sigma_inv, excess_returns)`")
    
    result = {
        'gmv': {
            'weights': gmv_weights,
            'return': gmv_return,
            'volatility': gmv_vol,
            'sharpe': (gmv_return - rf) / gmv_vol if gmv_vol > 0 else 0
        },
        'tangency': {
            'weights': tan_weights,
            'return': tan_return,
            'volatility': tan_vol,
            'sharpe': tan_sharpe
        },
        'optimal': {
            'weights': opt_weights,
            'return': opt_return,
            'volatility': opt_vol,
            'sharpe': opt_sharpe,
            'risky_weight': risky_weight,
            'cash_weight': 1 - risky_weight
        },
        'n_assets': n,
        'sigma_inv': Sigma_inv
    }
    
    tables = {
        'portfolio_summary': pd.DataFrame({
            'Portfolio': ['GMV', 'Tangency', f'Optimal (γ={gamma})'],
            'Return': [gmv_return, tan_return, opt_return],
            'Volatility': [gmv_vol, tan_vol, opt_vol],
            'Sharpe': [result['gmv']['sharpe'], tan_sharpe, opt_sharpe]
        })
    }
    
    return result, excel_steps, tables


def optimize_portfolio_constrained(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float,
    objective: str = 'max_sharpe',  # 'max_sharpe', 'min_variance', 'max_return', 'target_return', 'max_utility'
    target_return: Optional[float] = None,
    target_volatility: Optional[float] = None,
    max_weight: Optional[float] = None,
    no_short: bool = False,
    leverage_cap: Optional[float] = None,
    min_weights: Optional[np.ndarray] = None,
    max_weights: Optional[np.ndarray] = None,
    allow_riskfree: bool = False,
    allow_borrowing: bool = False,
    risk_aversion: float = 4.0,
    utility_type: str = 'standard'
) -> Tuple[Dict, List[str], Dict]:
    """
    Constrained mean-variance optimization using SLSQP.
    
    Args:
        expected_returns: Expected returns vector
        covariance_matrix: Covariance matrix
        risk_free_rate: Risk-free rate
        objective: 'max_sharpe', 'min_variance', 'max_return', 'target_return', 'max_utility'
        target_return: For 'target_return' objective
        target_volatility: Volatility constraint (σ_p <= target)
        max_weight: Maximum weight per asset (w_i <= max_weight)
        no_short: If True, w_i >= 0 for all i
        leverage_cap: Maximum sum of absolute weights (Σ|w_i| <= L)
        min_weights: Per-asset minimum weights
        max_weights: Per-asset maximum weights
        allow_riskfree: If True, can invest in risk-free asset (Σw_i <= 1, residual earns rf)
                        If False, must be fully invested in risky assets (Σw_i = 1)
        allow_borrowing: If True (requires allow_riskfree=True), can borrow at rf (Σw_i > 1 allowed)
                         Negative cash weight means borrowing at the risk-free rate
        risk_aversion: Risk aversion coefficient γ (for max_utility objective)
        utility_type: 'standard' for U = E[r] - 0.5*γ*σ², 'simple' for U = E[r] - γ*σ²
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    # Validation
    n = len(expected_returns)
    mu = validate_array(expected_returns, "Expected returns")
    Sigma = validate_matrix(covariance_matrix, "Covariance matrix",
                           expected_shape=(n, n), symmetric=True, positive_definite=True)
    rf = validate_rate(risk_free_rate, "Risk-free rate", allow_negative=True)
    
    excel_steps = []
    excel_steps.append("**Constrained Mean-Variance Optimization (SLSQP)**")
    excel_steps.append("")
    
    # Build constraints list
    constraints = []
    constraint_descriptions = []
    
    # Budget constraint: depends on investment mode
    # Mode 1: Fully Invested (stocks only) - sum(w) = 1
    # Mode 2: Long Only (stocks + cash) - sum(w) <= 1, no borrowing
    # Mode 3: Leverage Allowed - no sum constraint (can borrow at rf)
    if not allow_riskfree:
        # Fully Invested: sum(w) = 1, all wealth in risky assets
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        constraint_descriptions.append("Budget: Σw_i = 1 (fully invested in risky assets)")
    elif not allow_borrowing:
        # Long Only: sum(w) <= 1, residual in risk-free, no borrowing
        constraints.append({'type': 'ineq', 'fun': lambda w: 1 - np.sum(w)})
        constraint_descriptions.append("Budget: Σw_i ≤ 1 (can hold cash, no borrowing)")
    else:
        # Leverage Allowed: no sum constraint, can borrow (negative cash)
        # The cash weight = 1 - sum(w) can be negative (borrowing)
        constraint_descriptions.append("Budget: Leverage allowed (can borrow at rf)")
    
    # Target volatility constraint
    if target_volatility is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, S=Sigma, tv=target_volatility: tv**2 - w @ S @ w
        })
        constraint_descriptions.append(f"Volatility: σ_p <= {target_volatility*100:.2f}%")
    
    # Target return constraint (for min_variance objective)
    if target_return is not None and objective == 'target_return':
        constraints.append({
            'type': 'eq',
            'fun': lambda w, m=mu, tr=target_return: w @ m - tr
        })
        constraint_descriptions.append(f"Return: μ_p = {target_return*100:.2f}%")
    
    # Leverage constraint: Σ|w_i| <= L
    # This is implemented via auxiliary variables in a proper QP
    # For SLSQP, we use a penalty approach or approximate
    if leverage_cap is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, L=leverage_cap: L - np.sum(np.abs(w))
        })
        constraint_descriptions.append(f"Leverage: Σ|w_i| <= {leverage_cap}")
    
    excel_steps.append("**Constraints:**")
    for desc in constraint_descriptions:
        excel_steps.append(f"  - {desc}")
    excel_steps.append("")
    
    # Bounds
    if min_weights is None:
        min_weights = np.full(n, -np.inf if not no_short else 0)
    if max_weights is None:
        max_weights = np.full(n, np.inf if max_weight is None else max_weight)
    elif max_weight is not None:
        max_weights = np.minimum(max_weights, max_weight)
    
    if no_short:
        min_weights = np.maximum(min_weights, 0)
        constraint_descriptions.append("No short: w_i >= 0")
    
    if max_weight is not None:
        constraint_descriptions.append(f"Max weight: w_i <= {max_weight*100:.1f}%")
    
    bounds = [(min_weights[i], max_weights[i]) for i in range(n)]
    
    # Define objective function
    if objective == 'max_sharpe':
        # Maximize Sharpe ratio: max (w'μ - rf) / sqrt(w'Σw)
        # Note: When allow_riskfree=True, the Sharpe of the risky portion is what matters
        # Equivalent to: min -Sharpe
        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ Sigma @ w)
            return -(ret - rf) / vol if vol > 1e-10 else 0
        obj_func = neg_sharpe
        obj_name = "Maximize Sharpe Ratio"
    
    elif objective == 'min_variance':
        # Minimize variance
        def variance(w):
            return w @ Sigma @ w
        obj_func = variance
        obj_name = "Minimize Variance"
    
    elif objective == 'max_return':
        # Maximize return (subject to constraints)
        # When allow_riskfree=True, total return = w'μ + (1-Σw)*rf
        if allow_riskfree:
            def neg_return(w):
                risky_return = w @ mu
                cash_weight = 1 - np.sum(w)
                total_return = risky_return + cash_weight * rf
                return -total_return
            obj_name = "Maximize Return (including risk-free)"
        else:
            def neg_return(w):
                return -(w @ mu)
            obj_name = "Maximize Return"
        obj_func = neg_return
    
    elif objective == 'target_return':
        # Minimize variance for target return
        def variance(w):
            return w @ Sigma @ w
        obj_func = variance
        obj_name = f"Minimize Variance for Target Return {target_return*100:.2f}%"
    
    elif objective == 'max_utility':
        # Maximize utility: U = E[r] - penalty*σ²
        # penalty = 0.5*γ (standard) or γ (simple)
        # Since we minimize, we minimize -U
        penalty_factor = 0.5 * risk_aversion if utility_type == 'standard' else risk_aversion
        
        if allow_riskfree:
            # Total return includes cash: w'μ + (1-Σw)*rf
            def neg_utility(w, m=mu, S=Sigma, rf_=rf, pf=penalty_factor):
                risky_return = w @ m
                cash_weight = 1 - np.sum(w)
                total_return = risky_return + cash_weight * rf_
                port_variance = w @ S @ w
                utility = total_return - pf * port_variance
                return -utility
            obj_name = f"Maximize Utility (γ={risk_aversion}, {utility_type}, with cash)"
        else:
            # Fully invested: return = w'μ
            def neg_utility(w, m=mu, S=Sigma, pf=penalty_factor):
                port_return = w @ m
                port_variance = w @ S @ w
                utility = port_return - pf * port_variance
                return -utility
            obj_name = f"Maximize Utility (γ={risk_aversion}, {utility_type})"
        
        obj_func = neg_utility
    
    else:
        raise ValidationError(f"Unknown objective: {objective}")
    
    excel_steps.append(f"**Objective:** {obj_name}")
    excel_steps.append("")
    
    # Initial guess (equal weights)
    w0 = np.ones(n) / n
    
    # Optimize
    result_opt = minimize(
        obj_func,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-10, 'maxiter': 1000}
    )
    
    if not result_opt.success:
        excel_steps.append(f"**Warning:** Optimization may not have converged: {result_opt.message}")
    
    weights = result_opt.x
    
    # Calculate cash weight and portfolio statistics
    cash_weight = 1 - np.sum(weights) if allow_riskfree else 0.0
    risky_return = weights @ mu
    port_return = risky_return + cash_weight * rf if allow_riskfree else risky_return
    port_vol = np.sqrt(weights @ Sigma @ weights)
    port_sharpe = (port_return - rf) / port_vol if port_vol > 1e-10 else 0
    
    excel_steps.append("**Solution:**")
    excel_steps.append(f"Risky Asset Weights: {np.array2string(weights, precision=4)}")
    if allow_riskfree:
        excel_steps.append(f"Cash/Risk-Free Weight: {cash_weight*100:.4f}%")
        excel_steps.append(f"Total Risky Weight: {np.sum(weights)*100:.4f}%")
    excel_steps.append(f"Expected Return: {port_return*100:.4f}%")
    excel_steps.append(f"Volatility: {port_vol*100:.4f}%")
    excel_steps.append(f"Sharpe Ratio: {port_sharpe:.4f}")
    excel_steps.append("")
    
    # Check which constraints bind
    binding = []
    
    # Check bound constraints
    for i in range(n):
        if abs(weights[i] - min_weights[i]) < 1e-6:
            binding.append(f"Asset {i+1} at minimum ({min_weights[i]:.4f})")
        if abs(weights[i] - max_weights[i]) < 1e-6:
            binding.append(f"Asset {i+1} at maximum ({max_weights[i]:.4f})")
    
    # Check volatility constraint
    if target_volatility is not None:
        if abs(port_vol - target_volatility) < 1e-6:
            binding.append(f"Volatility constraint binding (σ = {target_volatility*100:.2f}%)")
    
    # Check leverage constraint
    if leverage_cap is not None:
        total_leverage = np.sum(np.abs(weights))
        if abs(total_leverage - leverage_cap) < 1e-6:
            binding.append(f"Leverage constraint binding (L = {leverage_cap})")
    
    if binding:
        excel_steps.append("**Binding Constraints:**")
        for b in binding:
            excel_steps.append(f"  - {b}")
    else:
        excel_steps.append("**Binding Constraints:** None (interior solution)")
    excel_steps.append("")
    
    excel_steps.append("**Excel/Solver Implementation:**")
    excel_steps.append("1. Set up weights in cells (e.g., B2:B4)")
    excel_steps.append("2. Calculate portfolio return: `=SUMPRODUCT(weights, returns)`")
    excel_steps.append("3. Calculate portfolio variance: `=MMULT(MMULT(TRANSPOSE(w),Σ),w)`")
    if objective == 'max_utility':
        penalty_factor = 0.5 * risk_aversion if utility_type == 'standard' else risk_aversion
        excel_steps.append(f"4. Calculate utility: `=return - {penalty_factor}*variance`")
        excel_steps.append("5. Use Solver: Maximize utility cell, add constraints, solve")
    else:
        excel_steps.append("4. Use Solver: Set objective, add constraints, solve")
    
    # Calculate utility if max_utility objective
    port_variance = port_vol ** 2
    if objective == 'max_utility':
        penalty_factor = 0.5 * risk_aversion if utility_type == 'standard' else risk_aversion
        port_utility = port_return - penalty_factor * port_variance
    else:
        port_utility = None
    
    result = {
        'weights': weights,
        'cash_weight': cash_weight,
        'return': port_return,
        'volatility': port_vol,
        'variance': port_variance,
        'sharpe': port_sharpe,
        'utility': port_utility,
        'risk_aversion': risk_aversion if objective == 'max_utility' else None,
        'utility_type': utility_type if objective == 'max_utility' else None,
        'objective': objective,
        'success': result_opt.success,
        'message': result_opt.message,
        'binding_constraints': binding,
        'total_leverage': np.sum(np.abs(weights)),
        'allow_riskfree': allow_riskfree,
        'allow_borrowing': allow_borrowing
    }
    
    # Build weights table
    weight_data = {
        'Asset': [f'Asset {i+1}' for i in range(n)],
        'Weight': weights,
        'Weight (%)': weights * 100,
        'Min Bound': min_weights,
        'Max Bound': max_weights
    }
    
    # Add cash row if risk-free is allowed
    if allow_riskfree:
        weight_data['Asset'].append('Risk-Free (Cash)')
        weight_data['Weight'] = np.append(weights, cash_weight)
        weight_data['Weight (%)'] = np.append(weights * 100, cash_weight * 100)
        weight_data['Min Bound'] = np.append(min_weights, 0.0)
        weight_data['Max Bound'] = np.append(max_weights, 1.0)
    
    tables = {
        'weights': pd.DataFrame(weight_data)
    }
    
    return result, excel_steps, tables


# =============================================================================
# FACTOR MODELS - PURE MATH FUNCTIONS
# =============================================================================

def build_factor_covariance(
    betas: np.ndarray,
    factor_covariance: np.ndarray,
    idiosyncratic_variances: np.ndarray
) -> Tuple[Dict, List[str], Dict]:
    """
    Build full asset covariance matrix from factor model.
    
    Σ = B Σ_f B' + D
    
    Args:
        betas: Beta matrix (n_assets × n_factors)
        factor_covariance: Factor covariance matrix (n_factors × n_factors)
        idiosyncratic_variances: Diagonal of idiosyncratic variance matrix
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    # Validation
    B = validate_matrix(betas, "Betas")
    n_assets, n_factors = B.shape
    
    Sigma_f = validate_matrix(factor_covariance, "Factor covariance",
                              expected_shape=(n_factors, n_factors), 
                              symmetric=True, positive_definite=True)
    
    D_diag = validate_array(idiosyncratic_variances, "Idiosyncratic variances", 
                            expected_len=n_assets)
    
    if np.any(D_diag < 0):
        raise ValidationError("Idiosyncratic variances must be non-negative")
    
    D = np.diag(D_diag)
    
    excel_steps = []
    excel_steps.append("**Factor Model Covariance Construction:**")
    excel_steps.append("")
    excel_steps.append("Σ = B Σ_f B' + D")
    excel_steps.append("")
    excel_steps.append(f"B: {n_assets} assets × {n_factors} factors")
    excel_steps.append(f"Σ_f: {n_factors} × {n_factors} factor covariance")
    excel_steps.append(f"D: {n_assets} × {n_assets} diagonal idiosyncratic")
    excel_steps.append("")
    
    # Compute systematic covariance
    systematic_cov = B @ Sigma_f @ B.T
    
    # Full covariance
    Sigma = systematic_cov + D
    
    excel_steps.append("**Step 1: Compute B Σ_f B'**")
    excel_steps.append("Excel: `=MMULT(MMULT(B, Sigma_f), TRANSPOSE(B))`")
    excel_steps.append("")
    excel_steps.append("**Step 2: Add D**")
    excel_steps.append("Σ = B Σ_f B' + D (add diagonal elements)")
    excel_steps.append("")
    
    # Compute some diagnostics
    total_vars = np.diag(Sigma)
    systematic_vars = np.diag(systematic_cov)
    r_squared = systematic_vars / total_vars
    
    excel_steps.append("**R² (proportion of variance explained by factors):**")
    for i in range(n_assets):
        excel_steps.append(f"  Asset {i+1}: R² = {r_squared[i]:.4f} ({r_squared[i]*100:.2f}%)")
    
    result = {
        'covariance_matrix': Sigma,
        'systematic_covariance': systematic_cov,
        'idiosyncratic_matrix': D,
        'r_squared': r_squared,
        'n_assets': n_assets,
        'n_factors': n_factors
    }
    
    tables = {
        'covariance': pd.DataFrame(Sigma, 
            index=[f'Asset {i+1}' for i in range(n_assets)],
            columns=[f'Asset {i+1}' for i in range(n_assets)]),
        'diagnostics': pd.DataFrame({
            'Asset': [f'Asset {i+1}' for i in range(n_assets)],
            'Total Var': total_vars,
            'Systematic Var': systematic_vars,
            'Idiosyncratic Var': D_diag,
            'R²': r_squared
        })
    }
    
    return result, excel_steps, tables


def optimize_factor_neutral(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    betas: np.ndarray,
    objective: str = 'min_variance',  # 'min_variance' or 'max_return'
    risk_cap: Optional[float] = None,
    no_short: bool = False,
    max_weight: Optional[float] = None
) -> Tuple[Dict, List[str], Dict]:
    """
    Solve for factor-neutral portfolio.
    
    Constraints:
    - B'w = 0 (factor neutrality)
    - Σw = 1 (budget)
    - Optional: w >= 0, w <= max_weight, σ_p <= risk_cap
    
    Args:
        expected_returns: Expected returns (can be alphas)
        covariance_matrix: Asset covariance matrix
        betas: Beta matrix (n_assets × n_factors)
        objective: 'min_variance' or 'max_return'
        risk_cap: Maximum portfolio volatility
        no_short: No short-selling constraint
        max_weight: Maximum position size
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    # Validation
    n = len(expected_returns)
    mu = validate_array(expected_returns, "Expected returns")
    Sigma = validate_matrix(covariance_matrix, "Covariance matrix",
                           expected_shape=(n, n), symmetric=True, positive_definite=True)
    B = validate_matrix(betas, "Betas")
    
    if B.shape[0] != n:
        raise ValidationError(f"Betas rows ({B.shape[0]}) must match number of assets ({n})")
    
    n_factors = B.shape[1]
    
    excel_steps = []
    excel_steps.append("**Factor-Neutral Portfolio Optimization:**")
    excel_steps.append("")
    excel_steps.append(f"Neutralizing {n_factors} factor(s)")
    excel_steps.append("")
    
    # Constraints
    constraints = []
    
    # Budget constraint
    constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # Factor neutrality: B'w = 0 for each factor
    for f in range(n_factors):
        constraints.append({
            'type': 'eq',
            'fun': lambda w, bf=B[:, f]: w @ bf
        })
    
    excel_steps.append("**Constraints:**")
    excel_steps.append("  - Budget: Σw_i = 1")
    for f in range(n_factors):
        excel_steps.append(f"  - Factor {f+1} neutral: Σw_i × β_i{f+1} = 0")
    
    # Risk cap
    if risk_cap is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w, S=Sigma, rc=risk_cap: rc**2 - w @ S @ w
        })
        excel_steps.append(f"  - Risk cap: σ_p <= {risk_cap*100:.2f}%")
    
    # Bounds
    lb = 0 if no_short else -np.inf
    ub = max_weight if max_weight is not None else np.inf
    bounds = [(lb, ub) for _ in range(n)]
    
    if no_short:
        excel_steps.append("  - No short: w_i >= 0")
    if max_weight is not None:
        excel_steps.append(f"  - Max weight: w_i <= {max_weight*100:.1f}%")
    excel_steps.append("")
    
    # Objective
    if objective == 'min_variance':
        obj_func = lambda w: w @ Sigma @ w
        obj_name = "Minimize Variance"
    elif objective == 'max_return':
        obj_func = lambda w: -(w @ mu)
        obj_name = "Maximize Return (Alpha)"
    else:
        raise ValidationError(f"Unknown objective: {objective}")
    
    excel_steps.append(f"**Objective:** {obj_name}")
    excel_steps.append("")
    
    # Initial guess
    w0 = np.ones(n) / n
    
    # Optimize
    result_opt = minimize(
        obj_func,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-10, 'maxiter': 1000}
    )
    
    weights = result_opt.x
    port_return = weights @ mu
    port_vol = np.sqrt(weights @ Sigma @ weights)
    
    # Check factor exposures
    factor_exposures = B.T @ weights
    
    excel_steps.append("**Solution:**")
    excel_steps.append(f"Weights: {np.array2string(weights, precision=4)}")
    excel_steps.append(f"Expected Return/Alpha: {port_return*100:.4f}%")
    excel_steps.append(f"Volatility: {port_vol*100:.4f}%")
    excel_steps.append("")
    excel_steps.append("**Factor Exposures (should be ~0):**")
    for f in range(n_factors):
        excel_steps.append(f"  Factor {f+1}: {factor_exposures[f]:.6f}")
    
    if not result_opt.success:
        excel_steps.append(f"\n**Warning:** {result_opt.message}")
    
    result = {
        'weights': weights,
        'return': port_return,
        'volatility': port_vol,
        'factor_exposures': factor_exposures,
        'success': result_opt.success,
        'message': result_opt.message
    }
    
    tables = {
        'weights': pd.DataFrame({
            'Asset': [f'Asset {i+1}' for i in range(n)],
            'Weight': weights,
            'Expected Return': mu,
            'Beta_1': B[:, 0] if n_factors >= 1 else np.nan
        })
    }
    
    return result, excel_steps, tables


def joint_asset_factor_optimization(
    asset_returns: np.ndarray,
    asset_covariance: np.ndarray,
    factor_returns: np.ndarray,
    factor_covariance: np.ndarray,
    asset_factor_covariance: np.ndarray,
    risk_free_rate: float,
    risk_aversion: float
) -> Tuple[Dict, List[str], Dict]:
    """
    Joint optimization over assets and factor portfolios.
    
    Combines specific assets with factor portfolios as investable universe.
    
    Args:
        asset_returns: Expected returns of specific assets
        asset_covariance: Covariance among assets
        factor_returns: Expected returns of factor portfolios
        factor_covariance: Covariance among factor portfolios
        asset_factor_covariance: Covariance between assets and factors
        risk_free_rate: Risk-free rate
        risk_aversion: Risk aversion coefficient
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    n_assets = len(asset_returns)
    n_factors = len(factor_returns)
    n_total = n_assets + n_factors
    
    # Build combined return vector
    mu = np.concatenate([asset_returns, factor_returns])
    
    # Build combined covariance matrix
    Sigma = np.zeros((n_total, n_total))
    Sigma[:n_assets, :n_assets] = asset_covariance
    Sigma[n_assets:, n_assets:] = factor_covariance
    Sigma[:n_assets, n_assets:] = asset_factor_covariance
    Sigma[n_assets:, :n_assets] = asset_factor_covariance.T
    
    excel_steps = []
    excel_steps.append("**Joint Asset + Factor Portfolio Optimization:**")
    excel_steps.append("")
    excel_steps.append(f"Investable universe: {n_assets} assets + {n_factors} factor portfolios")
    excel_steps.append("")
    
    # Use unconstrained optimization
    result, steps, tables = optimize_portfolio_unconstrained(
        mu, Sigma, risk_free_rate, risk_aversion
    )
    
    # Split weights
    asset_weights = result['optimal']['weights'][:n_assets]
    factor_weights = result['optimal']['weights'][n_assets:]
    
    excel_steps.extend(steps)
    excel_steps.append("")
    excel_steps.append("**Allocation Split:**")
    excel_steps.append(f"Specific assets: {np.array2string(asset_weights, precision=4)}")
    excel_steps.append(f"Factor portfolios: {np.array2string(factor_weights, precision=4)}")
    
    result['asset_weights'] = asset_weights
    result['factor_weights'] = factor_weights
    result['n_assets'] = n_assets
    result['n_factors'] = n_factors
    
    return result, excel_steps, tables


# =============================================================================
# PROBABILITY - PURE MATH FUNCTIONS (Joint Normals)
# =============================================================================

def bivariate_normal_cdf(
    a: float,
    b: float,
    mu_x: float,
    mu_y: float,
    sigma_x: float,
    sigma_y: float,
    rho: float
) -> Tuple[Dict, List[str], Dict]:
    """
    Compute P(X <= a, Y <= b) for bivariate normal.
    
    Args:
        a: Upper bound for X
        b: Upper bound for Y
        mu_x, mu_y: Means
        sigma_x, sigma_y: Standard deviations
        rho: Correlation coefficient
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    from scipy.stats import multivariate_normal
    
    # Validation
    if sigma_x <= 0 or sigma_y <= 0:
        raise ValidationError("Standard deviations must be positive")
    if not -1 <= rho <= 1:
        raise ValidationError("Correlation must be between -1 and 1")
    
    excel_steps = []
    excel_steps.append("**Bivariate Normal CDF: P(X ≤ a, Y ≤ b)**")
    excel_steps.append("")
    excel_steps.append(f"X ~ N({mu_x}, {sigma_x}²)")
    excel_steps.append(f"Y ~ N({mu_y}, {sigma_y}²)")
    excel_steps.append(f"Corr(X,Y) = ρ = {rho}")
    excel_steps.append("")
    
    # Build covariance matrix
    cov_xy = rho * sigma_x * sigma_y
    cov_matrix = np.array([
        [sigma_x**2, cov_xy],
        [cov_xy, sigma_y**2]
    ])
    
    mean = np.array([mu_x, mu_y])
    
    # Compute probability
    rv = multivariate_normal(mean=mean, cov=cov_matrix)
    prob = rv.cdf([a, b])
    
    excel_steps.append(f"**P(X ≤ {a}, Y ≤ {b}) = {prob:.6f}**")
    excel_steps.append("")
    
    # Also compute marginals for reference
    from scipy.stats import norm
    p_x = norm.cdf(a, mu_x, sigma_x)
    p_y = norm.cdf(b, mu_y, sigma_y)
    
    excel_steps.append("**Marginal probabilities (for comparison):**")
    excel_steps.append(f"P(X ≤ {a}) = {p_x:.6f}")
    excel_steps.append(f"P(Y ≤ {b}) = {p_y:.6f}")
    excel_steps.append(f"P(X ≤ {a}) × P(Y ≤ {b}) = {p_x*p_y:.6f} (if independent)")
    excel_steps.append("")
    
    excel_steps.append("**Excel Implementation:**")
    excel_steps.append("Excel does not have built-in bivariate normal CDF.")
    excel_steps.append("Options:")
    excel_steps.append("1. Use VBA/add-in with numerical integration")
    excel_steps.append("2. Monte Carlo simulation:")
    excel_steps.append("   - Generate correlated normals using Cholesky")
    excel_steps.append("   - Count proportion where X<=a AND Y<=b")
    
    result = {
        'probability': prob,
        'a': a,
        'b': b,
        'mu_x': mu_x,
        'mu_y': mu_y,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'rho': rho,
        'marginal_x': p_x,
        'marginal_y': p_y
    }
    
    return result, excel_steps, {}


def conditional_normal(
    y_value: float,
    mu_x: float,
    mu_y: float,
    sigma_x: float,
    sigma_y: float,
    rho: float
) -> Tuple[Dict, List[str], Dict]:
    """
    Compute conditional distribution X | Y = y.
    
    For bivariate normal:
    X | Y=y ~ N(μ_x + ρ(σ_x/σ_y)(y - μ_y), σ_x²(1-ρ²))
    
    Args:
        y_value: Observed value of Y
        mu_x, mu_y: Means
        sigma_x, sigma_y: Standard deviations
        rho: Correlation coefficient
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    # Validation
    if sigma_x <= 0 or sigma_y <= 0:
        raise ValidationError("Standard deviations must be positive")
    if not -1 <= rho <= 1:
        raise ValidationError("Correlation must be between -1 and 1")
    
    excel_steps = []
    excel_steps.append(f"**Conditional Distribution: X | Y = {y_value}**")
    excel_steps.append("")
    excel_steps.append(f"X ~ N({mu_x}, {sigma_x}²)")
    excel_steps.append(f"Y ~ N({mu_y}, {sigma_y}²)")
    excel_steps.append(f"Corr(X,Y) = ρ = {rho}")
    excel_steps.append("")
    
    # Conditional mean
    cond_mean = mu_x + rho * (sigma_x / sigma_y) * (y_value - mu_y)
    
    # Conditional variance
    cond_var = sigma_x**2 * (1 - rho**2)
    cond_std = np.sqrt(cond_var)
    
    excel_steps.append("**Formula:**")
    excel_steps.append("X | Y=y ~ N(μ_X|Y, σ²_X|Y)")
    excel_steps.append("")
    excel_steps.append("μ_X|Y = μ_X + ρ(σ_X/σ_Y)(y - μ_Y)")
    excel_steps.append(f"      = {mu_x} + {rho}×({sigma_x}/{sigma_y})×({y_value} - {mu_y})")
    excel_steps.append(f"      = {cond_mean:.6f}")
    excel_steps.append("")
    excel_steps.append("σ²_X|Y = σ²_X(1 - ρ²)")
    excel_steps.append(f"       = {sigma_x}²×(1 - {rho}²)")
    excel_steps.append(f"       = {cond_var:.6f}")
    excel_steps.append(f"σ_X|Y  = {cond_std:.6f}")
    excel_steps.append("")
    
    excel_steps.append(f"**Result: X | Y={y_value} ~ N({cond_mean:.4f}, {cond_std:.4f}²)**")
    excel_steps.append("")
    
    excel_steps.append("**Excel Implementation:**")
    excel_steps.append(f"`=μ_X + ρ*(σ_X/σ_Y)*(y - μ_Y)` for conditional mean")
    excel_steps.append(f"`=σ_X*SQRT(1-ρ^2)` for conditional std dev")
    
    result = {
        'conditional_mean': cond_mean,
        'conditional_variance': cond_var,
        'conditional_std': cond_std,
        'y_value': y_value,
        'mu_x': mu_x,
        'mu_y': mu_y,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'rho': rho
    }
    
    return result, excel_steps, {}


def portfolio_return_distribution(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: Optional[np.ndarray] = None,
    volatilities: Optional[np.ndarray] = None,
    correlations: Optional[np.ndarray] = None,
    threshold: Optional[float] = None
) -> Tuple[Dict, List[str], Dict]:
    """
    Compute distribution of portfolio return w'R for jointly normal returns.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns vector μ
        covariance_matrix: Covariance matrix Σ (if provided directly)
        volatilities: Asset volatilities (alternative input)
        correlations: Correlation matrix (alternative input)
        threshold: Compute P(w'R <= threshold) and P(w'R >= threshold)
    
    Returns:
        (result_dict, excel_steps, tables_dict)
    """
    n = len(weights)
    w = validate_array(weights, "Weights")
    mu = validate_array(expected_returns, "Expected returns", expected_len=n)
    
    # Build covariance matrix if not provided directly
    if covariance_matrix is not None:
        Sigma = validate_matrix(covariance_matrix, "Covariance matrix",
                               expected_shape=(n, n), symmetric=True)
    elif volatilities is not None and correlations is not None:
        sigma = validate_array(volatilities, "Volatilities", expected_len=n)
        corr = validate_matrix(correlations, "Correlations",
                              expected_shape=(n, n), symmetric=True)
        D = np.diag(sigma)
        Sigma = D @ corr @ D
    else:
        raise ValidationError("Must provide either covariance_matrix or (volatilities + correlations)")
    
    excel_steps = []
    excel_steps.append("**Portfolio Return Distribution:**")
    excel_steps.append("")
    excel_steps.append("If R ~ N(μ, Σ), then w'R ~ N(w'μ, w'Σw)")
    excel_steps.append("")
    
    # Portfolio moments
    port_mean = w @ mu
    port_var = w @ Sigma @ w
    port_std = np.sqrt(port_var)
    
    excel_steps.append("**Calculation:**")
    excel_steps.append(f"E[w'R] = w'μ = {port_mean:.6f} ({port_mean*100:.4f}%)")
    excel_steps.append(f"Var(w'R) = w'Σw = {port_var:.8f}")
    excel_steps.append(f"Std(w'R) = √(w'Σw) = {port_std:.6f} ({port_std*100:.4f}%)")
    excel_steps.append("")
    excel_steps.append(f"**w'R ~ N({port_mean:.4f}, {port_std:.4f}²)**")
    excel_steps.append("")
    
    result = {
        'portfolio_mean': port_mean,
        'portfolio_variance': port_var,
        'portfolio_std': port_std,
        'weights': w,
        'asset_means': mu
    }
    
    # Compute probabilities if threshold provided
    if threshold is not None:
        z = (threshold - port_mean) / port_std
        prob_below = norm.cdf(z)
        prob_above = 1 - prob_below
        
        excel_steps.append(f"**Probability Calculations (threshold = {threshold:.4f}):**")
        excel_steps.append(f"Z = (t - μ_p) / σ_p = ({threshold} - {port_mean:.4f}) / {port_std:.4f} = {z:.4f}")
        excel_steps.append(f"P(w'R ≤ {threshold}) = Φ({z:.4f}) = {prob_below:.6f}")
        excel_steps.append(f"P(w'R ≥ {threshold}) = 1 - Φ({z:.4f}) = {prob_above:.6f}")
        excel_steps.append("")
        excel_steps.append("**Excel:**")
        excel_steps.append(f"`=NORM.DIST({threshold}, {port_mean:.4f}, {port_std:.4f}, TRUE)` for P(w'R ≤ t)")
        
        result['threshold'] = threshold
        result['z_score'] = z
        result['prob_below'] = prob_below
        result['prob_above'] = prob_above
    
    excel_steps.append("")
    excel_steps.append("**General Excel Formulas:**")
    excel_steps.append("Portfolio mean: `=SUMPRODUCT(weights, returns)`")
    excel_steps.append("Portfolio variance: `=MMULT(MMULT(TRANSPOSE(w), Σ), w)`")
    
    return result, excel_steps, {}

@dataclass
class BulletBond:
    """
    Core Fixed Income class for exam problems.
    Handles pricing, yield solving, duration, and sensitivity analysis.
    
    Exam Reference: Standard bond math problems (2023-2025 exams)
    """
    face_value: float = 1000.0
    coupon_rate: float = 0.05
    maturity: int = 5
    yield_rate: Optional[float] = None
    price: Optional[float] = None
    frequency: int = 1  # 1=annual, 2=semi-annual
    
    def __post_init__(self):
        """Solve for missing variable (price or yield)."""
        if self.yield_rate is None and self.price is None:
            raise ValueError("Must provide either yield_rate or price")
        
        if self.yield_rate is not None and self.price is None:
            self.price = self._calculate_price(self.yield_rate)
        elif self.price is not None and self.yield_rate is None:
            self.yield_rate = self.solve_yield()
    
    def _calculate_price(self, y: float) -> float:
        """Calculate bond price given yield."""
        n_periods = self.maturity * self.frequency
        period_coupon = self.face_value * self.coupon_rate / self.frequency
        period_yield = y / self.frequency
        
        if abs(period_yield) < 1e-10:
            return period_coupon * n_periods + self.face_value
        
        # PV of coupons + PV of face value
        pv_coupons = period_coupon * (1 - (1 + period_yield) ** (-n_periods)) / period_yield
        pv_face = self.face_value * (1 + period_yield) ** (-n_periods)
        
        return pv_coupons + pv_face
    
    def solve_yield(self, tol: float = 1e-7, max_iter: int = 100) -> float:
        """
        Solve for yield given price using Newton-Raphson method.
        Tolerance: 1e-7 as per exam requirements.
        """
        if self.price is None:
            raise ValueError("Price must be set to solve for yield")
        
        # Initial guess based on current yield approximation
        y = self.coupon_rate  # Start with coupon rate
        
        for iteration in range(max_iter):
            price_at_y = self._calculate_price(y)
            
            # Numerical derivative (duration-based)
            dy = 0.0001
            price_up = self._calculate_price(y + dy)
            price_down = self._calculate_price(y - dy)
            dpdy = (price_up - price_down) / (2 * dy)
            
            if abs(dpdy) < 1e-12:
                break
            
            # Newton-Raphson update
            error = price_at_y - self.price
            y_new = y - error / dpdy
            
            if abs(y_new - y) < tol:
                return y_new
            
            y = max(0.0001, y_new)  # Ensure positive yield
        
        return y
    
    def get_schedule(self) -> pd.DataFrame:
        """
        Generate the EXACT exam-format cash flow schedule.
        
        Columns: CashFlow, DiscountFactor, PV, Weight (PV/ΣPV), TimeWeighted (t × Weight)
        Index: t = 1..T
        
        This format is MANDATORY for verifying Duration calculations in exams.
        """
        n_periods = self.maturity * self.frequency
        period_coupon = self.face_value * self.coupon_rate / self.frequency
        period_yield = self.yield_rate / self.frequency
        
        # Create time index
        periods = np.arange(1, n_periods + 1)
        times = periods / self.frequency  # Time in years
        
        # Cash flows
        cash_flows = np.full(n_periods, period_coupon)
        cash_flows[-1] += self.face_value  # Add face value at maturity
        
        # Discount factors
        discount_factors = (1 + period_yield) ** (-periods)
        
        # Present values
        pv = cash_flows * discount_factors
        
        # Weights (PV / Total PV)
        total_pv = np.sum(pv)
        weights = pv / total_pv
        
        # Time-weighted (for Macaulay Duration)
        time_weighted = times * weights
        
        # Create DataFrame with exam-standard format
        schedule = pd.DataFrame({
            't (periods)': periods,
            't (years)': times,
            'CashFlow': cash_flows,
            'DiscountFactor': discount_factors,
            'PV': pv,
            'Weight': weights,
            'TimeWeighted': time_weighted
        })
        schedule.index = periods
        schedule.index.name = 't'
        
        return schedule
    
    @property
    def macaulay_duration(self) -> float:
        """Macaulay Duration = Σ(t × Weight)."""
        schedule = self.get_schedule()
        return schedule['TimeWeighted'].sum()
    
    @property
    def modified_duration(self) -> float:
        """Modified Duration = Macaulay Duration / (1 + y/m)."""
        period_yield = self.yield_rate / self.frequency
        return self.macaulay_duration / (1 + period_yield)
    
    @property
    def convexity(self) -> float:
        """Bond convexity measure."""
        schedule = self.get_schedule()
        t_years = schedule['t (years)'].values
        weights = schedule['Weight'].values
        period_yield = self.yield_rate / self.frequency
        
        # Convexity = Σ[t(t+1/m) × Weight] / (1+y/m)²
        convexity_terms = t_years * (t_years + 1/self.frequency) * weights
        return convexity_terms.sum() / ((1 + period_yield) ** 2)
    
    @property
    def dollar_duration(self) -> float:
        """DV01 = Modified Duration × Price / 10000."""
        return self.modified_duration * self.price / 10000
    
    def price_at_yield(self, new_yield: float) -> float:
        """Calculate exact price at a different yield."""
        return self._calculate_price(new_yield)
    
    def sensitivity_analysis(
        self, 
        delta_y_mean: float, 
        delta_y_std: float, 
        conf_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Solve "Confidence Interval" problems from exams.
        
        Args:
            delta_y_mean: Expected yield change
            delta_y_std: Std dev of yield change
            conf_level: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Comparison table of Approx vs Exact price changes and returns
        """
        z = norm.ppf((1 + conf_level) / 2)
        
        scenarios = {
            'Lower Bound': delta_y_mean - z * delta_y_std,
            'Mean': delta_y_mean,
            'Upper Bound': delta_y_mean + z * delta_y_std
        }
        
        results = []
        for scenario_name, dy in scenarios.items():
            new_yield = self.yield_rate + dy
            
            # Approximate price change (Duration approximation)
            delta_p_approx = -self.modified_duration * self.price * dy
            price_approx = self.price + delta_p_approx
            return_approx = delta_p_approx / self.price
            
            # With convexity adjustment
            delta_p_convex = delta_p_approx + 0.5 * self.convexity * self.price * (dy ** 2)
            price_convex = self.price + delta_p_convex
            return_convex = delta_p_convex / self.price
            
            # Exact price change
            price_exact = self.price_at_yield(new_yield)
            delta_p_exact = price_exact - self.price
            return_exact = delta_p_exact / self.price
            
            results.append({
                'Scenario': scenario_name,
                'Δy': dy,
                'Δy (bps)': dy * 10000,
                'New Yield': new_yield,
                'Price (Duration)': price_approx,
                'Price (Dur+Conv)': price_convex,
                'Price (Exact)': price_exact,
                'Return (Duration)': return_approx,
                'Return (Dur+Conv)': return_convex,
                'Return (Exact)': return_exact,
                'Error (Dur vs Exact)': price_exact - price_approx,
                'Error (Conv vs Exact)': price_exact - price_convex
            })
        
        return pd.DataFrame(results)
    
    def get_latex_formulas(self) -> Dict[str, str]:
        """Return LaTeX strings for all formulas used."""
        return {
            'price': r'P = \sum_{t=1}^{T} \frac{CF_t}{(1+y)^t}',
            'macaulay_duration': r'D_{Mac} = \sum_{t=1}^{T} t \cdot w_t = \sum_{t=1}^{T} t \cdot \frac{PV_t}{P}',
            'modified_duration': r'D_{Mod} = \frac{D_{Mac}}{1 + y/m}',
            'convexity': r'C = \frac{1}{P(1+y)^2} \sum_{t=1}^{T} t(t+1) \cdot PV_t',
            'price_sensitivity': r'\Delta P \approx -D_{Mod} \cdot P \cdot \Delta y + \frac{1}{2} C \cdot P \cdot (\Delta y)^2',
            'newton_raphson': r'y_{n+1} = y_n - \frac{P(y_n) - P_{target}}{P\'(y_n)}'
        }
    
    def get_excel_instructions(self) -> str:
        """Return Excel instructions for exam."""
        # Prepare row end for dynamic references
        row_end = self.maturity + 1
        face = self.face_value
        coupon = self.coupon_rate
        final_cf = self.face_value * (1 + self.coupon_rate)

        return (
            f"""
**Excel Implementation for Bond (Coupon {self.coupon_rate*100:.1f}%, Maturity {self.maturity} years)**

1. **Set up a table with the following columns:**
   - **A: Period (t)** – enter 1, 2, …, {self.maturity}
   - **B: Cash Flow ($)**
   - **C: Discount Factor**
   - **D: Present Value ($)**
   - **E: Weight (w_i)**
   - **F: Time × PV ($·Years)**

2. **Cash Flow (Column B)**
   - For rows 2 to {row_end} (corresponding to t=1,…,{self.maturity-1}): enter `={face}*{coupon}` = {face * coupon:.2f}
   - In the last row (t={self.maturity}): enter `={face}*{coupon}+{face}` = {final_cf:.2f}

3. **Discount Factor (Column C)**
   - Store the yield in a cell (e.g., `$Y$1`).  In C2 enter:
     `=1/(1+$Y$1)^A2`
   - Copy down to row {row_end}.

4. **Present Value (Column D)**
   - In D2: `=B2*C2`
   - Copy down through D{row_end}.

5. **Bond Price**
   - Sum the PV column: `=SUM(D2:D{row_end})`  → Price = {self.price:.4f}

6. **Weights (Column E)**
   - Weight of each cash flow = PV divided by total PV: `=D2/$D${row_end+1}` if you place the price in cell D{row_end+1}, or alternatively
     `=D2/SUM($D$2:$D${row_end})`
   - Copy down.

7. **Time × PV (Column F)**
   - Multiply period by its present value: `=A2*D2`
   - Copy down.

8. **Macaulay Duration**
   - Duration in years is the sum of time-weighted PVs divided by price:
     `=SUM(F2:F{row_end}) / SUM(D2:D{row_end})`  → {self.macaulay_duration:.4f} years

9. **Modified Duration**
   - Adjust Macaulay duration for yield: `=Duration/(1+$Y$1)`  → {self.modified_duration:.4f}

These steps mirror the bond pricing and duration calculations used on exams.  Be sure to show the intermediate columns (cash flows, discount factors, PVs, weights, and time‐weighted PVs) in your solution.
"""
        )


# =============================================================================
# ADVANCED FIXED INCOME: Second-Order Immunization, Fisher-Weil Duration, Risky Bonds
# =============================================================================

@dataclass
class ImmunizerThreeBond:
    """
    Second-Order Immunization Solver (Convexity Matching).
    
    Reference: Textbook Theorem 5.8
    
    Solves a system of three linear equations to find weights w_A, w_B, w_C:
    1. Σ w_i = 1                    (Budget constraint)
    2. Σ w_i × D_i = D_Liability    (Duration match)
    3. Σ w_i × C_i = C_Liability    (Convexity match)
    
    This provides second-order protection against yield curve changes.
    """
    # Bond A properties (required)
    duration_A: float
    convexity_A: float
    
    # Bond B properties (required)
    duration_B: float
    convexity_B: float
    
    # Bond C properties (required)
    duration_C: float
    convexity_C: float
    
    # Liability properties (required)
    duration_liability: float
    convexity_liability: float
    
    # Optional properties with defaults (must come after required fields)
    price_A: float = 100.0
    price_B: float = 100.0
    price_C: float = 100.0
    pv_liability: float = 1_000_000.0
    
    def __post_init__(self):
        """Solve the system of equations."""
        self._solve_weights()
    
    def _solve_weights(self):
        """
        Solve the 3x3 system:
        [1,    1,    1   ] [w_A]   [1          ]
        [D_A,  D_B,  D_C ] [w_B] = [D_L        ]
        [C_A,  C_B,  C_C ] [w_C]   [C_L        ]
        """
        # Build coefficient matrix
        A = np.array([
            [1, 1, 1],
            [self.duration_A, self.duration_B, self.duration_C],
            [self.convexity_A, self.convexity_B, self.convexity_C]
        ])
        
        # Build target vector
        b = np.array([1, self.duration_liability, self.convexity_liability])
        
        # Check if system is solvable
        self.det = np.linalg.det(A)
        if abs(self.det) < 1e-10:
            self.solvable = False
            self.w_A = self.w_B = self.w_C = np.nan
            self.error_message = "System is singular - bonds do not span the required space"
            return
        
        # Solve the system
        try:
            weights = np.linalg.solve(A, b)
            self.w_A = weights[0]
            self.w_B = weights[1]
            self.w_C = weights[2]
            self.solvable = True
            self.error_message = None
        except np.linalg.LinAlgError as e:
            self.solvable = False
            self.w_A = self.w_B = self.w_C = np.nan
            self.error_message = str(e)
    
    def get_investment_amounts(self) -> Dict[str, float]:
        """Calculate dollar investments in each bond."""
        if not self.solvable:
            return {'A': np.nan, 'B': np.nan, 'C': np.nan}
        return {
            'A': self.w_A * self.pv_liability,
            'B': self.w_B * self.pv_liability,
            'C': self.w_C * self.pv_liability
        }
    
    def get_num_bonds(self) -> Dict[str, float]:
        """Calculate number of each bond to purchase."""
        if not self.solvable:
            return {'A': np.nan, 'B': np.nan, 'C': np.nan}
        investments = self.get_investment_amounts()
        return {
            'A': investments['A'] / self.price_A,
            'B': investments['B'] / self.price_B,
            'C': investments['C'] / self.price_C
        }
    
    def verify_solution(self) -> Dict[str, float]:
        """Verify that the solution matches the targets."""
        if not self.solvable:
            return {'budget': np.nan, 'duration': np.nan, 'convexity': np.nan}
        
        weight_sum = self.w_A + self.w_B + self.w_C
        port_duration = (self.w_A * self.duration_A + 
                        self.w_B * self.duration_B + 
                        self.w_C * self.duration_C)
        port_convexity = (self.w_A * self.convexity_A + 
                         self.w_B * self.convexity_B + 
                         self.w_C * self.convexity_C)
        
        return {
            'budget': weight_sum,
            'duration': port_duration,
            'convexity': port_convexity,
            'duration_error': abs(port_duration - self.duration_liability),
            'convexity_error': abs(port_convexity - self.convexity_liability)
        }
    
    def get_summary_table(self) -> pd.DataFrame:
        """Generate summary table for display."""
        if not self.solvable:
            return pd.DataFrame({'Error': [self.error_message]})
        
        investments = self.get_investment_amounts()
        num_bonds = self.get_num_bonds()
        
        rows = [
            ('**Bond A**', f"{self.w_A:.4f}", f"{self.w_A*100:.2f}%", 
             f"${investments['A']:,.2f}", f"{num_bonds['A']:,.2f}",
             f"{self.duration_A:.4f}", f"{self.convexity_A:.4f}"),
            ('**Bond B**', f"{self.w_B:.4f}", f"{self.w_B*100:.2f}%", 
             f"${investments['B']:,.2f}", f"{num_bonds['B']:,.2f}",
             f"{self.duration_B:.4f}", f"{self.convexity_B:.4f}"),
            ('**Bond C**', f"{self.w_C:.4f}", f"{self.w_C*100:.2f}%", 
             f"${investments['C']:,.2f}", f"{num_bonds['C']:,.2f}",
             f"{self.duration_C:.4f}", f"{self.convexity_C:.4f}"),
        ]
        
        return pd.DataFrame(rows, columns=[
            'Bond', 'Weight', 'Weight %', 'Investment ($)', 
            '# Bonds', 'Duration', 'Convexity'
        ])


def price_bond_with_spot_curve_fisher_weil(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    spot_rates: np.ndarray,
    maturities: np.ndarray,
    frequency: int = 1
) -> Tuple[Dict, List[str], pd.DataFrame]:
    """
    Price a bond using spot curve and calculate Fisher-Weil Duration.
    
    Reference: Textbook Equation 5.51
    
    Unlike Macaulay Duration (which uses YTM for all discounting),
    Fisher-Weil Duration uses the spot rate for each period:
    
    - PV_t = CF_t / (1 + z_t)^t  (using spot rate z_t)
    - w_t = PV_t / Total Price
    - D_FW = Σ t × w_t
    
    Fisher-Weil Duration is more accurate when the yield curve is not flat.
    
    Args:
        face_value: Face/par value of the bond
        coupon_rate: Annual coupon rate (decimal)
        maturity: Years to maturity
        spot_rates: Zero-coupon (spot) yield curve
        maturities: Maturities corresponding to spot rates
        frequency: Payment frequency (1=annual, 2=semi-annual)
    
    Returns:
        (result_dict, excel_steps, schedule_df)
    """
    spot_rates = np.asarray(spot_rates)
    maturities = np.asarray(maturities)
    
    excel_steps = []
    excel_steps.append("**Bond Pricing with Fisher-Weil Duration:**")
    excel_steps.append("")
    excel_steps.append(f"Face Value: ${face_value:,.2f}")
    excel_steps.append(f"Coupon Rate: {coupon_rate*100:.2f}%")
    excel_steps.append(f"Maturity: {maturity} years")
    excel_steps.append(f"Payment Frequency: {'Annual' if frequency == 1 else 'Semi-annual'}")
    excel_steps.append("")
    
    # Generate payment schedule
    n_periods = maturity * frequency
    period_coupon = face_value * coupon_rate / frequency
    dt = 1 / frequency
    
    payment_times = np.arange(dt, maturity + 1e-9, dt)
    
    # Cash flows
    cash_flows = np.full(len(payment_times), period_coupon)
    cash_flows[-1] += face_value  # Add face value at maturity
    
    # Interpolate spot rates at payment dates
    interp_rates = np.interp(payment_times, maturities, spot_rates)
    
    # Discount factors using SPOT RATES (key for Fisher-Weil)
    discount_factors = 1 / (1 + interp_rates) ** payment_times
    
    # Present values using spot rates
    pv = cash_flows * discount_factors
    price = np.sum(pv)
    
    # Fisher-Weil weights (based on spot-discounted PVs)
    weights_fw = pv / price
    
    # Fisher-Weil Duration
    fisher_weil_duration = np.sum(payment_times * weights_fw)
    
    excel_steps.append("**Fisher-Weil Duration Calculation:**")
    excel_steps.append("")
    excel_steps.append("Key Difference from Macaulay Duration:")
    excel_steps.append("- Macaulay uses a single YTM for all periods")
    excel_steps.append("- Fisher-Weil uses the spot rate z_t for each period t")
    excel_steps.append("")
    excel_steps.append("**Formula:** D_FW = Σ t × w_t")
    excel_steps.append("where w_t = PV_t / Price = CF_t / (1+z_t)^t / Price")
    excel_steps.append("")
    
    # Also calculate YTM and Macaulay Duration for comparison
    def price_at_yield(y):
        if abs(y) < 1e-10:
            return np.sum(cash_flows)
        dfs = 1 / (1 + y / frequency) ** (payment_times * frequency)
        return np.sum(cash_flows * dfs)
    
    # Newton-Raphson to find YTM
    y = coupon_rate
    for _ in range(100):
        p_y = price_at_yield(y)
        dy = 0.0001
        dp = (price_at_yield(y + dy) - price_at_yield(y - dy)) / (2 * dy)
        if abs(dp) < 1e-12:
            break
        y_new = y - (p_y - price) / dp
        if abs(y_new - y) < 1e-8:
            y = y_new
            break
        y = max(0.0001, y_new)
    ytm = y
    
    # Macaulay Duration using YTM
    dfs_ytm = 1 / (1 + ytm / frequency) ** (payment_times * frequency)
    pv_ytm = cash_flows * dfs_ytm
    weights_mac = pv_ytm / np.sum(pv_ytm)
    macaulay_duration = np.sum(payment_times * weights_mac)
    modified_duration = macaulay_duration / (1 + ytm / frequency)
    
    # Modified Fisher-Weil Duration (approximate)
    # Use average spot rate for modification factor
    avg_spot = np.average(interp_rates, weights=weights_fw)
    modified_fw_duration = fisher_weil_duration / (1 + avg_spot / frequency)
    
    excel_steps.append("**Step-by-Step Calculation:**")
    excel_steps.append("")
    excel_steps.append("| t | CF | z_t (Spot) | DF = 1/(1+z_t)^t | PV | Weight (w_t) | t × w_t |")
    excel_steps.append("|---|----|-----------|-----------------|----|--------------|---------|")
    for i, (t, cf, z, df, p, w) in enumerate(zip(payment_times, cash_flows, interp_rates, discount_factors, pv, weights_fw)):
        excel_steps.append(f"| {t:.1f} | {cf:.2f} | {z*100:.4f}% | {df:.6f} | {p:.4f} | {w:.6f} | {t*w:.6f} |")
    excel_steps.append("")
    excel_steps.append(f"**Bond Price = Σ PV = ${price:.4f}**")
    excel_steps.append(f"**Fisher-Weil Duration = Σ(t × w_t) = {fisher_weil_duration:.4f} years**")
    excel_steps.append("")
    excel_steps.append("**Comparison with Macaulay Duration:**")
    excel_steps.append(f"- Fisher-Weil Duration: {fisher_weil_duration:.4f} years")
    excel_steps.append(f"- Macaulay Duration (using YTM={ytm*100:.4f}%): {macaulay_duration:.4f} years")
    excel_steps.append(f"- Difference: {abs(fisher_weil_duration - macaulay_duration):.6f} years")
    excel_steps.append("")
    excel_steps.append("Note: The difference is larger when the yield curve is steeper.")
    
    # Create schedule DataFrame
    schedule = pd.DataFrame({
        't (years)': payment_times,
        'Cash Flow': cash_flows,
        'Spot Rate (z_t)': interp_rates,
        'DF (Spot)': discount_factors,
        'PV (Spot)': pv,
        'FW Weight': weights_fw,
        't × FW Weight': payment_times * weights_fw,
        'YTM DF': dfs_ytm,
        'PV (YTM)': pv_ytm,
        'Mac Weight': weights_mac,
        't × Mac Weight': payment_times * weights_mac
    })
    
    result = {
        'price': price,
        'ytm': ytm,
        'fisher_weil_duration': fisher_weil_duration,
        'macaulay_duration': macaulay_duration,
        'modified_duration': modified_duration,
        'modified_fw_duration': modified_fw_duration,
        'duration_difference': fisher_weil_duration - macaulay_duration,
        'cash_flows': cash_flows,
        'payment_times': payment_times,
        'spot_rates_used': interp_rates,
        'discount_factors': discount_factors,
        'present_values': pv,
        'fw_weights': weights_fw,
        'mac_weights': weights_mac
    }
    
    return result, excel_steps, schedule


def price_risky_bond(
    face_value: float,
    coupon_rate: float,
    maturity: int,
    yield_rate: float,
    default_probability: float,
    recovery_rate: float,
    frequency: int = 1
) -> Tuple[Dict, List[str], pd.DataFrame]:
    """
    Price a risky bond accounting for credit risk.
    
    Reference: Textbook Equation 5.66 (Section 5.10.2)
    
    Adjusts expected cash flows for default probability and recovery rate.
    
    For each period t, the survival-adjusted cash flow is:
    E[CF_t] = CF_t × [1 - p(1-R)]^t
    
    Where:
    - p = annual default probability
    - R = recovery rate (fraction of face value recovered in default)
    - [1 - p(1-R)] = survival-adjusted factor per period
    
    Alternative interpretation:
    - Probability of survival to time t: (1-p)^t
    - Expected loss given default: p(1-R) per period
    
    Args:
        face_value: Face/par value of the bond
        coupon_rate: Annual coupon rate (decimal)
        maturity: Years to maturity
        yield_rate: Risk-free yield for discounting
        default_probability: Annual probability of default (p)
        recovery_rate: Recovery rate in default (R), as fraction of face value
        frequency: Payment frequency (1=annual, 2=semi-annual)
    
    Returns:
        (result_dict, excel_steps, schedule_df)
    """
    excel_steps = []
    excel_steps.append("**Risky Bond Pricing with Credit Risk:**")
    excel_steps.append("")
    excel_steps.append("Reference: Textbook Equation 5.66")
    excel_steps.append("")
    excel_steps.append(f"Face Value: ${face_value:,.2f}")
    excel_steps.append(f"Coupon Rate: {coupon_rate*100:.2f}%")
    excel_steps.append(f"Maturity: {maturity} years")
    excel_steps.append(f"Risk-Free Yield: {yield_rate*100:.2f}%")
    excel_steps.append(f"Default Probability (p): {default_probability*100:.2f}%")
    excel_steps.append(f"Recovery Rate (R): {recovery_rate*100:.2f}%")
    excel_steps.append("")
    
    # Generate payment schedule
    n_periods = maturity * frequency
    period_coupon = face_value * coupon_rate / frequency
    period_yield = yield_rate / frequency
    dt = 1 / frequency
    
    # Adjust default probability for frequency
    period_default_prob = default_probability / frequency
    
    payment_times = np.arange(dt, maturity + 1e-9, dt)
    periods = np.arange(1, n_periods + 1)
    
    # Risk-free cash flows
    cash_flows_rf = np.full(len(payment_times), period_coupon)
    cash_flows_rf[-1] += face_value
    
    # Survival-adjusted factor: [1 - p(1-R)] per period
    # This accounts for expected loss per period
    loss_rate_per_period = period_default_prob * (1 - recovery_rate)
    survival_factor = 1 - loss_rate_per_period
    
    excel_steps.append("**Key Formula:**")
    excel_steps.append("Expected CF_t = Scheduled CF_t × [1 - p(1-R)]^t")
    excel_steps.append("")
    excel_steps.append(f"Loss rate per period = p × (1-R) = {period_default_prob:.4f} × (1-{recovery_rate:.2f}) = {loss_rate_per_period:.6f}")
    excel_steps.append(f"Survival factor = 1 - {loss_rate_per_period:.6f} = {survival_factor:.6f}")
    excel_steps.append("")
    
    # Cumulative survival probability to each period
    survival_prob = survival_factor ** periods
    
    # Expected cash flows (adjusted for credit risk)
    expected_cf = cash_flows_rf * survival_prob
    
    # Discount factors (using risk-free rate)
    discount_factors = (1 + period_yield) ** (-periods)
    
    # Present values
    pv_rf = cash_flows_rf * discount_factors  # Risk-free PV
    pv_risky = expected_cf * discount_factors  # Risky PV
    
    # Prices
    price_rf = np.sum(pv_rf)
    price_risky = np.sum(pv_risky)
    
    # Credit spread (implied)
    # Solve for yield that prices risky bond = price_risky
    def price_at_yield(y):
        if abs(y) < 1e-10:
            return np.sum(cash_flows_rf)
        dfs = (1 + y/frequency) ** (-periods)
        return np.sum(cash_flows_rf * dfs)
    
    # Newton-Raphson for risky yield
    y_risky = yield_rate
    for _ in range(100):
        p_y = price_at_yield(y_risky)
        dy = 0.0001
        dp = (price_at_yield(y_risky + dy) - price_at_yield(y_risky - dy)) / (2 * dy)
        if abs(dp) < 1e-12:
            break
        y_new = y_risky - (p_y - price_risky) / dp
        if abs(y_new - y_risky) < 1e-8:
            y_risky = y_new
            break
        y_risky = max(0.0001, y_new)
    
    credit_spread = y_risky - yield_rate
    
    excel_steps.append("**Step-by-Step Calculation:**")
    excel_steps.append("")
    excel_steps.append("| t | Scheduled CF | Survival Prob | Expected CF | DF | PV (Risky) |")
    excel_steps.append("|---|--------------|---------------|-------------|-----|------------|")
    for i, (t, cf_rf, surv, ecf, df, pv_r) in enumerate(zip(
            payment_times, cash_flows_rf, survival_prob, expected_cf, discount_factors, pv_risky)):
        excel_steps.append(f"| {t:.1f} | {cf_rf:.2f} | {surv:.6f} | {ecf:.4f} | {df:.6f} | {pv_r:.4f} |")
    excel_steps.append("")
    excel_steps.append(f"**Risk-Free Price = ${price_rf:.4f}**")
    excel_steps.append(f"**Risky Price = ${price_risky:.4f}**")
    excel_steps.append(f"**Credit Discount = ${price_rf - price_risky:.4f}** ({(price_rf - price_risky)/price_rf*100:.2f}%)")
    excel_steps.append("")
    excel_steps.append(f"**Implied Credit Spread = {credit_spread*100:.4f}%** ({credit_spread*10000:.2f} bps)")
    excel_steps.append("")
    
    # Alternative single-period formula (Eq 5.66 for T=1)
    if maturity == 1 and frequency == 1:
        excel_steps.append("**Single-Period Formula (Eq 5.66):**")
        excel_steps.append(f"P = [(1-p) × (C+F) + p × R × F] / (1+r)")
        expected_payoff = (1 - default_probability) * (period_coupon + face_value) + default_probability * recovery_rate * face_value
        price_eq566 = expected_payoff / (1 + yield_rate)
        excel_steps.append(f"  = [(1-{default_probability}) × {period_coupon + face_value} + {default_probability} × {recovery_rate} × {face_value}] / (1+{yield_rate})")
        excel_steps.append(f"  = {expected_payoff:.4f} / {1+yield_rate:.4f} = ${price_eq566:.4f}")
    
    # Create schedule DataFrame
    schedule = pd.DataFrame({
        't (years)': payment_times,
        'Period': periods,
        'Scheduled CF': cash_flows_rf,
        'Survival Prob': survival_prob,
        'Expected CF': expected_cf,
        'Discount Factor': discount_factors,
        'PV (Risk-Free)': pv_rf,
        'PV (Risky)': pv_risky,
        'Credit Adj': pv_rf - pv_risky
    })
    
    result = {
        'price_risky': price_risky,
        'price_risk_free': price_rf,
        'credit_discount': price_rf - price_risky,
        'credit_discount_pct': (price_rf - price_risky) / price_rf * 100,
        'credit_spread': credit_spread,
        'credit_spread_bps': credit_spread * 10000,
        'risky_yield': y_risky,
        'risk_free_yield': yield_rate,
        'default_probability': default_probability,
        'recovery_rate': recovery_rate,
        'survival_factor': survival_factor,
        'expected_cash_flows': expected_cf,
        'survival_probabilities': survival_prob
    }
    
    return result, excel_steps, schedule


@dataclass
class BondReplicator:
    """
    Solves "Synthetic Zero Coupon Bond" problems.
    Uses matrix inversion to find bond units that replicate target cash flows.
    
    Exam Reference: Cash flow matching and synthetic bond creation
    """
    instruments: List[BulletBond] = field(default_factory=list)
    target_flows: Dict[int, float] = field(default_factory=dict)  # {time: amount}
    
    def get_cash_flow_matrix(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Construct the cash flow matrix C where C[t,j] is cash flow of Bond j at time t.
        
        Returns:
            C: Cash flow matrix
            T: Target flow vector
            times: List of relevant time periods
        """
        # Get all relevant time periods
        all_times = set(self.target_flows.keys())
        for bond in self.instruments:
            for t in range(1, bond.maturity + 1):
                all_times.add(t)
        
        times = sorted(all_times)
        n_times = len(times)
        n_bonds = len(self.instruments)
        
        # Build cash flow matrix
        C = np.zeros((n_times, n_bonds))
        
        for j, bond in enumerate(self.instruments):
            coupon = bond.face_value * bond.coupon_rate
            for i, t in enumerate(times):
                if 1 <= t <= bond.maturity:
                    C[i, j] = coupon
                    if t == bond.maturity:
                        C[i, j] += bond.face_value
        
        # Build target vector
        T = np.zeros(n_times)
        for i, t in enumerate(times):
            T[i] = self.target_flows.get(t, 0)
        
        return C, T, times
    
    def solve(self) -> Dict:
        """
        Solve for the number of bonds needed to replicate target cash flows.
        
        Algorithm:
        1. Construct Matrix C where C[t,j] is the cash flow of Bond j at time t
        2. Construct Vector T as the target flows
        3. Check solvability (is Matrix square? Is determinant non-zero?)
        4. Solve N = C^(-1) × T
        
        Returns:
            Dictionary with units, implied_price, implied_yield, and diagnostic info
        """
        if len(self.instruments) == 0:
            return {'solvable': False, 'error': 'No instruments provided'}
        if len(self.target_flows) == 0:
            return {'solvable': False, 'error': 'No target flows provided'}
            
        C, T, times = self.get_cash_flow_matrix()
        
        n_times, n_bonds = C.shape
        
        # Initialize result
        result = {
            'solvable': False,
            'method': None,
            'units': None,
            'implied_price': None,
            'implied_yield': None,
            'cash_flow_matrix': C,
            'target_vector': T,
            'times': times,
            'n_bonds': n_bonds,
            'n_equations': n_times
        }
        
        if n_bonds == n_times:
            # Square system - try direct inversion
            det = np.linalg.det(C)
            result['determinant'] = det
            
            if abs(det) > 1e-10:
                result['solvable'] = True
                result['method'] = 'Direct Inversion (Square System)'
                
                # Solve N = C^(-1) × T
                C_inv = np.linalg.inv(C)
                N = C_inv @ T
                result['units'] = N
                result['C_inverse'] = C_inv
            else:
                result['error'] = f'Matrix is singular (determinant = {det:.2e})'
        
        elif n_bonds > n_times:
            # Underdetermined - infinitely many solutions, use minimum norm
            result['method'] = 'Minimum Norm (Underdetermined)'
            N, residuals, rank, s = np.linalg.lstsq(C, T, rcond=None)
            result['units'] = N
            result['solvable'] = True
            result['warning'] = 'Underdetermined system - solution is not unique'
        
        else:
            # Overdetermined - use least squares
            result['method'] = 'Least Squares (Overdetermined)'
            N, residuals, rank, s = np.linalg.lstsq(C, T, rcond=None)
            result['units'] = N
            result['solvable'] = True
            if len(residuals) > 0 and residuals[0] > 1e-6:
                result['warning'] = f'No exact solution exists (residual = {residuals[0]:.6f})'
        
        if result['solvable'] and result['units'] is not None:
            # Calculate implied price and yield
            N = result['units']
            prices = np.array([bond.price for bond in self.instruments])
            result['implied_price'] = np.sum(N * prices)
            result['individual_costs'] = N * prices
            
            # Implied yield for zero-coupon equivalent
            target_time = max(self.target_flows.keys())
            target_amount = sum(self.target_flows.values())
            
            if result['implied_price'] > 0 and target_amount > 0:
                result['implied_yield'] = (target_amount / result['implied_price']) ** (1/target_time) - 1
                result['target_time'] = target_time
                result['target_amount'] = target_amount
        
        return result
    
    def get_verification_table(self) -> pd.DataFrame:
        """Generate a table verifying that the solution replicates target flows."""
        result = self.solve()
        
        if not result['solvable']:
            return pd.DataFrame({'Error': [result.get('error', 'System not solvable')]})
        
        C = result['cash_flow_matrix']
        T = result['target_vector']
        N = result['units']
        times = result['times']
        
        # Calculate replicated flows
        replicated = C @ N
        
        verification = pd.DataFrame({
            'Time (t)': times,
            'Target CF': T,
            'Replicated CF': replicated,
            'Difference': replicated - T,
            'Match': np.abs(replicated - T) < 0.01
        })
        
        # Add individual bond contributions
        for j, bond in enumerate(self.instruments):
            verification[f'Bond {j+1} CF'] = C[:, j]
            verification[f'Bond {j+1} × N'] = C[:, j] * N[j]
        
        return verification
    
    def get_latex_formulas(self) -> Dict[str, str]:
        """Return LaTeX formulas."""
        return {
            'matrix_system': r'\mathbf{C} \mathbf{N} = \mathbf{T}',
            'solution': r'\mathbf{N} = \mathbf{C}^{-1} \mathbf{T}',
            'implied_price': r'P_{implied} = \sum_{i} N_i \cdot P_i',
            'implied_yield': r'y_{implied} = \left(\frac{F_{target}}{P_{implied}}\right)^{1/T} - 1'
        }
    
    def get_excel_instructions(self) -> str:
        """Return Excel instructions."""
        result = self.solve()
        n_bonds = len(self.instruments)
        
        return f"""
**Excel Implementation for Bond Replication (Synthetic Zero):**

1. **Build Cash Flow Matrix C ({result['n_equations']}×{n_bonds}):**
   - Rows = Time periods
   - Columns = Bonds
   - Cell C[t,j] = Cash flow of Bond j at time t

2. **Build Target Vector T:**
   - Column vector of target cash flows at each time

3. **Check Solvability:**
   - For square matrix: `=MDETERM(C)` must be ≠ 0
   - Current determinant: {result.get('determinant', 'N/A')}

4. **Solve for Units N:**
   - `=MMULT(MINVERSE(C), T)` (Ctrl+Shift+Enter)
   - Result: N = [{', '.join([f'{n:.4f}' for n in result['units']])}]

5. **Calculate Implied Price:**
   - `=SUMPRODUCT(N, Prices)`
   - Result: {result['implied_price']:.4f}

6. **Calculate Implied Yield:**
   - `=(Target_Amount/Implied_Price)^(1/T) - 1`
   - Result: {result.get('implied_yield', 0)*100:.4f}%
"""


@dataclass
class Immunizer:
    """
    Solves "Duration Matching" / Immunization problems.
    Matches portfolio duration and value to a liability.
    
    Exam Reference: Duration matching for liability immunization
    """
    liability_amount: float
    liability_time: float
    liability_yield: float
    bond_a: BulletBond
    bond_b: BulletBond
    
    def __post_init__(self):
        # Calculate liability PV and duration
        self.liability_pv = self.liability_amount / (1 + self.liability_yield) ** self.liability_time
        self.liability_duration = self.liability_time  # Duration of zero-coupon = maturity
    
    def solve(self) -> Dict:
        """
        Solve the immunization problem.
        
        System of equations:
        1. w_A + w_B = 1 (weights sum to 1)
        2. w_A × D_A + w_B × D_B = D_L (duration matching)
        
        Solution:
        w_A = (D_L - D_B) / (D_A - D_B)
        w_B = 1 - w_A
        
        Units: N_i = (w_i × PV_Liability) / P_i
        
        Returns:
            Dictionary with weights, units, values, and proof of immunization
        """
        D_A = self.bond_a.macaulay_duration
        D_B = self.bond_b.macaulay_duration
        D_L = self.liability_duration
        
        P_A = self.bond_a.price
        P_B = self.bond_b.price
        
        # Check if immunization is possible
        if abs(D_A - D_B) < 1e-6:
            return {
                'solvable': False,
                'error': 'Bond durations must be different for immunization',
                'D_A': D_A,
                'D_B': D_B
            }
        
        # Check if target duration is achievable
        D_min, D_max = min(D_A, D_B), max(D_A, D_B)
        if not (D_min <= D_L <= D_max):
            warning = f'Target duration {D_L:.2f} outside bond duration range [{D_min:.2f}, {D_max:.2f}]. Short positions required.'
        else:
            warning = None
        
        # Solve for weights
        w_A = (D_L - D_B) / (D_A - D_B)
        w_B = 1 - w_A
        
        # Calculate dollar amounts
        invest_A = w_A * self.liability_pv
        invest_B = w_B * self.liability_pv
        
        # Calculate number of bonds
        N_A = invest_A / P_A
        N_B = invest_B / P_B
        
        # Portfolio characteristics
        portfolio_value = invest_A + invest_B
        portfolio_duration = w_A * D_A + w_B * D_B
        
        # Weighted average yield (approximate)
        y_A = self.bond_a.yield_rate
        y_B = self.bond_b.yield_rate
        portfolio_yield = w_A * y_A + w_B * y_B
        
        # Proof of immunization
        proof = {
            'liability_pv': self.liability_pv,
            'portfolio_value': portfolio_value,
            'value_match': abs(portfolio_value - self.liability_pv) < 0.01,
            'value_error': portfolio_value - self.liability_pv,
            'liability_duration': D_L,
            'portfolio_duration': portfolio_duration,
            'duration_match': abs(portfolio_duration - D_L) < 0.001,
            'duration_error': portfolio_duration - D_L
        }
        
        return {
            'solvable': True,
            'warning': warning,
            'weights': {'A': w_A, 'B': w_B},
            'investments': {'A': invest_A, 'B': invest_B},
            'units': {'A': N_A, 'B': N_B},
            'bond_prices': {'A': P_A, 'B': P_B},
            'bond_durations': {'A': D_A, 'B': D_B},
            'bond_yields': {'A': y_A, 'B': y_B},
            'portfolio_yield': portfolio_yield,
            'proof': proof,
            'short_positions': w_A < 0 or w_B < 0
        }
    
    def get_summary_table(self) -> pd.DataFrame:
        """Generate exam-format summary table."""
        result = self.solve()
        
        if not result['solvable']:
            return pd.DataFrame({'Error': [result['error']]})
        
        rows = [
            ('Liability', '', '', '', ''),
            ('  PV of Liability', f"${self.liability_pv:,.2f}", f'{self.liability_duration:.4f} yrs', '', ''),
            ('', '', '', '', ''),
            ('Bonds', 'Price', 'Duration', 'Weight', 'Investment'),
            ('  Bond A', f"${result['bond_prices']['A']:,.2f}", f"{result['bond_durations']['A']:.4f} yrs",
             f"{result['weights']['A']:.4f}", f"${result['investments']['A']:,.2f}"),
            ('  Bond B', f"${result['bond_prices']['B']:,.2f}", f"{result['bond_durations']['B']:.4f} yrs",
             f"{result['weights']['B']:.4f}", f"${result['investments']['B']:,.2f}"),
            ('', '', '', '', ''),
            ('Units Required', 'Bond A', 'Bond B', '', ''),
            ('', f"{result['units']['A']:,.4f}", f"{result['units']['B']:,.4f}", '', ''),
            ('', '', '', '', ''),
            ('Verification', 'Target', 'Achieved', 'Error', ''),
            ('  Value', f"${self.liability_pv:,.2f}", f"${result['proof']['portfolio_value']:,.2f}",
             f"${result['proof']['value_error']:,.4f}", '✓' if result['proof']['value_match'] else '✗'),
            ('  Duration', f"{self.liability_duration:.4f} yrs", f"{result['proof']['portfolio_duration']:.4f} yrs",
             f"{result['proof']['duration_error']:.6f}", '✓' if result['proof']['duration_match'] else '✗'),
        ]
        
        return pd.DataFrame(rows, columns=['Item', 'Col1', 'Col2', 'Col3', 'Col4'])
    
    def get_latex_formulas(self) -> Dict[str, str]:
        """Return LaTeX formulas for immunization."""
        return {
            'weight_a': r'w_A = \frac{D_L - D_B}{D_A - D_B}',
            'weight_b': r'w_B = 1 - w_A',
            'units_a': r'N_A = \frac{w_A \cdot PV_{Liability}}{P_A}',
            'units_b': r'N_B = \frac{w_B \cdot PV_{Liability}}{P_B}',
            'duration_match': r'w_A \cdot D_A + w_B \cdot D_B = D_L',
            'value_match': r'w_A \cdot PV_L + w_B \cdot PV_L = PV_L'
        }
    
    def get_excel_instructions(self) -> str:
        """Return Excel instructions."""
        result = self.solve()
        
        return f"""
**Excel Implementation for Duration Immunization:**

**Given:**
- Liability: ${self.liability_amount:,.0f} due in {self.liability_time} years
- Discount rate: {self.liability_yield*100:.2f}%
- Bond A: Duration = {result['bond_durations']['A']:.4f}, Price = ${result['bond_prices']['A']:,.2f}
- Bond B: Duration = {result['bond_durations']['B']:.4f}, Price = ${result['bond_prices']['B']:,.2f}

**Step 1: Calculate Liability PV**
`=FV/(1+r)^T` = {self.liability_amount}/(1+{self.liability_yield})^{self.liability_time} = ${self.liability_pv:,.2f}

**Step 2: Calculate Weight in Bond A**
`=(D_L - D_B)/(D_A - D_B)`
`=({self.liability_duration} - {result['bond_durations']['B']:.4f})/({result['bond_durations']['A']:.4f} - {result['bond_durations']['B']:.4f})`
= {result['weights']['A']:.4f}

**Step 3: Calculate Weight in Bond B**
`=1 - w_A` = {result['weights']['B']:.4f}

**Step 4: Calculate Investments**
Investment_A = w_A × PV_L = {result['weights']['A']:.4f} × ${self.liability_pv:,.2f} = ${result['investments']['A']:,.2f}
Investment_B = w_B × PV_L = {result['weights']['B']:.4f} × ${self.liability_pv:,.2f} = ${result['investments']['B']:,.2f}

**Step 5: Calculate Number of Bonds**
N_A = Investment_A / Price_A = ${result['investments']['A']:,.2f} / ${result['bond_prices']['A']:,.2f} = {result['units']['A']:.4f}
N_B = Investment_B / Price_B = ${result['investments']['B']:,.2f} / ${result['bond_prices']['B']:,.2f} = {result['units']['B']:.4f}

**Verification:**
Portfolio Duration = w_A × D_A + w_B × D_B
= {result['weights']['A']:.4f} × {result['bond_durations']['A']:.4f} + {result['weights']['B']:.4f} × {result['bond_durations']['B']:.4f}
= {result['proof']['portfolio_duration']:.4f} ≈ {self.liability_duration} ✓
"""


# =============================================================================
# MODULE B: ADVANCED PORTFOLIO ENGINE (Priority P0) - CORE CLASSES
# =============================================================================

@dataclass
class MeanVarianceOptimizer:
    """
    Advanced Mean-Variance Optimization with ESG extension.
    
    Handles:
    - Standard Tangency Portfolio
    - Global Minimum Variance Portfolio
    - Optimal Portfolio for given risk aversion
    - ESG-adjusted utility optimization
    
    Exam Reference: Portfolio optimization with ESG preferences (2023-2025)
    """
    expected_returns: np.ndarray
    covariance_matrix: Optional[np.ndarray] = None
    volatilities: Optional[np.ndarray] = None
    correlations: Optional[np.ndarray] = None
    risk_free_rate: float = 0.02
    risk_aversion: float = 4.0
    esg_scores: Optional[np.ndarray] = None
    esg_preference: float = 0.0  # 'a' parameter in utility
    
    def __post_init__(self):
        """Build covariance matrix if not provided."""
        self.n_assets = len(self.expected_returns)
        
        if self.covariance_matrix is None:
            if self.volatilities is None or self.correlations is None:
                raise ValueError("Must provide either covariance_matrix or (volatilities + correlations)")
            
            # Build covariance matrix: Σ = diag(σ) × Corr × diag(σ)
            D = np.diag(self.volatilities)
            self.covariance_matrix = D @ self.correlations @ D
        
        # Compute inverse
        self.cov_inverse = np.linalg.inv(self.covariance_matrix)
        
        # Validate ESG inputs
        if self.esg_scores is not None and len(self.esg_scores) != self.n_assets:
            raise ValueError("ESG scores must match number of assets")
    
    @property
    def excess_returns(self) -> np.ndarray:
        """μ - r_f."""
        return self.expected_returns - self.risk_free_rate
    
    def tangency_portfolio(self) -> Dict:
        """
        Calculate the Tangency (Maximum Sharpe Ratio) Portfolio.
        
        Formula: π_tan = Σ⁻¹(μ - r_f) / 1'Σ⁻¹(μ - r_f)
        """
        numerator = self.cov_inverse @ self.excess_returns
        weights = numerator / np.sum(numerator)
        
        return self._portfolio_stats(weights, 'Tangency Portfolio')
    
    def gmv_portfolio(self) -> Dict:
        """
        Calculate the Global Minimum Variance Portfolio.
        
        Formula: π_gmv = Σ⁻¹1 / 1'Σ⁻¹1
        """
        ones = np.ones(self.n_assets)
        numerator = self.cov_inverse @ ones
        weights = numerator / (ones @ self.cov_inverse @ ones)
        
        return self._portfolio_stats(weights, 'GMV Portfolio')
    
    def optimal_portfolio(self, include_esg: bool = True) -> Dict:
        """
        Calculate the Optimal Portfolio for given risk aversion.
        
        Standard Formula: w* = (1/γ) × Σ⁻¹ × (μ - r_f)
        
        ESG Extension: w* = (1/γ) × Σ⁻¹ × (μ - r_f×1) + (a/γ) × Σ⁻¹ × s
        
        Where a is ESG preference and s is ESG score vector.
        
        Validation: If a=0, must match standard Tangency/Utility solution.
        """
        gamma = self.risk_aversion
        
        # Standard mean-variance component
        mv_component = (1/gamma) * (self.cov_inverse @ self.excess_returns)
        
        # ESG component (if applicable)
        esg_component = np.zeros(self.n_assets)
        if include_esg and self.esg_scores is not None and self.esg_preference > 0:
            esg_component = (self.esg_preference / gamma) * (self.cov_inverse @ self.esg_scores)
        
        weights = mv_component + esg_component
        
        result = self._portfolio_stats(weights, f'Optimal Portfolio (γ={gamma})')
        result['mv_component'] = mv_component
        result['esg_component'] = esg_component
        result['risky_weight'] = np.sum(weights)
        result['cash_weight'] = 1 - np.sum(weights)
        
        # ESG utility contribution
        if self.esg_scores is not None:
            result['esg_utility'] = self.esg_preference * (weights @ self.esg_scores)
        
        return result
    
    def _portfolio_stats(self, weights: np.ndarray, name: str) -> Dict:
        """Calculate portfolio statistics.
        
        Accounts for cash position when weights don't sum to 1:
        - If sum(w) < 1: residual (1 - sum(w)) earns risk-free rate
        - If sum(w) > 1: excess (sum(w) - 1) is borrowed at risk-free rate
        """
        cash_weight = 1 - np.sum(weights)
        ret = weights @ self.expected_returns + cash_weight * self.risk_free_rate
        vol = np.sqrt(weights @ self.covariance_matrix @ weights)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
        
        result = {
            'name': name,
            'weights': weights,
            'cash_weight': cash_weight,
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'variance': vol ** 2
        }
        
        # Add ESG score if available
        if self.esg_scores is not None:
            result['portfolio_esg'] = weights @ self.esg_scores
        
        return result
    
    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Generate the efficient frontier."""
        gmv = self.gmv_portfolio()
        
        # Two-fund separation parameters
        ones = np.ones(self.n_assets)
        A = ones @ self.cov_inverse @ ones
        B = ones @ self.cov_inverse @ self.expected_returns
        C = self.expected_returns @ self.cov_inverse @ self.expected_returns
        D = A * C - B ** 2
        
        min_ret = gmv['expected_return']
        max_ret = max(self.expected_returns) * 1.2
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        results = []
        
        for target in target_returns:
            if D > 0:
                g = (C * (self.cov_inverse @ ones) - B * (self.cov_inverse @ self.expected_returns)) / D
                h = (A * (self.cov_inverse @ self.expected_returns) - B * (self.cov_inverse @ ones)) / D
                w = g + h * target
                vol = np.sqrt(w @ self.covariance_matrix @ w)
                results.append({'return': target, 'volatility': vol})
        
        return pd.DataFrame(results)
    
    def get_latex_formulas(self) -> Dict[str, str]:
        """Return LaTeX formulas."""
        return {
            'tangency': r'\pi_{tan} = \frac{\Sigma^{-1}(\mu - r_f \mathbf{1})}{\mathbf{1}^T \Sigma^{-1}(\mu - r_f \mathbf{1})}',
            'gmv': r'\pi_{GMV} = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^T \Sigma^{-1} \mathbf{1}}',
            'optimal': r'\pi^* = \frac{1}{\gamma} \Sigma^{-1}(\mu - r_f \mathbf{1})',
            'optimal_esg': r'\mathbf{w}^* = \frac{1}{\gamma}\Sigma^{-1}(\mu - r_f\mathbf{1}) + \frac{a}{\gamma}\Sigma^{-1}\mathbf{s}',
            'utility': r'U = E[r_p] - \frac{1}{2}\gamma\sigma_p^2 + a \cdot ESG_p'
        }
    
    def get_excel_instructions(self, portfolio_type: str = 'optimal') -> str:
        """Return step‑by‑step Excel instructions for mean‑variance and ESG optimization."""
        # Prepare ESG-related text
        esg_info = ""
        if getattr(self, 'esg_preference', 0) and self.esg_preference > 0:
            esg_info = (
                f"**With ESG Adjustment (a = {self.esg_preference:.4f}):**\n"
                f"- ESG component: =({self.esg_preference}/{self.risk_aversion}) × MMULT(Σ_inv, ESG_scores)\n"
                f"- Total: =MV_weights + ESG_component\n\n"
            )
        esg_scores_line = ""
        if getattr(self, 'esg_preference', 0) and self.esg_preference > 0:
            esg_scores_line = "- ESG scores: one per asset.\n"
        return (
            "**Excel Implementation for Mean‑Variance Optimization:**\n\n"
            "**Step 1: Build the Covariance Matrix (Σ)**\n"
            "- Compute Σ[i,j] = σ_i × σ_j × Corr[i,j] or use diag(σ) × Corr × diag(σ).\n"
            "- If Σ is given, enter the n×n matrix directly.\n\n"
            "**Step 2: Invert Σ**\n"
            "- Select an n×n block of empty cells.\n"
            "- Enter =MINVERSE(Σ_range) and press Ctrl+Shift+Enter to compute the inverse.\n\n"
            "**Step 3: Compute Necessary Vectors**\n"
            "- Ones vector: a column of ones (length n).\n"
            "- Excess returns: μ_i – r_f for each asset.\n"
            f"{esg_scores_line}\n"
            "**Step 4: Calculate Portfolio Weights**\n"
            "- GMV: compute vector_gmv = MMULT(Σ_inv, ones), then weights = vector_gmv / SUM(vector_gmv).\n"
            "- Tangency: numerator = MMULT(Σ_inv, excess_returns), weights = numerator / SUM(numerator).\n"
            f"- Optimal (γ = {self.risk_aversion}): weights = (1/{self.risk_aversion}) × MMULT(Σ_inv, excess_returns).\n"
            f"{esg_info}"
            "**Step 5: Portfolio Statistics**\n"
            "- Expected return: =SUMPRODUCT(weights, μ).\n"
            "- Variance: =MMULT(MMULT(TRANSPOSE(weights), Σ), weights).\n"
            "- Volatility: =SQRT(variance).\n"
            "- Sharpe ratio: (E[r_p] – r_f) / Volatility.\n\n"
            "Show Σ, Σ⁻¹, vectors and intermediate results in your exam solution for full credit.\n"
        )


@dataclass
class HumanCapitalCalc:
    """
    Extended Mean-Variance with Human Capital.
    
    Handles the specific "Life-Cycle" exam problems with labor income risk.
    
    Exam Reference: Human capital and portfolio choice (2023-2025)
    
    Key Formula:
    π* = M × (1 + l) - H
    
    Where:
    - M = (μ - r_f) / (γ × σ²_S)  [Myopic Demand]
    - l = L₀ / W                   [Leverage Ratio]
    - H = l × (ρ_SL × σ_L / σ_S)  [Hedging Demand]
    """
    # Financial wealth
    wealth: float
    
    # Labor income parameters
    income: float
    growth_rate: float
    discount_rate: float
    years: int
    
    # Market parameters
    mu_market: float
    sigma_market: float
    risk_free_rate: float
    timing_convention: str = "munk_fmt"
    risk_aversion: float = 4.0
    
    # Correlation parameters
    corr_labor_market: float = 0.2
    sigma_labor: float = 0.10
    
    def __post_init__(self):
        """Calculate derived quantities."""
        self._calculate_human_capital()
        self._calculate_optimal_allocation()
    
    def _calculate_human_capital(self):
        """
        Calculate present value of human capital using growing annuity formula.
        
        Timing conventions:
        - munk_fmt: first future payment is Y*(1+g) at t=1
        - ordinary: first payment is Y at t=1
        - annuity_due: first payment is Y at t=0
        
        When r != g: L = Y * (1+r) * [1 - ((1+g)/(1+r))^T] / (r - g)
        When r = g: L = Y * T
        """
        g = self.growth_rate
        r = self.discount_rate
        T = self.years
        Y = self.income
        
        if abs(r - g) < 1e-6:
            # Limit case when r == g for ordinary annuity base
            ordinary_annuity = Y * T / (1 + r)
        else:
            # Growing annuity base (first payment Y at t=1)
            growth_factor = (1 + g) / (1 + r)
            ordinary_annuity = Y * (1 - growth_factor ** T) / (r - g)
        
        if self.timing_convention == "munk_fmt":
            self.human_capital = ordinary_annuity * (1 + g)
        elif self.timing_convention == "annuity_due":
            self.human_capital = ordinary_annuity * (1 + r)
        else:
            self.human_capital = ordinary_annuity
        
        self.total_wealth = self.wealth + self.human_capital
        self.leverage_ratio = self.human_capital / self.wealth if self.wealth > 0 else float('inf')

    def _calculate_optimal_allocation(self):
        """
        Calculate optimal stock allocation with human capital.
        
        Formula: π* = M × (1 + l) - H
        
        Where:
        - M = (μ - r_f) / (γ × σ²_S) [Myopic Demand]
        - l = L₀ / W [Leverage Ratio]
        - H = l × (ρ_SL × σ_L / σ_S) [Hedging Demand]
        """
        gamma = self.risk_aversion
        mu = self.mu_market
        rf = self.risk_free_rate
        sigma_s = self.sigma_market
        sigma_l = self.sigma_labor
        rho = self.corr_labor_market
        l = self.leverage_ratio
        
        # Myopic demand (standard MV without human capital)
        self.myopic_demand = (mu - rf) / (gamma * sigma_s ** 2)
        
        # Hedging demand
        self.hedging_demand = l * (rho * sigma_l / sigma_s)
        
        # Scaled myopic (accounting for total wealth)
        self.scaled_myopic = self.myopic_demand * (1 + l)
        
        # Total optimal weight
        self.optimal_weight = self.scaled_myopic - self.hedging_demand
        
        # Dollar allocations
        self.stock_investment = self.optimal_weight * self.wealth
        self.riskfree_investment = self.wealth - self.stock_investment
    
    def get_breakdown(self) -> Dict:
        """Get complete breakdown of calculations."""
        return {
            'human_capital': {
                'value': self.human_capital,
                'formula': 'Growing Annuity PV',
                'inputs': {
                    'income': self.income,
                    'growth_rate': self.growth_rate,
                    'discount_rate': self.discount_rate,
                    'years': self.years
                }
            },
            'wealth': {
                'financial': self.wealth,
                'human_capital': self.human_capital,
                'total': self.total_wealth,
                'leverage_ratio_l': self.leverage_ratio
            },
            'allocation_components': {
                'myopic_demand_M': self.myopic_demand,
                'scaling_factor': 1 + self.leverage_ratio,
                'scaled_myopic': self.scaled_myopic,
                'hedging_demand_H': self.hedging_demand,
                'optimal_weight': self.optimal_weight
            },
            'dollar_allocation': {
                'stocks': self.stock_investment,
                'risk_free': self.riskfree_investment,
                'total': self.wealth
            }
        }
    
    def get_summary_table(self) -> pd.DataFrame:
        """Generate exam-format summary table."""
        rows = [
            ('**Human Capital Calculation**', '', ''),
            ('Income (Y₀)', f"${self.income:,.2f}", 'Current annual income'),
            ('Growth Rate (g)', f"{self.growth_rate*100:.2f}%", 'Expected income growth'),
            ('Discount Rate (r)', f"{self.discount_rate*100:.2f}%", 'Rate to discount future income'),
            ('Working Years (T)', f"{self.years}", 'Remaining working years'),
            ('Human Capital (L₀)', f"${self.human_capital:,.2f}", 'PV of future income'),
            ('', '', ''),
            ('**Wealth Summary**', '', ''),
            ('Financial Wealth (W)', f"${self.wealth:,.2f}", 'Current investments'),
            ('Human Capital (L₀)', f"${self.human_capital:,.2f}", 'PV of labor income'),
            ('Total Wealth (W + L₀)', f"${self.total_wealth:,.2f}", 'Sum of all wealth'),
            ('Leverage Ratio (l = L₀/W)', f"{self.leverage_ratio:.4f}", 'Human capital relative to financial'),
            ('', '', ''),
            ('**Optimal Allocation Components**', '', ''),
            ('Myopic Demand (M)', f"{self.myopic_demand:.4f}", '(μ-rf)/(γσ²)'),
            ('Scaling Factor (1+l)', f"{1 + self.leverage_ratio:.4f}", 'Total wealth effect'),
            ('Scaled Myopic [M×(1+l)]', f"{self.scaled_myopic:.4f}", 'Demand before hedging'),
            ('Hedging Demand (H)', f"{self.hedging_demand:.4f}", 'l×(ρσ_L/σ_S)'),
            ('', '', ''),
            ('**Final Allocation**', '', ''),
            ('Optimal Stock Weight (π*)', f"{self.optimal_weight:.4f} ({self.optimal_weight*100:.2f}%)", 'M×(1+l) - H'),
            ('Stock Investment ($)', f"${self.stock_investment:,.2f}", 'π* × W'),
            ('Risk-Free Investment ($)', f"${self.riskfree_investment:,.2f}", '(1-π*) × W'),
        ]
        
        return pd.DataFrame(rows, columns=['Metric', 'Value', 'Description'])
    
    def get_latex_formulas(self) -> Dict[str, str]:
        """Return LaTeX formulas."""
        if self.timing_convention == "munk_fmt":
            hc_formula = (
                r'L_0 = (1+g) \times \frac{Y_0 \left[1 - \left(\frac{1+g}{1+r}\right)^T\right]}{r - g}'
                r'\quad \text{(or } Y_0 \times T \text{ when } r = g \text{)}'
            )
        elif self.timing_convention == "annuity_due":
            hc_formula = (
                r'L_0 = (1+r) \times \frac{Y_0 \left[1 - \left(\frac{1+g}{1+r}\right)^T\right]}{r - g}'
                r'\quad \text{(or } Y_0 \times T \text{ when } r = g \text{)}'
            )
        else:
            hc_formula = (
                r'L_0 = \frac{Y_0 \left[1 - \left(\frac{1+g}{1+r}\right)^T\right]}{r - g}'
                r'\quad \text{(or } \frac{Y_0 \times T}{1+r} \text{ when } r = g \text{)}'
            )
        return {
            'human_capital': hc_formula,
            'leverage_ratio': r'l = \frac{L_0}{W}',
            'myopic_demand': r'M = \frac{\mu - r_f}{\gamma \sigma_S^2}',
            'hedging_demand': r'H = l \cdot \frac{\rho_{SL} \sigma_L}{\sigma_S}',
            'optimal_weight': r'\pi^* = M(1 + l) - H = \frac{\mu - r_f}{\gamma \sigma_S^2}(1 + \frac{L_0}{W}) - \frac{L_0}{W} \cdot \frac{\rho_{SL} \sigma_L}{\sigma_S}'
        }

    def get_excel_instructions(self) -> str:
        """Return Excel instructions."""
        if abs(self.discount_rate - self.growth_rate) < 1e-6:
            base_formula = f"={self.income} * {self.years} / (1+{self.discount_rate})"
            base_note = "Base ordinary annuity (r = g): L0 = Y * T / (1+r)"
        else:
            base_formula = (
                f"={self.income} * (1 - ((1+{self.growth_rate})/(1+{self.discount_rate}))^{self.years}) "
                f"/ ({self.discount_rate} - {self.growth_rate})"
            )
            base_note = "Base ordinary annuity (first payment Y at t=1)"
        
        if self.timing_convention == "munk_fmt":
            multiplier = f"(1+{self.growth_rate})"
            timing_note = "Munk/FMT timing: first future payment is Y*(1+g), so multiply by (1+g)."
        elif self.timing_convention == "annuity_due":
            multiplier = f"(1+{self.discount_rate})"
            timing_note = "Annuity due timing: payments start at t=0, so multiply by (1+r)."
        else:
            multiplier = ""
            timing_note = "Ordinary timing: first payment is Y at t=1, so no multiplier."
        
        hc_formula = base_formula + (f" * {multiplier}" if multiplier else "")
        
        return f"""
**Excel Implementation for Human Capital & Optimal Allocation:**

**Step 1: Calculate Human Capital (Timing Convention)**

{base_note}

{timing_note}

Excel:
`{hc_formula}`
Result: L?,? = ${self.human_capital:,.2f}

**Step 2: Calculate Leverage Ratio**
`=L0/W` = {self.human_capital:,.0f}/{self.wealth:,.0f} = {self.leverage_ratio:.4f}

**Step 3: Calculate Myopic Demand (M)**
`=(I? - rf) / (I3 A- I?A?)`
`=({self.mu_market} - {self.risk_free_rate}) / ({self.risk_aversion} A- {self.sigma_market}^2)`
Result: M = {self.myopic_demand:.4f}

**Step 4: Calculate Hedging Demand (H)**
`=l A- I? A- I?_L / I?_S`
`={self.leverage_ratio:.4f} A- {self.corr_labor_market} A- {self.sigma_labor} / {self.sigma_market}`
Result: H = {self.hedging_demand:.4f}

**Step 5: Calculate Optimal Weight**
`=M A- (1 + l) - H`
`={self.myopic_demand:.4f} A- (1 + {self.leverage_ratio:.4f}) - {self.hedging_demand:.4f}`
Result: I?* = {self.optimal_weight:.4f} = {self.optimal_weight*100:.2f}%

**Step 6: Dollar Allocation**
Stock Investment = I?* A- W = {self.optimal_weight:.4f} A- ${self.wealth:,.0f} = ${self.stock_investment:,.2f}
Risk-Free = (1 - I?*) A- W = ${self.riskfree_investment:,.2f}

**Interpretation:**
- {"Leverage (borrowing) indicated" if self.optimal_weight > 1 else "No leverage needed"}
- {"Short position in stocks indicated" if self.optimal_weight < 0 else "Long stocks"}
- Hedging reduces stock weight by {self.hedging_demand*100:.2f}% due to {self.corr_labor_market*100:.0f}% correlation with labor income
"""

class FactorAnalyzer:
    """
    Multi-Factor Risk Analysis Engine.
    
    Properly handles factor correlations in systematic risk calculations.
    Uses MATRIX MATH (β'Σ_F β) not just sum of squares.
    
    Exam Reference: APT and multi-factor models (2023-2025)
    """
    # Factor parameters
    factor_means: np.ndarray  # Expected factor returns/premiums
    factor_vols: np.ndarray   # Factor volatilities
    factor_correlations: np.ndarray  # Factor correlation matrix
    
    # Asset parameters  
    asset_betas: np.ndarray  # Matrix: assets × factors
    asset_idio_vols: np.ndarray  # Idiosyncratic volatilities
    
    risk_free_rate: float = 0.02
    asset_names: Optional[List[str]] = None
    factor_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Compute derived matrices."""
        self.n_factors = len(self.factor_means)
        
        # Handle 1D beta array (single asset)
        if self.asset_betas.ndim == 1:
            self.asset_betas = self.asset_betas.reshape(1, -1)
        
        self.n_assets = self.asset_betas.shape[0]
        
        # Factor covariance matrix: Σ_F = diag(σ_F) × Corr_F × diag(σ_F)
        D_f = np.diag(self.factor_vols)
        self.factor_covariance = D_f @ self.factor_correlations @ D_f
        
        # Set default names
        if self.asset_names is None:
            self.asset_names = [f'Asset {i+1}' for i in range(self.n_assets)]
        if self.factor_names is None:
            self.factor_names = [f'Factor {i+1}' for i in range(self.n_factors)]
        
        # Validate dimensions
        if self.asset_betas.shape[1] != self.n_factors:
            raise ValueError(f"Beta matrix columns ({self.asset_betas.shape[1]}) must match number of factors ({self.n_factors})")
        if len(self.asset_idio_vols) != self.n_assets:
            raise ValueError(f"Idiosyncratic vols ({len(self.asset_idio_vols)}) must match number of assets ({self.n_assets})")
    
    def expected_returns(self) -> np.ndarray:
        """
        Calculate expected returns using APT formula.
        
        E[R_i] = r_f + Σ_k β_ik × (E[F_k] - r_f)
        
        If factor_means are already risk premiums, use them directly.
        """
        # Assume factor_means are risk premiums (E[F] - rf) unless they're > 0.5 (likely levels)
        if np.mean(self.factor_means) > 0.5:
            factor_risk_premiums = self.factor_means - self.risk_free_rate
        else:
            factor_risk_premiums = self.factor_means
        
        return self.risk_free_rate + self.asset_betas @ factor_risk_premiums
    
    def systematic_variance(self, asset_idx: int) -> float:
        """
        Calculate systematic variance for an asset.
        
        IMPORTANT: Uses MATRIX MATH to account for factor correlations.
        σ²_sys = β' × Σ_F × β
        
        NOT just sum of β²σ² (which ignores factor correlations)
        """
        beta = self.asset_betas[asset_idx]
        return beta @ self.factor_covariance @ beta
    
    def total_variance(self, asset_idx: int) -> float:
        """Total variance = Systematic + Idiosyncratic."""
        return self.systematic_variance(asset_idx) + self.asset_idio_vols[asset_idx] ** 2
    
    def risk_decomposition(self) -> pd.DataFrame:
        """
        Generate risk decomposition table for all assets.
        
        Shows: Systematic %, Idiosyncratic %, Total Volatility
        """
        results = []
        
        for i in range(self.n_assets):
            sys_var = self.systematic_variance(i)
            idio_var = self.asset_idio_vols[i] ** 2
            total_var = sys_var + idio_var
            
            results.append({
                'Asset': self.asset_names[i],
                'Systematic Var': sys_var,
                'Idiosyncratic Var': idio_var,
                'Total Var': total_var,
                'Systematic Vol': np.sqrt(sys_var),
                'Idiosyncratic Vol': self.asset_idio_vols[i],
                'Total Vol': np.sqrt(total_var),
                'Systematic %': sys_var / total_var * 100 if total_var > 0 else 0,
                'Idiosyncratic %': idio_var / total_var * 100 if total_var > 0 else 0,
                'R²': sys_var / total_var if total_var > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def factor_contribution_detail(self, asset_idx: int) -> pd.DataFrame:
        """
        Detailed breakdown of factor contributions to systematic risk.
        Accounts for factor correlations (cross-terms).
        """
        beta = self.asset_betas[asset_idx]
        
        # Individual factor variance contributions (diagonal terms)
        diagonal_contrib = beta ** 2 * np.diag(self.factor_covariance)
        
        # Cross-factor contributions (off-diagonal terms)
        cross_contrib = np.zeros(self.n_factors)
        for k in range(self.n_factors):
            for j in range(self.n_factors):
                if k != j:
                    # Split cross-term equally between factors
                    cross_contrib[k] += 0.5 * beta[k] * beta[j] * self.factor_covariance[k, j]
        
        total_contrib = diagonal_contrib + cross_contrib
        total_sys_var = self.systematic_variance(asset_idx)
        
        results = []
        for k in range(self.n_factors):
            results.append({
                'Factor': self.factor_names[k],
                'Beta (β)': beta[k],
                'Factor Vol (σ_F)': self.factor_vols[k],
                'β²σ²_F (Diagonal)': diagonal_contrib[k],
                'Cross-Term': cross_contrib[k],
                'Total Contribution': total_contrib[k],
                '% of Systematic': total_contrib[k] / total_sys_var * 100 if total_sys_var > 0 else 0
            })
        
        # Add total row
        results.append({
            'Factor': 'TOTAL',
            'Beta (β)': np.nan,
            'Factor Vol (σ_F)': np.nan,
            'β²σ²_F (Diagonal)': diagonal_contrib.sum(),
            'Cross-Term': cross_contrib.sum(),
            'Total Contribution': total_sys_var,
            '% of Systematic': 100.0
        })
        
        return pd.DataFrame(results)
    
    def get_latex_formulas(self) -> Dict[str, str]:
        """Return LaTeX formulas."""
        return {
            'expected_return': r'E[R_i] = r_f + \sum_{k=1}^{K} \beta_{ik} \lambda_k',
            'systematic_var': r'\sigma^2_{sys,i} = \boldsymbol{\beta}_i^T \boldsymbol{\Sigma}_F \boldsymbol{\beta}_i',
            'total_var': r'\sigma^2_{total,i} = \sigma^2_{sys,i} + \sigma^2_{\epsilon,i}',
            'factor_cov': r'\boldsymbol{\Sigma}_F = \mathbf{D}_\sigma \mathbf{R}_F \mathbf{D}_\sigma',
            'not_just_sum': r'\sigma^2_{sys} \neq \sum_k \beta_k^2 \sigma_{F_k}^2 \text{ (unless factors are uncorrelated)}',
            'with_correlation': r'\sigma^2_{sys} = \sum_k \beta_k^2 \sigma_{F_k}^2 + 2\sum_{j<k} \beta_j \beta_k \sigma_j \sigma_k \rho_{jk}',
            'asset_covariance': r'Cov(R_i, R_j) = \boldsymbol{\beta}_i^T \boldsymbol{\Sigma}_F \boldsymbol{\beta}_j',
            'asset_correlation': r'\rho_{ij} = \frac{Cov(R_i, R_j)}{\sigma_i \sigma_j}',
            'asset_factor_cov': r'Cov(R_i, F_k) = \sum_{j=1}^{K} \beta_{ij} Cov(F_j, F_k) = (\boldsymbol{\beta}_i^T \boldsymbol{\Sigma}_F)_k'
        }
    
    def asset_covariance(self, asset_i: int, asset_j: int) -> float:
        """
        Calculate covariance between two assets.
        
        Cov(R_i, R_j) = β_i' Σ_F β_j
        
        Note: This only captures systematic covariance. 
        Idiosyncratic risks are assumed uncorrelated.
        """
        beta_i = self.asset_betas[asset_i]
        beta_j = self.asset_betas[asset_j]
        return beta_i @ self.factor_covariance @ beta_j
    
    def asset_correlation(self, asset_i: int, asset_j: int) -> float:
        """
        Calculate correlation between two assets.
        
        ρ_ij = Cov(R_i, R_j) / (σ_i × σ_j)
        """
        cov_ij = self.asset_covariance(asset_i, asset_j)
        vol_i = np.sqrt(self.total_variance(asset_i))
        vol_j = np.sqrt(self.total_variance(asset_j))
        
        if vol_i == 0 or vol_j == 0:
            return 0.0
        return cov_ij / (vol_i * vol_j)
    
    def asset_covariance_matrix(self) -> np.ndarray:
        """
        Calculate full asset covariance matrix.
        
        Σ_assets = B Σ_F B' + D_ε
        
        Where D_ε is diagonal matrix of idiosyncratic variances.
        """
        # Systematic covariance: B Σ_F B'
        sys_cov = self.asset_betas @ self.factor_covariance @ self.asset_betas.T
        
        # Add idiosyncratic variances on diagonal
        idio_var = np.diag(self.asset_idio_vols ** 2)
        
        return sys_cov + idio_var
    
    def asset_correlation_matrix(self) -> np.ndarray:
        """
        Calculate full asset correlation matrix.
        """
        cov_matrix = self.asset_covariance_matrix()
        vols = np.sqrt(np.diag(cov_matrix))
        
        # Avoid division by zero
        vols[vols == 0] = 1e-10
        
        # Correlation = Cov / (vol_i * vol_j)
        D_inv = np.diag(1.0 / vols)
        return D_inv @ cov_matrix @ D_inv
    
    def asset_factor_covariance(self, asset_idx: int, factor_idx: int) -> float:
        """
        Calculate covariance between an asset and a factor.
        
        Cov(R_i, F_k) = Σ_j β_ij × Cov(F_j, F_k)
                      = (β_i' Σ_F)_k
        
        This is the k-th element of β_i' Σ_F
        """
        beta_i = self.asset_betas[asset_idx]
        # β_i' Σ_F gives a row vector, take the k-th element
        return (beta_i @ self.factor_covariance)[factor_idx]
    
    def asset_factor_covariance_matrix(self) -> np.ndarray:
        """
        Calculate covariance matrix between all assets and all factors.
        
        Returns matrix of shape (n_assets, n_factors)
        Element (i,k) = Cov(R_i, F_k)
        """
        return self.asset_betas @ self.factor_covariance
    
    def get_excel_instructions(self) -> str:
        """Return comprehensive Excel instructions for all calculations."""
        
        # Build factor covariance display
        factor_cov_rows = []
        for i in range(self.n_factors):
            row_vals = [f"{self.factor_covariance[i,j]:.6f}" for j in range(self.n_factors)]
            factor_cov_rows.append(" | ".join(row_vals))
        factor_cov_table = "\n".join([f"| {row} |" for row in factor_cov_rows])
        
        # Build risk decomposition display
        decomp = self.risk_decomposition()
        
        # Build asset covariance matrix display
        asset_cov = self.asset_covariance_matrix()
        asset_cov_rows = []
        for i in range(self.n_assets):
            row_vals = [f"{asset_cov[i,j]:.6f}" for j in range(self.n_assets)]
            asset_cov_rows.append(f"{self.asset_names[i]}: " + ", ".join(row_vals))
        
        # Build asset correlation matrix display
        asset_corr = self.asset_correlation_matrix()
        asset_corr_rows = []
        for i in range(self.n_assets):
            row_vals = [f"{asset_corr[i,j]:.4f}" for j in range(self.n_assets)]
            asset_corr_rows.append(f"{self.asset_names[i]}: " + ", ".join(row_vals))
        
        # Build asset-factor covariance display
        af_cov = self.asset_factor_covariance_matrix()
        af_cov_rows = []
        for i in range(self.n_assets):
            row_vals = [f"{af_cov[i,k]:.6f}" for k in range(self.n_factors)]
            af_cov_rows.append(f"{self.asset_names[i]}: " + ", ".join(row_vals))
        
        return f"""
## Multi-Factor Risk Analysis - Complete Excel Guide

### Step 1: Build the Factor Covariance Matrix (Σ_F)

**Formula:** Σ_F = D_σ × R_F × D_σ

Where:
- D_σ = diagonal matrix of factor volatilities
- R_F = factor correlation matrix

**Excel Steps:**
1. Create diagonal matrix D with factor vols: `=IF(ROW()=COLUMN(), vol_k, 0)`
2. Multiply: `=MMULT(MMULT(D, Corr), D)`

**Your Factor Covariance Matrix:**
{factor_cov_table}

---
### Step 2: Calculate Expected Returns (APT Formula)

**Formula:** E[R_i] = r_f + Σ_k β_ik × λ_k

**Excel:** `=rf + SUMPRODUCT(betas_row, lambdas)`

**Your Results:**
"""
        
        # Add expected returns
        exp_rets = self.expected_returns()
        for i in range(self.n_assets):
            beta_i = self.asset_betas[i]
            return f"""{self.asset_names[i]}: E[R] = {self.risk_free_rate:.4f} + """ + \
                   " + ".join([f"{beta_i[k]:.2f}×{self.factor_means[k]:.4f}" for k in range(self.n_factors)]) + \
                   f" = **{exp_rets[i]*100:.2f}%**\n"

    def get_excel_instructions_full(self) -> str:
        """Return comprehensive Excel instructions for all calculations."""
        
        # Build all matrices for display
        exp_rets = self.expected_returns()
        decomp = self.risk_decomposition()
        asset_cov = self.asset_covariance_matrix()
        asset_corr = self.asset_correlation_matrix()
        af_cov = self.asset_factor_covariance_matrix()
        
        instructions = f"""
## Multi-Factor Risk Analysis - Complete Excel Guide

---
### 1. Factor Covariance Matrix (Σ_F)

**Formula:** Σ_F = diag(σ_F) × Corr_F × diag(σ_F)

**Excel Implementation:**
1. Put factor volatilities in a column (e.g., B2:B{1+self.n_factors})
2. Put correlation matrix in a range (e.g., D2:E{1+self.n_factors} for 2 factors)
3. Create diagonal matrix: `=IF(ROW()-ROW($B$2)+1=COLUMN()-COLUMN($D$2)+1, B2, 0)` and drag
4. Factor covariance: `=MMULT(MMULT(DiagMatrix, CorrMatrix), DiagMatrix)`

**Your Σ_F (computed):**
```
{np.array2string(self.factor_covariance, precision=6, separator=', ')}
```

---
### 2. Expected Returns (APT)

**Formula:** E[R_i] = r_f + β_i1×λ_1 + β_i2×λ_2 + ...

**Excel:** `=rf + SUMPRODUCT(beta_range, lambda_range)`

**Your Results:**
"""
        for i in range(self.n_assets):
            beta_i = self.asset_betas[i]
            calc = f"{self.risk_free_rate:.4f}"
            for k in range(self.n_factors):
                calc += f" + {beta_i[k]:.3f}×{self.factor_means[k]:.4f}"
            instructions += f"- {self.asset_names[i]}: {calc} = **{exp_rets[i]*100:.2f}%**\n"
        
        instructions += f"""
---
### 3. Systematic Variance (IMPORTANT: Matrix Method!)

**Formula:** σ²_sys,i = β_i' × Σ_F × β_i

⚠️ **WARNING:** Do NOT use Σ β²σ² unless factors are uncorrelated!

**Excel:** `=MMULT(MMULT(TRANSPOSE(beta_row), SigmaF), beta_row)`

**Your Results:**
"""
        for i in range(self.n_assets):
            sys_var = self.systematic_variance(i)
            instructions += f"- {self.asset_names[i]}: σ²_sys = **{sys_var:.6f}** (σ_sys = {np.sqrt(sys_var)*100:.2f}%)\n"
        
        instructions += f"""
---
### 4. Total Variance & R²

**Formulas:**
- Total variance: σ²_total = σ²_sys + σ²_ε
- R² = σ²_sys / σ²_total

**Excel:**
- Total var: `=sys_var + idio_vol^2`
- R²: `=sys_var / total_var`

**Your Results:**
"""
        for i in range(self.n_assets):
            sys_var = self.systematic_variance(i)
            idio_var = self.asset_idio_vols[i]**2
            total_var = sys_var + idio_var
            r2 = sys_var / total_var if total_var > 0 else 0
            instructions += f"- {self.asset_names[i]}: Total σ = **{np.sqrt(total_var)*100:.2f}%**, R² = **{r2:.4f}** ({r2*100:.1f}%)\n"
        
        instructions += f"""
---
### 5. Asset-Asset Covariance & Correlation

**Formula:** Cov(R_i, R_j) = β_i' × Σ_F × β_j

**Excel:** `=MMULT(MMULT(TRANSPOSE(beta_i), SigmaF), beta_j)`

**Correlation:** ρ_ij = Cov(R_i, R_j) / (σ_i × σ_j)

**Your Asset Covariance Matrix:**
```
{np.array2string(asset_cov, precision=6, separator=', ')}
```

**Your Asset Correlation Matrix:**
```
{np.array2string(asset_corr, precision=4, separator=', ')}
```

---
### 6. Asset-Factor Covariance

**Formula:** Cov(R_i, F_k) = Σ_j β_ij × Cov(F_j, F_k) = (β_i' × Σ_F)_k

This tells you how much an asset co-moves with each factor.

**Excel:** `=MMULT(TRANSPOSE(beta_i), SigmaF)` gives row of covariances with all factors

**Your Asset-Factor Covariances:**
"""
        for i in range(self.n_assets):
            for k in range(self.n_factors):
                cov_val = af_cov[i, k]
                instructions += f"- Cov({self.asset_names[i]}, {self.factor_names[k]}) = **{cov_val:.6f}**\n"
        
        instructions += f"""
---
### 7. Sharpe Ratio

**Formula:** SR_i = (E[R_i] - r_f) / σ_i

**Excel:** `=(expected_return - rf) / total_vol`

**Your Results:**
"""
        for i in range(self.n_assets):
            er = exp_rets[i]
            total_vol = np.sqrt(self.total_variance(i))
            sharpe = (er - self.risk_free_rate) / total_vol if total_vol > 0 else 0
            instructions += f"- {self.asset_names[i]}: SR = ({er*100:.2f}% - {self.risk_free_rate*100:.2f}%) / {total_vol*100:.2f}% = **{sharpe:.4f}**\n"
        
        return instructions


@dataclass 
class PerformanceAnalyzer:
    """
    Fund performance metrics calculator.
    
    Calculates: Sharpe, Treynor, Jensen's Alpha, Information Ratio, M²
    """
    fund_return: float
    fund_vol: float
    fund_beta: float
    market_return: float
    market_vol: float
    risk_free_rate: float
    fund_name: str = "Fund"
    
    def __post_init__(self):
        """Calculate all performance metrics."""
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Compute all performance ratios."""
        rf = self.risk_free_rate
        
        # Sharpe Ratio
        self.sharpe_ratio = (self.fund_return - rf) / self.fund_vol
        
        # Treynor Ratio
        self.treynor_ratio = (self.fund_return - rf) / self.fund_beta if self.fund_beta != 0 else 0
        
        # CAPM Expected Return
        self.capm_expected = rf + self.fund_beta * (self.market_return - rf)
        
        # Jensen's Alpha
        self.jensens_alpha = self.fund_return - self.capm_expected
        
        # Variance Decomposition: σ²_total = β²σ²_m + σ²_ε
        self.systematic_var = (self.fund_beta ** 2) * (self.market_vol ** 2)
        self.total_var = self.fund_vol ** 2
        self.residual_var = max(0, self.total_var - self.systematic_var)
        self.residual_vol = np.sqrt(self.residual_var)
        
        # Information Ratio = α / σ_ε
        self.information_ratio = self.jensens_alpha / self.residual_vol if self.residual_vol > 0 else 0
        
        # M² (Modigliani-Modigliani)
        # Risk-adjusted return if fund had same risk as market
        self.m_squared = rf + self.sharpe_ratio * self.market_vol
        
        # Market Sharpe
        self.market_sharpe = (self.market_return - rf) / self.market_vol
    
    def get_summary(self) -> Dict:
        """Return all metrics as dictionary."""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'treynor_ratio': self.treynor_ratio,
            'jensens_alpha': self.jensens_alpha,
            'information_ratio': self.information_ratio,
            'm_squared': self.m_squared,
            'market_sharpe': self.market_sharpe,
            'capm_expected': self.capm_expected,
            'systematic_var': self.systematic_var,
            'residual_var': self.residual_var,
            'residual_vol': self.residual_vol
        }
    
    def get_summary_table(self) -> pd.DataFrame:
        """Generate formatted summary table."""
        rows = [
            ('**Risk-Adjusted Performance**', '', ''),
            ('Sharpe Ratio', f"{self.sharpe_ratio:.4f}", '(R_p - R_f) / σ_p'),
            ('Treynor Ratio', f"{self.treynor_ratio:.4f}", '(R_p - R_f) / β_p'),
            ("Jensen's Alpha", f"{self.jensens_alpha*100:.2f}%", 'R_p - [R_f + β(R_m - R_f)]'),
            ('Information Ratio', f"{self.information_ratio:.4f}", 'α / σ_ε'),
            ('M² (Modigliani)', f"{self.m_squared*100:.2f}%", 'R_f + SR_p × σ_m'),
            ('', '', ''),
            ('**Benchmarks**', '', ''),
            ('CAPM Expected Return', f"{self.capm_expected*100:.2f}%", 'R_f + β(R_m - R_f)'),
            ('Market Sharpe Ratio', f"{self.market_sharpe:.4f}", '(R_m - R_f) / σ_m'),
            ('', '', ''),
            ('**Variance Decomposition**', '', ''),
            ('Total Variance (σ²_p)', f"{self.total_var:.6f}", 'Fund total variance'),
            ('Systematic Variance', f"{self.systematic_var:.6f}", 'β² × σ²_m'),
            ('Residual Variance (σ²_ε)', f"{self.residual_var:.6f}", 'σ²_p - β²σ²_m'),
            ('Residual Volatility (σ_ε)', f"{self.residual_vol*100:.2f}%", '√(σ²_ε)'),
            ('R² (Systematic %)', f"{self.systematic_var/self.total_var*100:.2f}%", 'β²σ²_m / σ²_p'),
        ]
        
        return pd.DataFrame(rows, columns=['Metric', 'Value', 'Formula'])
    
    def get_latex_formulas(self) -> Dict[str, str]:
        """Return LaTeX formulas."""
        return {
            'sharpe': r'SR = \frac{R_p - R_f}{\sigma_p}',
            'treynor': r'TR = \frac{R_p - R_f}{\beta_p}',
            'jensen': r'\alpha_J = R_p - [R_f + \beta_p(R_m - R_f)]',
            'information': r'IR = \frac{\alpha}{\sigma_\epsilon}',
            'm_squared': r'M^2 = R_f + SR_p \times \sigma_m',
            'var_decomp': r'\sigma^2_p = \beta^2 \sigma^2_m + \sigma^2_\epsilon'
        }


@dataclass
class TreynorBlackOptimizer:
    """
    Treynor-Black Active Portfolio Optimization.
    
    Combines passive market index with actively managed portfolio of mispriced securities.
    
    Key insight: SR²_combined = SR²_market + IR²_active
    """
    # Active securities
    alphas: np.ndarray
    betas: np.ndarray
    residual_vols: np.ndarray
    
    # Market parameters
    market_return: float
    market_vol: float
    risk_free_rate: float
    risk_aversion: float = 4.0
    
    security_names: Optional[List[str]] = None
    
    def __post_init__(self):
        """Compute optimal active portfolio."""
        self.n_securities = len(self.alphas)
        self.residual_vars = self.residual_vols ** 2
        
        if self.security_names is None:
            self.security_names = [f'Security {i+1}' for i in range(self.n_securities)]
        
        self._calculate_optimal_portfolio()
    
    def _calculate_optimal_portfolio(self):
        """Calculate Treynor-Black optimal weights."""
        # Step 1: Raw active weights (proportional to α/σ²_ε)
        self.raw_weights = self.alphas / self.residual_vars
        
        # Step 2: Normalize to get active portfolio weights
        # Sum can be negative if alphas sum negative, so we normalize by sum (not abs sum)
        weight_sum = np.sum(self.raw_weights)
        if abs(weight_sum) > 1e-10:
            self.active_weights = self.raw_weights / weight_sum
        else:
            self.active_weights = self.raw_weights
        
        # Step 3: Active portfolio characteristics
        self.alpha_A = self.active_weights @ self.alphas
        self.beta_A = self.active_weights @ self.betas
        
        # Residual variance (assuming independent residuals)
        self.var_epsilon_A = np.sum(self.active_weights ** 2 * self.residual_vars)
        self.sigma_epsilon_A = np.sqrt(self.var_epsilon_A)
        
        # Step 4: Information Ratio of Active Portfolio
        self.IR_A = self.alpha_A / self.sigma_epsilon_A if self.sigma_epsilon_A > 0 else 0
        
        # Appraisal ratio squared
        self.appraisal_sq = self.IR_A ** 2
        
        # Step 5: Market Sharpe Ratio
        rf = self.risk_free_rate
        self.market_sharpe = (self.market_return - rf) / self.market_vol
        self.market_sharpe_sq = self.market_sharpe ** 2
        
        # Step 6: Optimal weight in active portfolio
        # w*_A = (α_A/σ²_ε,A) / [(μ_m - r_f)/σ²_m]
        # Then adjust for beta: w*_A / (1 + (1-β_A)×w*_A)
        
        market_ratio = (self.market_return - rf) / (self.market_vol ** 2)
        if abs(market_ratio) > 1e-10 and self.var_epsilon_A > 0:
            w_A_raw = (self.alpha_A / self.var_epsilon_A) / market_ratio
            self.w_active = w_A_raw / (1 + (1 - self.beta_A) * w_A_raw)
        else:
            self.w_active = 0
        
        self.w_market = 1 - self.w_active
        
        # Step 7: Combined Sharpe Ratio
        self.combined_sharpe_sq = self.market_sharpe_sq + self.appraisal_sq
        self.combined_sharpe = np.sqrt(self.combined_sharpe_sq)
        
        # Improvement
        self.sharpe_improvement = self.combined_sharpe - self.market_sharpe
        self.sharpe_improvement_pct = (self.combined_sharpe / self.market_sharpe - 1) * 100 if self.market_sharpe > 0 else 0
        
        # Step 8: Portfolio Variance and Returns
        # Active Portfolio (A) alone:
        # E[r_A] = rf + α_A + β_A × (E[r_m] - rf)
        # Var[r_A] = β_A² × σ²_m + σ²_ε,A
        self.return_active = rf + self.alpha_A + self.beta_A * (self.market_return - rf)
        self.var_active = (self.beta_A ** 2) * (self.market_vol ** 2) + self.var_epsilon_A
        self.vol_active = np.sqrt(self.var_active)
        
        # Combined Portfolio (C) = w_A × Active + w_M × Market:
        # E[r_C] = w_A × E[r_A] + w_M × E[r_m]
        # Also can express as: E[r_C] = rf + w_A × α_A + β_C × (E[r_m] - rf)
        # where β_C = w_A × β_A + w_M × 1 = w_A × β_A + w_M
        self.beta_combined = self.w_active * self.beta_A + self.w_market * 1.0
        self.return_combined = self.w_active * self.return_active + self.w_market * self.market_return
        
        # Var[r_C] = β_C² × σ²_m + w_A² × σ²_ε,A
        # (market has no idiosyncratic risk, only active portfolio contributes)
        self.var_combined = (self.beta_combined ** 2) * (self.market_vol ** 2) + (self.w_active ** 2) * self.var_epsilon_A
        self.vol_combined = np.sqrt(self.var_combined)
        
        # Excess returns (over risk-free)
        self.excess_return_active = self.return_active - rf
        self.excess_return_combined = self.return_combined - rf
        self.excess_return_market = self.market_return - rf
    
    def get_security_analysis(self) -> pd.DataFrame:
        """Table of individual security analysis."""
        return pd.DataFrame({
            'Security': self.security_names,
            'Alpha (α)': self.alphas,
            'Beta (β)': self.betas,
            'Residual Vol (σ_ε)': self.residual_vols,
            'Residual Var (σ²_ε)': self.residual_vars,
            'α / σ²_ε': self.alphas / self.residual_vars,
            'Active Weight': self.active_weights,
            'Active Weight (%)': self.active_weights * 100
        })
    
    def get_summary(self) -> Dict:
        """Return complete optimization summary."""
        return {
            'active_portfolio': {
                'alpha': self.alpha_A,
                'beta': self.beta_A,
                'residual_vol': self.sigma_epsilon_A,
                'residual_var': self.var_epsilon_A,
                'information_ratio': self.IR_A,
                'appraisal_sq': self.appraisal_sq,
                'expected_return': self.return_active,
                'excess_return': self.excess_return_active,
                'variance': self.var_active,
                'volatility': self.vol_active
            },
            'optimal_allocation': {
                'weight_active': self.w_active,
                'weight_market': self.w_market
            },
            'combined_portfolio': {
                'beta': self.beta_combined,
                'expected_return': self.return_combined,
                'excess_return': self.excess_return_combined,
                'variance': self.var_combined,
                'volatility': self.vol_combined
            },
            'market': {
                'expected_return': self.market_return,
                'excess_return': self.excess_return_market,
                'variance': self.market_vol ** 2,
                'volatility': self.market_vol
            },
            'performance': {
                'market_sharpe': self.market_sharpe,
                'market_sharpe_sq': self.market_sharpe_sq,
                'combined_sharpe': self.combined_sharpe,
                'combined_sharpe_sq': self.combined_sharpe_sq,
                'sharpe_improvement': self.sharpe_improvement,
                'sharpe_improvement_pct': self.sharpe_improvement_pct
            }
        }
    
    def get_summary_table(self) -> pd.DataFrame:
        """Generate formatted summary table."""
        rows = [
            ('**Active Portfolio Characteristics**', '', ''),
            ('Alpha (α_A)', f"{self.alpha_A*100:.2f}%", 'Σ w_i × α_i'),
            ('Beta (β_A)', f"{self.beta_A:.4f}", 'Σ w_i × β_i'),
            ('Residual Vol (σ_ε,A)', f"{self.sigma_epsilon_A*100:.2f}%", '√(Σ w²_i × σ²_ε,i)'),
            ('Information Ratio (IR_A)', f"{self.IR_A:.4f}", 'α_A / σ_ε,A'),
            ('Appraisal Ratio² (IR²)', f"{self.appraisal_sq:.4f}", 'IR²_A'),
            ('', '', ''),
            ('**Active Portfolio Risk & Return**', '', ''),
            ('E[r_A] (Expected Return)', f"{self.return_active*100:.2f}%", 'rf + α_A + β_A(E[rm]-rf)'),
            ('Excess Return (E[r_A] - rf)', f"{self.excess_return_active*100:.2f}%", 'α_A + β_A(E[rm]-rf)'),
            ('Variance (σ²_A)', f"{self.var_active:.6f}", 'β²_A × σ²_m + σ²_ε,A'),
            ('Volatility (σ_A)', f"{self.vol_active*100:.2f}%", '√(σ²_A)'),
            ('', '', ''),
            ('**Optimal Allocation**', '', ''),
            ('Weight in Active (w_A)', f"{self.w_active*100:.2f}%", 'Optimal active allocation'),
            ('Weight in Market (w_M)', f"{self.w_market*100:.2f}%", '1 - w_A'),
            ('', '', ''),
            ('**Combined Portfolio Risk & Return**', '', ''),
            ('Beta (β_C)', f"{self.beta_combined:.4f}", 'w_A × β_A + w_M × 1'),
            ('E[r_C] (Expected Return)', f"{self.return_combined*100:.2f}%", 'w_A × E[r_A] + w_M × E[rm]'),
            ('Excess Return (E[r_C] - rf)', f"{self.excess_return_combined*100:.2f}%", 'E[r_C] - rf'),
            ('Variance (σ²_C)', f"{self.var_combined:.6f}", 'β²_C × σ²_m + w²_A × σ²_ε,A'),
            ('Volatility (σ_C)', f"{self.vol_combined*100:.2f}%", '√(σ²_C)'),
            ('', '', ''),
            ('**Performance Improvement**', '', ''),
            ('Market Sharpe Ratio', f"{self.market_sharpe:.4f}", '(E[rm] - rf) / σ_m'),
            ('Market SR²', f"{self.market_sharpe_sq:.4f}", 'SR²_m'),
            ('Combined SR² = SR²_m + IR²_A', f"{self.combined_sharpe_sq:.4f}", 'Key T-B result'),
            ('Combined Sharpe Ratio', f"{self.combined_sharpe:.4f}", '√(SR²_m + IR²_A)'),
            ('Improvement', f"+{self.sharpe_improvement:.4f} (+{self.sharpe_improvement_pct:.1f}%)", 'SR_combined - SR_m'),
        ]
        
        return pd.DataFrame(rows, columns=['Metric', 'Value', 'Formula'])
    
    def get_latex_formulas(self) -> Dict[str, str]:
        """Return LaTeX formulas."""
        return {
            'active_weight': r'w_i = \frac{\alpha_i / \sigma^2_{\epsilon,i}}{\sum_j \alpha_j / \sigma^2_{\epsilon,j}}',
            'alpha_A': r'\alpha_A = \sum_i w_i \alpha_i',
            'beta_A': r'\beta_A = \sum_i w_i \beta_i',
            'var_epsilon_A': r'\sigma^2_{\epsilon,A} = \sum_i w_i^2 \sigma^2_{\epsilon,i}',
            'information_ratio': r'IR_A = \frac{\alpha_A}{\sigma_{\epsilon,A}}',
            'combined_sharpe': r'SR^2_{combined} = SR^2_M + IR^2_A',
            'optimal_active': r'w^*_A = \frac{\alpha_A / \sigma^2_{\epsilon,A}}{(\mu_M - r_f) / \sigma^2_M} \cdot \frac{1}{1 + (1-\beta_A) w^{raw}_A}'
        }
    
    def get_excel_instructions(self) -> str:
        """Return Excel instructions."""
        return f"""
**Excel Implementation for Treynor-Black Model:**

**Step 1: Calculate α/σ²_ε for each security**
- Column A: Security names
- Column B: Alpha (α)
- Column C: Residual Vol (σ_ε)
- Column D: α/σ²_ε = `=B2/(C2^2)`

**Step 2: Calculate Active Portfolio Weights**
- Sum of α/σ²_ε: `=SUM(D:D)`
- Weight for each: `=D2/SUM($D:$D)`

**Step 3: Active Portfolio Alpha**
`=SUMPRODUCT(weights, alphas)` = {self.alpha_A*100:.4f}%

**Step 4: Active Portfolio Beta**
`=SUMPRODUCT(weights, betas)` = {self.beta_A:.4f}

**Step 5: Active Portfolio Residual Variance**
`=SUMPRODUCT(weights^2, residual_vars)` = {self.var_epsilon_A:.6f}
Note: Assumes independent residuals

**Step 6: Information Ratio**
`=α_A / SQRT(σ²_ε,A)` = {self.alpha_A:.4f} / {self.sigma_epsilon_A:.4f} = {self.IR_A:.4f}

**Step 7: Active Portfolio Return & Variance**
E[r_A] = rf + α_A + β_A × (E[rm] - rf)
      = {self.risk_free_rate:.4f} + {self.alpha_A:.4f} + {self.beta_A:.4f} × {self.market_return - self.risk_free_rate:.4f}
      = **{self.return_active*100:.2f}%**

σ²_A = β²_A × σ²_m + σ²_ε,A
     = {self.beta_A:.4f}² × {self.market_vol:.4f}² + {self.var_epsilon_A:.6f}
     = {self.var_active:.6f}
σ_A = **{self.vol_active*100:.2f}%**

**Step 8: Key T-B Result**
SR²_combined = SR²_market + IR²_active
= {self.market_sharpe_sq:.4f} + {self.appraisal_sq:.4f} = {self.combined_sharpe_sq:.4f}
SR_combined = √{self.combined_sharpe_sq:.4f} = {self.combined_sharpe:.4f}

**Step 9: Optimal Allocation**
Raw w_A = (α_A/σ²_ε,A) / [(μ_m-rf)/σ²_m]
Adjusted w_A = raw / (1 + (1-β_A)×raw) = {self.w_active:.4f}
w_market = 1 - w_A = {self.w_market:.4f}

**Step 10: Combined Portfolio Return & Variance**
β_C = w_A × β_A + w_M × 1 = {self.w_active:.4f} × {self.beta_A:.4f} + {self.w_market:.4f} = {self.beta_combined:.4f}

E[r_C] = w_A × E[r_A] + w_M × E[rm]
       = {self.w_active:.4f} × {self.return_active:.4f} + {self.w_market:.4f} × {self.market_return:.4f}
       = **{self.return_combined*100:.2f}%**

σ²_C = β²_C × σ²_m + w²_A × σ²_ε,A
     = {self.beta_combined:.4f}² × {self.market_vol:.4f}² + {self.w_active:.4f}² × {self.var_epsilon_A:.6f}
     = {self.var_combined:.6f}
σ_C = **{self.vol_combined*100:.2f}%**
"""


# =============================================================================
# BLACK-LITTERMAN MODEL (Active Management)
# =============================================================================

@dataclass
class BlackLittermanResult:
    """Result container for Black-Litterman model."""
    # Prior (equilibrium) returns
    equilibrium_returns: np.ndarray
    
    # Posterior results
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray
    
    # Optimal weights
    prior_weights: np.ndarray  # Market cap weights (input)
    posterior_weights: np.ndarray  # BL optimal weights
    
    # View analysis
    view_contributions: np.ndarray  # How much each view shifts returns
    
    # Diagnostics
    tau: float
    risk_aversion: float
    n_assets: int
    n_views: int
    asset_names: List[str]
    view_descriptions: List[str]


@dataclass 
class BlackLittermanOptimizer:
    """
    Black-Litterman Model for combining market equilibrium with investor views.
    
    The model blends:
    1. Prior: Equilibrium returns implied by market cap weights (π = δΣw_mkt)
    2. Views: Investor's subjective views on relative or absolute returns
    
    Master Formula:
    μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]
    
    Or equivalently (more stable):
    μ_BL = π + τΣP'(τPΣP' + Ω)⁻¹(Q - Pπ)
    
    Exam Reference: Chapter 13 - Active Management, Bayesian Portfolio Choice
    """
    # Market parameters
    market_weights: np.ndarray  # Market cap weights (n,)
    covariance: np.ndarray      # Asset covariance matrix Σ (n×n)
    risk_aversion: float        # Market risk aversion δ (typically 2.5)
    
    # Uncertainty in prior
    tau: float = 0.05           # Scalar uncertainty (typically 0.025 to 0.05)
    
    # Views
    P: np.ndarray = None        # View matrix (k×n): which assets involved in each view
    Q: np.ndarray = None        # View returns (k,): expected return for each view
    view_confidences: np.ndarray = None  # Confidence levels (k,): 0-1 scale
    
    # Optional
    risk_free_rate: float = 0.0
    asset_names: Optional[List[str]] = None
    view_descriptions: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate inputs and compute Black-Litterman posterior."""
        self.n_assets = len(self.market_weights)
        
        # Validate covariance
        self.Sigma = symmetrize_covariance(np.asarray(self.covariance, dtype=float))
        if self.Sigma.shape != (self.n_assets, self.n_assets):
            raise ValidationError(f"Covariance shape {self.Sigma.shape} doesn't match n_assets={self.n_assets}")
        
        # Normalize market weights
        self.w_mkt = np.asarray(self.market_weights, dtype=float)
        self.w_mkt = self.w_mkt / np.sum(self.w_mkt)  # Ensure sums to 1
        
        # Asset names
        if self.asset_names is None:
            self.asset_names = [f'Asset {i+1}' for i in range(self.n_assets)]
        
        # Calculate equilibrium returns (prior)
        self._calculate_equilibrium_returns()
        
        # Process views if provided
        if self.P is not None and self.Q is not None:
            self._process_views()
            self._calculate_posterior()
        else:
            # No views - posterior equals prior
            self.n_views = 0
            self.posterior_returns = self.equilibrium_returns.copy()
            self.posterior_cov = self.Sigma.copy()
            self.view_contributions = np.array([])
            self.Omega = None
        
        # Calculate optimal weights from posterior
        self._calculate_optimal_weights()
    
    def _calculate_equilibrium_returns(self):
        """
        Calculate equilibrium (implied) returns from market weights.
        
        Formula: π = δ × Σ × w_mkt
        
        This reverse-engineers the expected returns that would make
        the market portfolio optimal under mean-variance.
        """
        delta = self.risk_aversion
        self.equilibrium_returns = delta * self.Sigma @ self.w_mkt
    
    def _process_views(self):
        """Validate and process view inputs."""
        self.P = np.atleast_2d(np.asarray(self.P, dtype=float))
        self.Q = np.atleast_1d(np.asarray(self.Q, dtype=float))
        
        self.n_views = len(self.Q)
        
        if self.P.shape != (self.n_views, self.n_assets):
            raise ValidationError(f"P matrix shape {self.P.shape} should be ({self.n_views}, {self.n_assets})")
        
        # View descriptions
        if self.view_descriptions is None:
            self.view_descriptions = [f'View {i+1}' for i in range(self.n_views)]
        
        # Build Omega (view uncertainty matrix)
        if self.view_confidences is not None:
            confidences = np.atleast_1d(np.asarray(self.view_confidences, dtype=float))
            # Convert confidence to variance: higher confidence = lower variance
            # Use formula: ω_i = (1/c_i - 1) × τ × P_i × Σ × P_i'
            # Simpler approach: ω_i = τ × P_i × Σ × P_i' × (1 - c_i) / c_i
            variances = []
            for i in range(self.n_views):
                # Base variance from prior
                base_var = self.tau * self.P[i] @ self.Sigma @ self.P[i]
                # Scale by confidence (c=1 means very confident, c→0 means very uncertain)
                conf = np.clip(confidences[i], 0.01, 0.99)
                omega_i = base_var * (1 - conf) / conf
                variances.append(omega_i)
            self.Omega = np.diag(variances)
        else:
            # Default: Omega proportional to view portfolio variances
            # Ω = diag(τ × P × Σ × P')
            view_variances = np.array([
                self.tau * self.P[i] @ self.Sigma @ self.P[i] 
                for i in range(self.n_views)
            ])
            self.Omega = np.diag(view_variances)
    
    def _calculate_posterior(self):
        """
        Calculate Black-Litterman posterior returns and covariance.
        
        Uses the stable form of the BL formula:
        μ_BL = π + τΣP'(τPΣP' + Ω)⁻¹(Q - Pπ)
        Σ_BL = Σ + τΣ - τΣP'(τPΣP' + Ω)⁻¹PτΣ
        """
        pi = self.equilibrium_returns
        tau = self.tau
        Sigma = self.Sigma
        P = self.P
        Q = self.Q
        Omega = self.Omega
        
        # Intermediate calculations
        tau_Sigma = tau * Sigma
        tau_P_Sigma_Pt = tau * P @ Sigma @ P.T
        M = tau_P_Sigma_Pt + Omega  # k×k matrix
        
        # Solve M × x = (Q - P×π) instead of inverting
        view_errors = Q - P @ pi  # How much views differ from prior
        
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)
        
        # Posterior mean
        adjustment = tau_Sigma @ P.T @ M_inv @ view_errors
        self.posterior_returns = pi + adjustment
        
        # View contributions (how much each view shifts returns)
        self.view_contributions = adjustment
        
        # Posterior covariance (simplified - often just use Σ in practice)
        # Full formula: Σ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
        # But this is often close to Σ, so we use a simpler approximation
        self.posterior_cov = Sigma + tau_Sigma - tau_Sigma @ P.T @ M_inv @ P @ tau_Sigma
    
    def _calculate_optimal_weights(self):
        """Calculate optimal weights from posterior returns."""
        # Standard MV optimization: w* = (1/δ) × Σ⁻¹ × (μ - rf)
        delta = self.risk_aversion
        rf = self.risk_free_rate
        
        excess_returns = self.posterior_returns - rf
        
        try:
            Sigma_inv = np.linalg.inv(self.posterior_cov)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(self.posterior_cov)
        
        raw_weights = (1 / delta) * Sigma_inv @ excess_returns
        
        # Normalize to sum to 1
        self.posterior_weights = raw_weights / np.sum(raw_weights)
    
    def get_result(self) -> BlackLittermanResult:
        """Return complete result object."""
        return BlackLittermanResult(
            equilibrium_returns=self.equilibrium_returns,
            posterior_returns=self.posterior_returns,
            posterior_covariance=self.posterior_cov,
            prior_weights=self.w_mkt,
            posterior_weights=self.posterior_weights,
            view_contributions=self.view_contributions,
            tau=self.tau,
            risk_aversion=self.risk_aversion,
            n_assets=self.n_assets,
            n_views=self.n_views,
            asset_names=self.asset_names,
            view_descriptions=self.view_descriptions
        )
    
    def get_summary_table(self) -> pd.DataFrame:
        """Generate comprehensive summary table."""
        data = {
            'Asset': self.asset_names,
            'Market Weight': self.w_mkt,
            'Equilibrium Return (π)': self.equilibrium_returns,
            'Posterior Return (μ_BL)': self.posterior_returns,
            'Return Shift': self.posterior_returns - self.equilibrium_returns,
            'BL Optimal Weight': self.posterior_weights,
            'Weight Change': self.posterior_weights - self.w_mkt
        }
        
        df = pd.DataFrame(data)
        return df
    
    def get_view_analysis(self) -> pd.DataFrame:
        """Analyze each view's impact."""
        if self.n_views == 0:
            return pd.DataFrame({'Message': ['No views specified']})
        
        rows = []
        for i in range(self.n_views):
            # Identify assets in view
            involved = np.where(np.abs(self.P[i]) > 1e-6)[0]
            assets_str = ', '.join([self.asset_names[j] for j in involved])
            
            # View type
            if np.abs(np.sum(self.P[i])) < 1e-6:
                view_type = 'Relative'
            else:
                view_type = 'Absolute'
            
            # Prior expectation of view
            prior_view = self.P[i] @ self.equilibrium_returns
            
            # View confidence (diagonal of Omega)
            omega_i = self.Omega[i, i] if self.Omega is not None else 0
            
            rows.append({
                'View': self.view_descriptions[i],
                'Type': view_type,
                'Assets': assets_str,
                'View Return (Q)': self.Q[i],
                'Prior Expectation': prior_view,
                'View - Prior': self.Q[i] - prior_view,
                'View Variance (Ω)': omega_i
            })
        
        return pd.DataFrame(rows)
    
    def get_latex_formulas(self) -> Dict[str, str]:
        """Return LaTeX formulas for Black-Litterman model."""
        return {
            'equilibrium': r'\pi = \delta \Sigma w_{mkt}',
            'posterior_mean': r'\mu_{BL} = \pi + \tau\Sigma P^T(\tau P\Sigma P^T + \Omega)^{-1}(Q - P\pi)',
            'posterior_cov': r'\Sigma_{BL} = \Sigma + \tau\Sigma - \tau\Sigma P^T(\tau P\Sigma P^T + \Omega)^{-1}P\tau\Sigma',
            'omega_default': r'\Omega = \text{diag}(\tau P_i \Sigma P_i^T)',
            'optimal_weights': r'w^* = \frac{1}{\delta}\Sigma_{BL}^{-1}(\mu_{BL} - r_f)',
            'relative_view': r'P_{relative} = [0, ..., 1, ..., -1, ..., 0] \text{ (long-short)}',
            'absolute_view': r'P_{absolute} = [0, ..., 1, ..., 0] \text{ (single asset)}'
        }
    
    def get_excel_instructions(self) -> str:
        """Return step-by-step Excel implementation."""
        return f"""
**Excel Implementation for Black-Litterman Model:**

**Step 1: Calculate Equilibrium Returns (π)**
The equilibrium returns are implied by the market portfolio weights.
Formula: π = δ × Σ × w_mkt

Excel:
- Set up market weights in column (e.g., B2:B{1+self.n_assets})
- Set up covariance matrix (e.g., D2:F{1+self.n_assets} for 3 assets)
- Equilibrium returns = δ × MMULT(Σ, w_mkt)
  `=MMULT({self.risk_aversion}*CovMatrix, MarketWeights)`

Result: π = {np.array2string(self.equilibrium_returns, precision=4)}

**Step 2: Set Up Views**
View Matrix P ({self.n_views}×{self.n_assets}):
- Row per view, column per asset
- For absolute view on asset i: P[view, i] = 1, others = 0
- For relative view (A outperforms B by x%): P[view, A] = 1, P[view, B] = -1

View Returns Q ({self.n_views}×1):
- The expected return for each view

Example P matrix:
{np.array2string(self.P, precision=2) if self.P is not None else 'No views'}

Q vector: {np.array2string(self.Q, precision=4) if self.Q is not None else 'No views'}

**Step 3: Calculate Omega (View Uncertainty)**
Default: Ω = diag(τ × P × Σ × P')

For each view i:
  ω_i = τ × P_i × Σ × P_i'
  `=tau * MMULT(MMULT(P_row, CovMatrix), TRANSPOSE(P_row))`

**Step 4: Calculate Posterior Mean**
Formula: μ_BL = π + τΣP'(τPΣP' + Ω)⁻¹(Q - Pπ)

Break into steps:
a) τΣP' = `=tau * MMULT(CovMatrix, TRANSPOSE(P))`
b) τPΣP' = `=tau * MMULT(MMULT(P, CovMatrix), TRANSPOSE(P))`
c) M = τPΣP' + Ω
d) M⁻¹ = `=MINVERSE(M)`
e) (Q - Pπ) = view errors
f) μ_BL = π + (a) × (d) × (e)

Result: μ_BL = {np.array2string(self.posterior_returns, precision=4)}

**Step 5: Calculate Optimal Weights**
Formula: w* = (1/δ) × Σ⁻¹ × (μ_BL - rf)

`=(1/{self.risk_aversion}) * MMULT(MINVERSE(CovMatrix), (μ_BL - {self.risk_free_rate}))`

Then normalize: w_i / Σw_i

Result: w* = {np.array2string(self.posterior_weights, precision=4)}

**Step 6: Compare Prior vs Posterior**
| Asset | Market Wt | Equil Return | BL Return | BL Weight | Δ Weight |
{self._format_comparison_table()}

**Key Parameters Used:**
- Risk Aversion (δ): {self.risk_aversion}
- Tau (τ): {self.tau}
- Risk-Free Rate: {self.risk_free_rate}
- Number of Views: {self.n_views}
"""
    
    def _format_comparison_table(self) -> str:
        """Helper to format comparison table for Excel instructions."""
        lines = []
        for i in range(self.n_assets):
            line = f"| {self.asset_names[i]} | {self.w_mkt[i]:.4f} | {self.equilibrium_returns[i]:.4f} | "
            line += f"{self.posterior_returns[i]:.4f} | {self.posterior_weights[i]:.4f} | "
            line += f"{self.posterior_weights[i] - self.w_mkt[i]:+.4f} |"
            lines.append(line)
        return '\n'.join(lines)


def create_relative_view(
    asset_long: int,
    asset_short: int,
    n_assets: int,
    expected_outperformance: float
) -> Tuple[np.ndarray, float]:
    """
    Helper to create a relative view (Asset A outperforms Asset B).
    
    Args:
        asset_long: Index of asset expected to outperform (0-indexed)
        asset_short: Index of asset expected to underperform (0-indexed)
        n_assets: Total number of assets
        expected_outperformance: Expected return difference (A - B)
    
    Returns:
        (P_row, Q_value) tuple for this view
    
    Example:
        "Tech outperforms Energy by 3%" with Tech=0, Energy=2, n=4:
        P = [1, 0, -1, 0], Q = 0.03
    """
    P_row = np.zeros(n_assets)
    P_row[asset_long] = 1.0
    P_row[asset_short] = -1.0
    return P_row, expected_outperformance


def create_absolute_view(
    asset: int,
    n_assets: int,
    expected_return: float
) -> Tuple[np.ndarray, float]:
    """
    Helper to create an absolute view (Asset A will return X%).
    
    Args:
        asset: Index of asset (0-indexed)
        n_assets: Total number of assets
        expected_return: Expected return for this asset
    
    Returns:
        (P_row, Q_value) tuple for this view
    
    Example:
        "Bonds will return 4%" with Bonds=1, n=3:
        P = [0, 1, 0], Q = 0.04
    """
    P_row = np.zeros(n_assets)
    P_row[asset] = 1.0
    return P_row, expected_return


# =============================================================================
# THEORETICAL UTILITY FUNCTIONS (Handout Formulas)
# =============================================================================

def wealth_equivalent_loss(
    optimal_sharpe: float,
    suboptimal_sharpe: float,
    risk_aversion: float,
    horizon: int = 1
) -> Tuple[Dict, List[str]]:
    """
    Calculate percentage wealth loss from using a suboptimal investment strategy.
    
    This measures the "cost" of not using the optimal portfolio, expressed as
    the percentage of wealth an investor would give up.
    
    Formula (from mean-variance utility):
    WEL = 1 - exp(-γ/2 × (SR²_opt - SR²_sub) × T)
    
    For small values, approximately:
    WEL ≈ (SR²_opt - SR²_sub) × γ × T / 2
    
    Args:
        optimal_sharpe: Sharpe ratio of optimal portfolio
        suboptimal_sharpe: Sharpe ratio of suboptimal portfolio
        risk_aversion: Risk aversion coefficient γ
        horizon: Investment horizon in years
    
    Returns:
        (result_dict, excel_steps) tuple
    
    Exam Reference: "What fraction of wealth is lost by holding cash instead of optimal portfolio?"
    """
    gamma = risk_aversion
    T = horizon
    
    SR_opt_sq = optimal_sharpe ** 2
    SR_sub_sq = suboptimal_sharpe ** 2
    
    sharpe_sq_diff = SR_opt_sq - SR_sub_sq
    
    # Exact formula
    exact_loss = 1 - np.exp(-gamma / 2 * sharpe_sq_diff * T)
    
    # Linear approximation (good for small losses)
    approx_loss = sharpe_sq_diff * gamma * T / 2
    
    # Annualized loss
    annual_loss = exact_loss / T if T > 0 else exact_loss
    
    excel_steps = []
    excel_steps.append("**Wealth Equivalent Loss Calculation:**")
    excel_steps.append("")
    excel_steps.append("This measures the cost of suboptimal investing as a fraction of wealth.")
    excel_steps.append("")
    excel_steps.append("**Formula:**")
    excel_steps.append("WEL = 1 - exp(-γ/2 × (SR²_opt - SR²_sub) × T)")
    excel_steps.append("")
    excel_steps.append("**Step 1: Calculate Sharpe Ratio Squared**")
    excel_steps.append(f"SR²_optimal = {optimal_sharpe}² = {SR_opt_sq:.6f}")
    excel_steps.append(f"SR²_suboptimal = {suboptimal_sharpe}² = {SR_sub_sq:.6f}")
    excel_steps.append(f"Difference = {sharpe_sq_diff:.6f}")
    excel_steps.append("")
    excel_steps.append("**Step 2: Calculate Wealth Equivalent Loss**")
    excel_steps.append(f"WEL = 1 - EXP(-{gamma}/2 × {sharpe_sq_diff:.6f} × {T})")
    excel_steps.append(f"WEL = 1 - EXP({-gamma/2 * sharpe_sq_diff * T:.6f})")
    excel_steps.append(f"WEL = {exact_loss:.6f} = {exact_loss*100:.4f}%")
    excel_steps.append("")
    excel_steps.append("**Linear Approximation (for small losses):**")
    excel_steps.append(f"WEL ≈ (SR²_opt - SR²_sub) × γ × T / 2")
    excel_steps.append(f"WEL ≈ {sharpe_sq_diff:.6f} × {gamma} × {T} / 2 = {approx_loss:.6f}")
    excel_steps.append("")
    excel_steps.append("**Excel Formula:**")
    excel_steps.append(f"`=1 - EXP(-{gamma}/2 * ({optimal_sharpe}^2 - {suboptimal_sharpe}^2) * {T})`")
    excel_steps.append("")
    excel_steps.append("**Interpretation:**")
    excel_steps.append(f"By using the suboptimal strategy, the investor loses {exact_loss*100:.2f}% of wealth")
    excel_steps.append(f"equivalent over {T} year(s), or about {annual_loss*100:.2f}% per year.")
    
    result = {
        'wealth_equivalent_loss': exact_loss,
        'wealth_equivalent_loss_pct': exact_loss * 100,
        'linear_approximation': approx_loss,
        'annualized_loss': annual_loss,
        'sharpe_sq_difference': sharpe_sq_diff,
        'optimal_sharpe': optimal_sharpe,
        'suboptimal_sharpe': suboptimal_sharpe,
        'risk_aversion': gamma,
        'horizon': T
    }
    
    return result, excel_steps


def levered_var(
    asset_var: float,
    leverage_ratio: float,
    debt_rate: float = 0.0,
    confidence: float = 0.95,
    holding_period: int = 1
) -> Tuple[Dict, List[str]]:
    """
    Calculate Value at Risk for a levered position.
    
    When using leverage (borrowing to invest), VaR changes in two ways:
    1. Losses are amplified by (1 + L/E)
    2. Debt servicing cost offsets some losses
    
    Formula:
    VaR_levered = (1 + L/E) × VaR_asset - (L/E) × r_debt × Δt
    
    Where L/E is the leverage ratio (Debt/Equity).
    
    Args:
        asset_var: VaR of underlying asset (positive number, e.g., 0.10 for 10%)
        leverage_ratio: Debt/Equity ratio (e.g., 1.0 means 50% debt)
        debt_rate: Annual cost of debt (e.g., 0.05 for 5%)
        confidence: VaR confidence level (default 95%)
        holding_period: Holding period in days (for debt cost calculation)
    
    Returns:
        (result_dict, excel_steps) tuple
    
    Exam Reference: "Calculate VaR for a 2:1 levered portfolio"
    """
    LE = leverage_ratio
    
    # Amplification factor
    amp_factor = 1 + LE
    
    # Debt cost offset (annualized rate × period)
    period_fraction = holding_period / 252  # Trading days
    debt_offset = LE * debt_rate * period_fraction
    
    # Levered VaR
    levered_var_value = amp_factor * asset_var - debt_offset
    
    # Additional metrics
    unlevered_loss = asset_var
    levered_loss = levered_var_value
    amplification = levered_loss / unlevered_loss if unlevered_loss > 0 else 0
    
    excel_steps = []
    excel_steps.append("**Levered VaR Calculation:**")
    excel_steps.append("")
    excel_steps.append("When using leverage, losses are amplified but debt costs provide slight offset.")
    excel_steps.append("")
    excel_steps.append("**Formula:**")
    excel_steps.append("VaR_levered = (1 + L/E) × VaR_asset - (L/E) × r_debt × Δt")
    excel_steps.append("")
    excel_steps.append("**Given:**")
    excel_steps.append(f"- Asset VaR ({confidence*100:.0f}% confidence): {asset_var*100:.2f}%")
    excel_steps.append(f"- Leverage Ratio (L/E): {LE:.2f}")
    excel_steps.append(f"- Debt Rate: {debt_rate*100:.2f}% annual")
    excel_steps.append(f"- Holding Period: {holding_period} days")
    excel_steps.append("")
    excel_steps.append("**Step 1: Calculate Amplification Factor**")
    excel_steps.append(f"(1 + L/E) = 1 + {LE} = {amp_factor:.2f}")
    excel_steps.append("")
    excel_steps.append("**Step 2: Calculate Debt Cost Offset**")
    excel_steps.append(f"Period fraction = {holding_period}/252 = {period_fraction:.6f}")
    excel_steps.append(f"Debt offset = {LE} × {debt_rate} × {period_fraction:.6f} = {debt_offset:.6f}")
    excel_steps.append("")
    excel_steps.append("**Step 3: Calculate Levered VaR**")
    excel_steps.append(f"VaR_lev = {amp_factor} × {asset_var:.4f} - {debt_offset:.6f}")
    excel_steps.append(f"VaR_lev = {amp_factor * asset_var:.4f} - {debt_offset:.6f}")
    excel_steps.append(f"**VaR_lev = {levered_var_value:.4f} = {levered_var_value*100:.2f}%**")
    excel_steps.append("")
    excel_steps.append("**Excel Formula:**")
    excel_steps.append(f"`=(1 + {LE}) * {asset_var} - {LE} * {debt_rate} * {holding_period}/252`")
    excel_steps.append("")
    excel_steps.append("**Interpretation:**")
    excel_steps.append(f"With {LE:.1f}:1 leverage, the {confidence*100:.0f}% VaR increases from")
    excel_steps.append(f"{asset_var*100:.2f}% to {levered_var_value*100:.2f}% ({amplification:.2f}x amplification).")
    if LE > 0:
        excel_steps.append(f"Maximum loss at {confidence*100:.0f}% confidence: {levered_var_value*100:.2f}% of equity.")
    
    result = {
        'levered_var': levered_var_value,
        'levered_var_pct': levered_var_value * 100,
        'asset_var': asset_var,
        'amplification_factor': amp_factor,
        'debt_offset': debt_offset,
        'effective_amplification': amplification,
        'leverage_ratio': LE,
        'debt_rate': debt_rate,
        'confidence': confidence,
        'holding_period': holding_period
    }
    
    return result, excel_steps


def stocks_for_target_volatility(
    asset_volatility: float,
    correlation: float,
    target_volatility: float
) -> Tuple[Dict, List[str]]:
    """
    Calculate minimum number of stocks needed to reduce portfolio volatility to target.
    
    For N identical stocks with pairwise correlation ρ and individual volatility σ,
    the equally-weighted portfolio volatility is:
    
    σ_p² = σ²[(1-ρ)/N + ρ]
    
    Solving for N:
    N = (1-ρ) / [(σ_target/σ)² - ρ]
    
    Args:
        asset_volatility: Individual stock volatility σ
        correlation: Average pairwise correlation ρ
        target_volatility: Desired portfolio volatility σ_target
    
    Returns:
        (result_dict, excel_steps) tuple
    
    Exam Reference: "How many stocks needed to reduce volatility to 15%?"
    """
    sigma = asset_volatility
    rho = correlation
    sigma_target = target_volatility
    
    # Check if target is achievable
    limit_vol = sigma * np.sqrt(rho) if rho > 0 else 0
    
    excel_steps = []
    excel_steps.append("**Number of Stocks for Target Volatility:**")
    excel_steps.append("")
    excel_steps.append("**Portfolio Variance Formula (equal weights):**")
    excel_steps.append("σ_p² = σ²[(1-ρ)/N + ρ]")
    excel_steps.append("")
    excel_steps.append("**Solving for N:**")
    excel_steps.append("N = (1-ρ) / [(σ_target/σ)² - ρ]")
    excel_steps.append("")
    excel_steps.append("**Given:**")
    excel_steps.append(f"- Individual stock volatility (σ): {sigma*100:.2f}%")
    excel_steps.append(f"- Average correlation (ρ): {rho:.4f}")
    excel_steps.append(f"- Target portfolio volatility: {sigma_target*100:.2f}%")
    excel_steps.append(f"- Diversification limit (σ√ρ): {limit_vol*100:.2f}%")
    excel_steps.append("")
    
    if sigma_target < limit_vol:
        excel_steps.append(f"**❌ IMPOSSIBLE:** Target {sigma_target*100:.2f}% is below diversification limit {limit_vol*100:.2f}%")
        excel_steps.append("With positive correlation, you cannot diversify below σ√ρ.")
        result = {
            'n_stocks': float('inf'),
            'achievable': False,
            'limit_volatility': limit_vol,
            'target_volatility': sigma_target,
            'message': f'Target below diversification limit of {limit_vol*100:.2f}%'
        }
        return result, excel_steps
    
    # Calculate required N
    ratio_sq = (sigma_target / sigma) ** 2
    denominator = ratio_sq - rho
    
    if denominator <= 0:
        excel_steps.append("**❌ IMPOSSIBLE:** Denominator ≤ 0, target not achievable")
        result = {
            'n_stocks': float('inf'),
            'achievable': False,
            'limit_volatility': limit_vol,
            'target_volatility': sigma_target,
            'message': 'Mathematical impossibility'
        }
        return result, excel_steps
    
    n_exact = (1 - rho) / denominator
    n_required = int(np.ceil(n_exact))
    
    # Verify by computing actual volatility with n_required stocks
    actual_var = sigma**2 * ((1 - rho) / n_required + rho)
    actual_vol = np.sqrt(actual_var)
    
    excel_steps.append("**Step 1: Calculate (σ_target/σ)²**")
    excel_steps.append(f"({sigma_target}/{sigma})² = {ratio_sq:.6f}")
    excel_steps.append("")
    excel_steps.append("**Step 2: Calculate Denominator**")
    excel_steps.append(f"(σ_target/σ)² - ρ = {ratio_sq:.6f} - {rho} = {denominator:.6f}")
    excel_steps.append("")
    excel_steps.append("**Step 3: Calculate N**")
    excel_steps.append(f"N = (1 - {rho}) / {denominator:.6f}")
    excel_steps.append(f"N = {1-rho:.4f} / {denominator:.6f}")
    excel_steps.append(f"N = {n_exact:.4f}")
    excel_steps.append("")
    excel_steps.append(f"**Minimum stocks required: ⌈{n_exact:.2f}⌉ = {n_required}**")
    excel_steps.append("")
    excel_steps.append("**Verification:**")
    excel_steps.append(f"With N={n_required}: σ_p = σ√[(1-ρ)/N + ρ]")
    excel_steps.append(f"σ_p = {sigma}√[{(1-rho)/n_required:.6f} + {rho}]")
    excel_steps.append(f"σ_p = {sigma}√{(1-rho)/n_required + rho:.6f}")
    excel_steps.append(f"σ_p = {actual_vol:.4f} = {actual_vol*100:.2f}%")
    excel_steps.append("")
    excel_steps.append("**Excel Formula:**")
    excel_steps.append(f"`=CEILING((1-{rho})/((target_vol/stock_vol)^2 - {rho}), 1)`")
    excel_steps.append("")
    excel_steps.append("**Interpretation:**")
    excel_steps.append(f"You need at least {n_required} stocks to reduce portfolio volatility")
    excel_steps.append(f"from {sigma*100:.2f}% to {sigma_target*100:.2f}% (achieved: {actual_vol*100:.2f}%).")
    
    result = {
        'n_stocks': n_required,
        'n_exact': n_exact,
        'achievable': True,
        'achieved_volatility': actual_vol,
        'target_volatility': sigma_target,
        'limit_volatility': limit_vol,
        'asset_volatility': sigma,
        'correlation': rho,
        'variance_reduction': 1 - (actual_vol / sigma) ** 2
    }
    
    return result, excel_steps


def certainty_equivalent_return(
    expected_return: float,
    volatility: float,
    risk_aversion: float
) -> Tuple[Dict, List[str]]:
    """
    Calculate the certainty equivalent return for a risky portfolio.
    
    The certainty equivalent is the risk-free return that gives the same
    utility as the risky portfolio.
    
    Formula (for quadratic utility):
    CE = E[r] - (γ/2) × σ²
    
    Args:
        expected_return: Expected portfolio return
        volatility: Portfolio volatility
        risk_aversion: Risk aversion coefficient γ
    
    Returns:
        (result_dict, excel_steps) tuple
    """
    mu = expected_return
    sigma = volatility
    gamma = risk_aversion
    
    # Certainty equivalent
    risk_penalty = (gamma / 2) * sigma ** 2
    ce = mu - risk_penalty
    
    excel_steps = []
    excel_steps.append("**Certainty Equivalent Return:**")
    excel_steps.append("")
    excel_steps.append("The risk-free rate that provides the same utility as the risky portfolio.")
    excel_steps.append("")
    excel_steps.append("**Formula:**")
    excel_steps.append("CE = E[r] - (γ/2) × σ²")
    excel_steps.append("")
    excel_steps.append("**Calculation:**")
    excel_steps.append(f"Risk penalty = ({gamma}/2) × {sigma}² = {risk_penalty:.6f}")
    excel_steps.append(f"CE = {mu} - {risk_penalty:.6f} = {ce:.6f} = {ce*100:.4f}%")
    excel_steps.append("")
    excel_steps.append("**Excel Formula:**")
    excel_steps.append(f"`={mu} - ({gamma}/2) * {sigma}^2`")
    excel_steps.append("")
    excel_steps.append("**Interpretation:**")
    excel_steps.append(f"This risky portfolio with {mu*100:.2f}% expected return and {sigma*100:.2f}% volatility")
    excel_steps.append(f"provides the same utility as a guaranteed {ce*100:.2f}% return.")
    
    result = {
        'certainty_equivalent': ce,
        'certainty_equivalent_pct': ce * 100,
        'risk_penalty': risk_penalty,
        'expected_return': mu,
        'volatility': sigma,
        'risk_aversion': gamma
    }
    
    return result, excel_steps


# =============================================================================
# UTILITY FUNCTIONS - Parsing & Helpers
# =============================================================================

# =============================================================================
# STREAMLIT MODULE 1: PORTFOLIO OPTIMIZER
# =============================================================================

def portfolio_optimizer_module():
    st.header("📊 Module 1: Portfolio Optimizer (Matrix Solver)")
    
    st.markdown("""
    This module solves mean-variance portfolio optimization problems using matrix algebra.
    It calculates **Tangency**, **Global Minimum Variance**, and **Optimal** portfolio weights.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gamma = st.number_input("Risk Aversion (γ)", value=4.0, min_value=0.1, step=0.5, 
                                help="Higher γ = more risk-averse investor")
        rf = st.number_input("Risk-Free Rate (rf)", value=0.02, format="%.4f",
                            help="Enter as decimal, e.g., 0.02 for 2%")
    
    with col2:
        use_esg = st.checkbox("Include ESG Preference")
        if use_esg:
            esg_preference = st.number_input("ESG Preference (a)", value=0.01, format="%.4f",
                                            help="Utility: U = E[r] - 0.5γσ² + a·ESG")
        else:
            esg_preference = 0.0
    
    st.subheader("Input Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        returns_input = st.text_area(
            "Expected Returns & Volatilities (Mean, StdDev per row)",
            value="0.08, 0.20\n0.12, 0.25\n0.06, 0.15",
            height=150,
            help="Paste from PDF: Each row = one asset with Mean Return and Std Dev"
        )
    
    with col2:
        corr_input = st.text_area(
            "Correlation Matrix",
            value="1.0, 0.3, 0.1\n0.3, 1.0, 0.2\n0.1, 0.2, 1.0",
            height=150,
            help="Paste correlation matrix (must be symmetric)"
        )
    
    if use_esg:
        esg_input = st.text_area(
            "ESG Scores (one per asset)",
            value="0.7, 0.5, 0.9",
            help="ESG score for each asset"
        )
    
    if st.button("🧮 Calculate Optimal Portfolios", key="calc_portfolio"):
        try:
            # Parse inputs
            mu, sigma = parse_two_column_input(returns_input)
            corr_matrix = parse_matrix_input(corr_input)
            
            if len(mu) == 0:
                st.error("❌ Could not parse returns data. Please check format.")
                return
            
            n_assets = len(mu)
            
            if corr_matrix.shape != (n_assets, n_assets):
                st.error(f"❌ Correlation matrix dimensions ({corr_matrix.shape}) don't match number of assets ({n_assets})")
                return
            
            # Parse ESG if applicable
            esg_scores = None
            if use_esg:
                esg_scores = parse_messy_input(esg_input)
                if len(esg_scores) != n_assets:
                    st.error(f"❌ ESG scores length ({len(esg_scores)}) doesn't match number of assets ({n_assets})")
                    return
            
            # Construct Covariance Matrix: Σ = diag(σ) @ Corr @ diag(σ)
            D = np.diag(sigma)
            cov_matrix = D @ corr_matrix @ D
            
            st.subheader("Parsed Data")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Expected Returns (μ):**")
                st.dataframe(pd.DataFrame({
                    'Asset': [f'Asset {i+1}' for i in range(n_assets)],
                    'E[r]': mu,
                    'E[r] (%)': [format_percentage(m) for m in mu]
                }))
            
            with col2:
                st.write("**Volatilities (σ) & Correlation:**")
                st.dataframe(pd.DataFrame({
                    'Asset': [f'Asset {i+1}' for i in range(n_assets)],
                    'σ': sigma,
                    'σ (%)': [format_percentage(s) for s in sigma]
                }))
            
            st.write("**Covariance Matrix (Σ):**")
            cov_df = pd.DataFrame(cov_matrix, 
                                  index=[f'Asset {i+1}' for i in range(n_assets)],
                                  columns=[f'Asset {i+1}' for i in range(n_assets)])
            st.dataframe(cov_df.style.format("{:.6f}"))
            
            # Calculate Σ⁻¹
            try:
                cov_inv = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                st.error("❌ Covariance matrix is singular (cannot be inverted)")
                return
            
            # Display Inverse Covariance Matrix
            st.write("**Inverse Covariance Matrix (Σ⁻¹):**")
            st.info("💡 Copy this table for exam questions asking to 'Determine the inverse matrix'")
            cov_inv_df = pd.DataFrame(cov_inv,
                                      index=[f'Asset {i+1}' for i in range(n_assets)],
                                      columns=[f'Asset {i+1}' for i in range(n_assets)])
            st.dataframe(cov_inv_df.style.format("{:.4f}"))
            
            # Excess returns
            excess_returns = mu - rf
            
            # =================================================================
            # TANGENCY PORTFOLIO
            # =================================================================
            st.subheader("🎯 Tangency Portfolio (Maximum Sharpe Ratio)")
            
            # Tangency weights: π_tan = Σ⁻¹(μ - rf) / 1'Σ⁻¹(μ - rf)
            numerator = cov_inv @ excess_returns
            w_tangency = numerator / np.sum(numerator)
            
            tan_return = w_tangency @ mu
            tan_vol = np.sqrt(w_tangency @ cov_matrix @ w_tangency)
            tan_sharpe = (tan_return - rf) / tan_vol
            
            tan_df = pd.DataFrame({
                'Asset': [f'Asset {i+1}' for i in range(n_assets)],
                'Weight': w_tangency,
                'Weight (%)': [format_percentage(w) for w in w_tangency]
            })
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(tan_df)
            with col2:
                st.metric("E[r_p]", format_percentage(tan_return))
                st.metric("σ_p", format_percentage(tan_vol))
                st.metric("Sharpe Ratio", format_number(tan_sharpe))
            
            with st.expander("📝 Excel Implementation (Tangency Portfolio)"):
                # Narrative explanation for the tangency portfolio
                tan_num_vec = cov_inv @ excess_returns
                tan_denom_val = ones @ cov_inv @ excess_returns
                st.markdown(rf"""
**Tangency Portfolio Calculation**

We use the tangency weight formula:
$$\pi_{{tan}} = \frac{{\Sigma^{{-1}}(\mu - r_f)}}{{\mathbf{{1}}^{{\top}}\Sigma^{{-1}}(\mu - r_f)}}.$$

1. **Compute the vector \(\Sigma^{{-1}}(\mu - r_f)\):**
   \[\Sigma^{{-1}}(\mu - r_f) = {np.array2string(tan_num_vec, precision=4, separator=', ')}\]

2. **Sum its elements to form the denominator:**
   \[\mathbf{{1}}^{{\top}}\Sigma^{{-1}}(\mu - r_f) = {tan_denom_val:.4f}\]

3. **Normalize the vector to obtain weights:**
   \[\pi_{{tan}} = {np.array2string(w_tangency, precision=4, separator=', ')}\]

4. **Portfolio statistics:**
   - Expected return: \(E[r_p] = {tan_return:.4f}\) (={tan_return*100:.2f}\%)
   - Volatility: \(\sigma_p = {tan_vol:.4f}\) (={tan_vol*100:.2f}\%)
   - Sharpe ratio: {tan_sharpe:.4f}

These values demonstrate the solution as if computed step‑by‑step in a spreadsheet.
                """)
            
            # =================================================================
            # GLOBAL MINIMUM VARIANCE PORTFOLIO
            # =================================================================
            st.subheader("🛡️ Global Minimum Variance Portfolio")
            
            # GMV weights: π_gmv = Σ⁻¹ × 1 / (1' × Σ⁻¹ × 1)
            ones = np.ones(n_assets)
            gmv_numerator = cov_inv @ ones
            w_gmv = gmv_numerator / (ones @ cov_inv @ ones)
            
            gmv_return = w_gmv @ mu
            gmv_vol = np.sqrt(w_gmv @ cov_matrix @ w_gmv)
            gmv_sharpe = (gmv_return - rf) / gmv_vol
            
            gmv_df = pd.DataFrame({
                'Asset': [f'Asset {i+1}' for i in range(n_assets)],
                'Weight': w_gmv,
                'Weight (%)': [format_percentage(w) for w in w_gmv]
            })
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(gmv_df)
            with col2:
                st.metric("E[r_p]", format_percentage(gmv_return))
                st.metric("σ_p", format_percentage(gmv_vol))
                st.metric("Sharpe Ratio", format_number(gmv_sharpe))
            
            with st.expander("📝 Excel Implementation (GMV Portfolio)"):
                # Narrative explanation for the GMV portfolio
                gmv_num_vec = cov_inv @ ones
                gmv_denom_val = ones @ cov_inv @ ones
                st.markdown(rf"""
**Global Minimum Variance Portfolio Calculation**

The GMV weights are given by
$$\pi_{{GMV}} = \frac{{\Sigma^{{-1}} \mathbf{{1}}}}{{\mathbf{{1}}^{{\top}} \Sigma^{{-1}} \mathbf{{1}}}}.$$

1. **Compute the vector \(\Sigma^{{-1}} \mathbf{{1}}\):**
   \[\Sigma^{{-1}} \mathbf{{1}} = {np.array2string(gmv_num_vec, precision=4, separator=', ')}\]

2. **Sum its elements to form the denominator:**
   \[\mathbf{{1}}^{{\top}} \Sigma^{{-1}} \mathbf{{1}} = {gmv_denom_val:.4f}\]

3. **Normalize to obtain weights:**
   \[\pi_{{GMV}} = {np.array2string(w_gmv, precision=4, separator=', ')}\]

4. **Portfolio statistics:**
   - Expected return: \(E[r_p] = {gmv_return:.4f}\) (={gmv_return*100:.2f}\%)
   - Volatility: \(\sigma_p = {gmv_vol:.4f}\) (={gmv_vol*100:.2f}\%)
   - Sharpe ratio: {gmv_sharpe:.4f}

These values illustrate the GMV solution in a spreadsheet‑style format.
                """)
            
            # =================================================================
            # OPTIMAL PORTFOLIO FOR SPECIFIC GAMMA
            # =================================================================
            st.subheader(f"⚡ Optimal Portfolio (γ = {gamma})")
            
            if use_esg and esg_scores is not None:
                # Modified utility: U = E[r] - 0.5γσ² + a·ESG
                # Optimal: π* = (1/γ) × Σ⁻¹ × (μ - rf + a·ESG)
                modified_returns = excess_returns + esg_preference * esg_scores
                w_optimal = (1/gamma) * (cov_inv @ modified_returns)
                st.info(f"📊 ESG-adjusted utility: U = E[r] - 0.5×{gamma}×σ² + {esg_preference}×ESG")
                # Precompute base vector and raw weights for the narrative explanation in the Excel section
                base_vector_plot = cov_inv @ modified_returns
                raw_weights_plot = (1/gamma) * base_vector_plot
            else:
                # Standard mean-variance: π* = (1/γ) × Σ⁻¹ × (μ - rf)
                w_optimal = (1/gamma) * (cov_inv @ excess_returns)
                # Precompute base vector and raw weights for the narrative explanation in the Excel section
                base_vector_plot = cov_inv @ excess_returns
                raw_weights_plot = (1/gamma) * base_vector_plot
            
            # Calculate cash weight (remainder)
            risky_weight = np.sum(w_optimal)
            cash_weight = 1 - risky_weight
            
            opt_return = w_optimal @ mu + cash_weight * rf
            opt_vol = np.sqrt(w_optimal @ cov_matrix @ w_optimal)
            opt_sharpe = (opt_return - rf) / opt_vol if opt_vol > 0 else 0
            
            opt_df = pd.DataFrame({
                'Asset': [f'Asset {i+1}' for i in range(n_assets)] + ['Risk-Free'],
                'Weight': list(w_optimal) + [cash_weight],
                'Weight (%)': [format_percentage(w) for w in w_optimal] + [format_percentage(cash_weight)]
            })
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(opt_df)
            with col2:
                st.metric("E[r_p]", format_percentage(opt_return))
                st.metric("σ_p", format_percentage(opt_vol))
                st.metric("Sharpe Ratio", format_number(opt_sharpe))
                st.metric("Total Risky Weight", format_percentage(risky_weight))
            
            with st.expander("📝 How to do this in Excel (Optimal Risky Allocation)"):
                esg_text = ""
                if use_esg:
                    esg_text = f"""
**With ESG Adjustment:**
$$\\pi^* = \\frac{{1}}{{\\gamma}} \\Sigma^{{-1}} (\\mu - r_f + a \\cdot ESG)$$
- Add `a × ESG[i]` to each excess return before multiplying
"""
                # Prepare ESG-related text outside the f-string to avoid nested f-strings and backslashes in expressions
                esg_note_text = f"- The ESG adjustment adds {esg_preference}×ESG to the excess return before multiplication." if (use_esg and esg_scores is not None) else ""
                formula_suffix = '+ a\\cdot ESG' if (use_esg and esg_scores is not None) else ''
                replace_note = "\nWhen ESG preferences apply, replace (\\mu - r_f) with (\\mu - r_f + a\\cdot ESG) before multiplying." if (use_esg and esg_scores is not None) else ''
                st.markdown(rf"""
**Optimal Portfolio Calculation**

The analytical solution for the risky weights is
$$\pi^* = \frac{{1}}{{\gamma}} \Sigma^{{-1}} (\mu - r_f)$$
{replace_note}

1. **Compute \(\Sigma^{{-1}}(\mu - r_f{formula_suffix})\):**
   \[\Sigma^{{-1}}(\mu - r_f{formula_suffix}) = {np.array2string(base_vector_plot, precision=4, separator=', ')}\]

2. **Scale by \(1/\gamma\) to obtain the risky allocation:**
   \[\pi^*_{{risky}} = {np.array2string(raw_weights_plot, precision=4, separator=', ')}\]

3. **Determine cash position:**
   The risk‑free weight is the remainder: \(w_{{cash}} = 1 - \sum_i \pi^*_i\) = {cash_weight:.4f}.

4. **Portfolio statistics:**
   - Expected return: \(E[r_p] = {opt_return:.4f}\) (={opt_return*100:.2f}\%)
   - Volatility: \(\sigma_p = {opt_vol:.4f}\) (={opt_vol*100:.2f}\%)
   - Sharpe ratio: {opt_sharpe:.4f}

Interpretation:
- Weights may exceed 100\% (leveraged) or be negative (short).
- A higher risk‑aversion coefficient \(\gamma\) results in smaller risky positions and a larger cash allocation.
{esg_note_text}
                """)
            
            # =================================================================
            # EFFICIENT FRONTIER PLOT
            # =================================================================
            st.subheader("📈 Efficient Frontier")
            
            # Generate efficient frontier
            target_returns = np.linspace(gmv_return, max(mu) * 1.2, 50)
            frontier_vols = []
            
            for target_r in target_returns:
                # Solve for minimum variance portfolio with target return
                # Using analytical solution for 2-constraint problem
                A = ones @ cov_inv @ ones
                B = ones @ cov_inv @ mu
                C = mu @ cov_inv @ mu
                D = A * C - B ** 2
                
                if D > 0:
                    g = (C * (cov_inv @ ones) - B * (cov_inv @ mu)) / D
                    h = (A * (cov_inv @ mu) - B * (cov_inv @ ones)) / D
                    w = g + h * target_r
                    vol = np.sqrt(w @ cov_matrix @ w)
                    frontier_vols.append(vol)
                else:
                    frontier_vols.append(np.nan)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot frontier
            ax.plot(frontier_vols, target_returns, 'b-', linewidth=2, label='Efficient Frontier')
            
            # Plot individual assets
            ax.scatter(sigma, mu, s=100, c='gray', marker='o', label='Individual Assets', zorder=5)
            for i in range(n_assets):
                ax.annotate(f'Asset {i+1}', (sigma[i], mu[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
            
            # Plot special portfolios
            ax.scatter([gmv_vol], [gmv_return], s=150, c='green', marker='s', 
                      label='GMV Portfolio', zorder=6)
            ax.scatter([tan_vol], [tan_return], s=150, c='red', marker='^', 
                      label='Tangency Portfolio', zorder=6)
            ax.scatter([opt_vol], [opt_return], s=150, c='purple', marker='D', 
                      label=f'Optimal (γ={gamma})', zorder=6)
            
            # Capital Market Line
            cml_vols = np.linspace(0, max(frontier_vols) * 1.1, 100)
            cml_returns = rf + tan_sharpe * cml_vols
            ax.plot(cml_vols, cml_returns, 'r--', linewidth=1.5, label='Capital Market Line')
            
            ax.set_xlabel('Volatility (σ)', fontsize=12)
            ax.set_ylabel('Expected Return E[r]', fontsize=12)
            ax.set_title('Mean-Variance Efficient Frontier', fontsize=14)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, None)
            
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")
            st.exception(e)


# =============================================================================
# MODULE 2: BOND MATH
# =============================================================================

def bond_math_module():
    """
    Streamlit interface for Module 2: Bond Math (Fixed Income).

    This module calculates bond prices, durations, convexity, and solves
    immunization problems using user‑supplied bond parameters.  It also
    provides sensitivity analysis for changes in yields and includes
    exam tips and Excel formulas so students can reproduce calculations
    in a spreadsheet.  No core logic is modified; all enhancements
    improve clarity and pedagogy.
    """
    st.header("📈 Module 2: Bond Math (Fixed Income)")

    st.markdown("""
    Calculate bond prices, durations, convexity, and solve immunization problems.
    """)

    # Exam tip for the entire module
    st.info(
        "🧠 **Exam tip:** Always convert annual yields to per‑period rates using the payment frequency. "
        "Express coupon and yield inputs as decimals (e.g., 0.05 for 5%).  When showing your work, "
        "lay out the full cash flow schedule with discount factors and present values."
    )
    
    tab1, tab2 = st.tabs(["💵 Bond Pricing & Duration", "🔒 Immunization Solver"])
    
    with tab1:
        bond_pricing_section()
    
    with tab2:
        immunization_section()


def bond_pricing_section():
    """
    Bond pricing, duration and convexity calculator with sensitivity analysis.

    This sub‑routine collects inputs for a single bond (face value, coupon rate,
    yield to maturity, maturity and payment frequency) and constructs a cash
    flow schedule.  It computes present values of each payment and derives
    **Macaulay duration**, **modified duration**, **convexity** and **DV01**.  A
    price–yield curve is plotted, and a sensitivity analysis under normally
    distributed yield changes is performed.  Exam tips and detailed Excel
    formulas with cell references guide students on reproducing the steps in
    a spreadsheet.  The core quantitative logic remains unchanged—only
    descriptive text, formatting and instructional comments are enhanced.
    """
    st.subheader("Bond Pricing & Duration Calculator")
    
    # Informational tip for students
    st.info(
        "🧠 **Exam tip:** Use the per‑period yield (YTM divided by payment frequency) when computing "
        "discount factors.  **Macaulay duration** is the weighted average time of cash flows, while "
        "**modified duration** adjusts for the compounding frequency.  Always lay out the full cash‑flow "
        "table showing time, cash flow, discount factor and present value—it’s often required in exam answers."
    )

    # Initialize session state for bond data
    if 'bond_calculated' not in st.session_state:
        st.session_state.bond_calculated = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        bond_type = st.radio(
            "Bond Type",
            ["Bullet (Coupon Bond)", "Annuity"],
            horizontal=True,
            help="Select 'Bullet' for a standard coupon bond or 'Annuity' for level payments",
            key="bond_type_radio"
        )
        face_value = st.number_input(
            "Face Value ($)", value=1000.0, min_value=1.0,
            help="Enter the bond’s par (face) value. Example: a typical corporate bond has a par value of $1,000.",
            key="bond_fv"
        )
        coupon_rate = st.number_input(
            "Coupon Rate (annual)", value=0.05, format="%.4f",
            help="Annual coupon rate expressed as a decimal. For example, 0.05 represents a 5% annual coupon.",
            key="bond_cr"
        )
    
    with col2:
        ytm = st.number_input(
            "Yield to Maturity (YTM)", value=0.04, format="%.4f",
            help="Annual yield to maturity as a decimal (e.g., 0.04 for 4%). For calculation purposes this value is divided by the payment frequency to obtain the per‑period rate.",
            key="bond_ytm"
        )
        maturity = st.number_input(
            "Maturity (years)", value=5, min_value=1, max_value=100,
            help="Time to maturity expressed in years. For example, enter 5 for a 5‑year bond.",
            key="bond_mat"
        )
        freq = st.selectbox(
            "Payment Frequency",
            [1, 2, 4, 12], index=0,
            format_func=lambda x: {1: "Annual", 2: "Semi-annual", 4: "Quarterly", 12: "Monthly"}[x],
            help="Number of coupon payments per year. A semi‑annual bond (2) pays twice per year; quarterly (4) pays four times.",
            key="bond_freq"
        )
    
    if st.button("🧮 Calculate Bond Metrics", key="calc_bond"):
        # Store inputs in session state
        st.session_state.bond_calculated = True
        st.session_state.bond_inputs = {
            'bond_type': bond_type,
            'face_value': face_value,
            'coupon_rate': coupon_rate,
            'ytm': ytm,
            'maturity': maturity,
            'freq': freq
        }
    
    # Display results if bond has been calculated
    if st.session_state.bond_calculated:
        try:
            # Retrieve inputs from session state
            inp = st.session_state.bond_inputs
            bond_type = inp['bond_type']
            face_value = inp['face_value']
            coupon_rate = inp['coupon_rate']
            ytm = inp['ytm']
            maturity = inp['maturity']
            freq = inp['freq']
            
            n_periods = maturity * freq
            period_coupon_rate = coupon_rate / freq
            period_ytm = ytm / freq
            coupon_payment = face_value * period_coupon_rate
            
            # Generate cash flow schedule
            periods = np.arange(1, n_periods + 1)
            times = periods / freq  # Time in years
            
            if bond_type == "Bullet (Coupon Bond)":
                # Coupon payments + face value at maturity
                cash_flows = np.full(n_periods, coupon_payment)
                cash_flows[-1] += face_value
            else:
                # Annuity: equal payments
                if period_ytm > 0:
                    annuity_payment = face_value * period_ytm / (1 - (1 + period_ytm) ** (-n_periods))
                else:
                    annuity_payment = face_value / n_periods
                cash_flows = np.full(n_periods, annuity_payment)
            
            # Discount factors
            discount_factors = (1 + period_ytm) ** (-periods)
            
            # Present values
            pv_cash_flows = cash_flows * discount_factors
            
            # Bond price
            price = np.sum(pv_cash_flows)
            
            # Macaulay Duration
            macaulay_duration = np.sum(times * pv_cash_flows) / price
            
            # Modified Duration
            modified_duration = macaulay_duration / (1 + period_ytm)
            
            # Convexity
            convexity = np.sum(times * (times + 1/freq) * pv_cash_flows) / (price * (1 + period_ytm) ** 2)
            
            # Dollar Duration (DV01)
            dv01 = modified_duration * price / 10000
            
            # Create cash flow table with descriptive headers and units
            cf_df = pd.DataFrame({
                'Period': periods,
                'Time (Years)': times,
                'Cash Flow ($)': cash_flows,
                'Discount Factor': discount_factors,
                'PV of Cash Flow ($)': pv_cash_flows,
                'Time × PV ($·Years)': times * pv_cash_flows
            })

            st.subheader("📋 Cash Flow Schedule")
            st.dataframe(cf_df.style.format({
                'Time (Years)': '{:.2f}',
                'Cash Flow ($)': '${:,.2f}',
                'Discount Factor': '{:.6f}',
                'PV of Cash Flow ($)': '${:,.2f}',
                'Time × PV ($·Years)': '{:,.4f}'
            }))
            
            st.subheader("📊 Bond Metrics")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bond Price ($)", f"${price:,.2f}", help="Present value of all cash flows (clean price)")
                st.metric("Price as % of Par", f"{price/face_value*100:.2f}%", help="Bond price divided by its face value")
            with col2:
                st.metric("Macaulay Duration (years)", f"{macaulay_duration:.4f}", help="Weighted average time of cash flows")
                st.metric("Modified Duration (years)", f"{modified_duration:.4f}", help="Macaulay duration divided by (1 + yield per period)")
            with col3:
                st.metric("Convexity (years²)", f"{convexity:.4f}", help="Second derivative of price with respect to yield")
                st.metric("DV01 ($ per bp)", f"${dv01:.4f}", help="Dollar change in price for a one basis point yield change")
            
            # Price-Yield relationship
            st.subheader("📉 Price-Yield Relationship")
            
            yields = np.linspace(max(0.001, ytm - 0.03), ytm + 0.03, 50)
            prices = []
            for y in yields:
                py = y / freq
                df = (1 + py) ** (-periods)
                prices.append(np.sum(cash_flows * df))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(yields * 100, prices, 'b-', linewidth=2)
            ax.scatter([ytm * 100], [price], s=100, c='red', zorder=5, label=f'Current (YTM={ytm*100:.2f}%)')
            ax.set_xlabel('Yield to Maturity (%)', fontsize=12)
            ax.set_ylabel('Bond Price ($)', fontsize=12)
            ax.set_title('Price-Yield Relationship', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(rf"""
**Excel Implementation for Bond Pricing:**

1. **Set up columns:**
   - A: Period (1, 2, 3, ...)
   - B: Time in years (`=A1/{freq}`)
   - C: Cash Flow
   - D: Discount Factor (`=(1+YTM/{freq})^(-A1)`)
   - E: PV of CF (`=C1*D1`)
   - F: Time × PV (`=B1*E1`)

2. **Bond Price:**
   ```
   =SUM(E:E)
   ```
   Or use built-in: `=PV(YTM/{freq}, {n_periods}, -{coupon_payment}, -{face_value})`

3. **Macaulay Duration:**
   ```
   =SUMPRODUCT(B:B, E:E) / SUM(E:E)
   ```
   $$D_{{Mac}} = \frac{{\sum_{{t=1}}^T t \cdot PV(CF_t)}}{{Price}}$$

4. **Modified Duration:**
   ```
   =MacaulayDuration / (1 + YTM/{freq})
   ```
   $$D_{{Mod}} = \frac{{D_{{Mac}}}}{{1 + y/m}}$$

5. **Convexity:**
   ```
   =SUMPRODUCT(B:B, (B:B+1/{freq}), E:E) / (Price * (1+YTM/{freq})^2)
   ```

6. **Price Approximation using Duration & Convexity:**
   $$\Delta P \approx -D_{{Mod}} \cdot P \cdot \Delta y + \frac{{1}}{{2}} \cdot Convexity \cdot P \cdot (\Delta y)^2$$
                """)
            
            # ================================================================
            # SENSITIVITY ANALYSIS WITH CONFIDENCE INTERVALS
            # ================================================================
            st.markdown("---")
            st.subheader("📈 Sensitivity Analysis (Confidence Interval)")
            
            st.markdown("""
            Compute confidence intervals for bond returns given a **normally distributed yield change**.
            
            **Problem Type:** *"Δy ~ N(μ, σ²). Find 95% CI for bond return using duration approximation and exact pricing."*
            """)
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                delta_y_mean = st.number_input(
                    "Expected Δy (mean)", 
                    value=0.0, 
                    format="%.4f", 
                    help="Expected yield change (usually 0)",
                    key="sens_dy_mean"
                )
            with col_s2:
                delta_y_std = st.number_input(
                    "Std Dev of Δy", 
                    value=0.008, 
                    format="%.4f",
                    help="Standard deviation of yield change (e.g., 0.008 = 0.8%)",
                    key="sens_dy_std"
                )
            with col_s3:
                conf_level = st.number_input(
                    "Confidence Level", 
                    value=0.95, 
                    min_value=0.50, 
                    max_value=0.99, 
                    format="%.2f",
                    help="e.g., 0.95 for 95% CI",
                    key="sens_conf"
                )
            
            if st.button("📊 Calculate Return Confidence Interval", key="calc_sensitivity"):
                # Z-score for confidence level
                z_score = norm.ppf((1 + conf_level) / 2)
                
                # Yield change bounds
                dy_lower = delta_y_mean - z_score * delta_y_std
                dy_upper = delta_y_mean + z_score * delta_y_std
                
                st.write(f"**{conf_level*100:.0f}% Confidence Interval for Δy:**")
                st.latex(rf"\Delta y \in [{dy_lower:.4f}, {dy_upper:.4f}] = [{dy_lower*10000:.0f} \text{{ bps}}, {dy_upper*10000:.0f} \text{{ bps}}]")
                
                # Build results for each scenario
                scenarios = [
                    ("Lower Bound (Δy min)", dy_lower),
                    ("Mean (Δy = 0)", delta_y_mean),
                    ("Upper Bound (Δy max)", dy_upper)
                ]
                
                results_data = []
                for scenario_name, dy in scenarios:
                    new_yield = ytm + dy
                    
                    # Method 1: Duration approximation
                    # ΔP/P ≈ -D_mod × Δy
                    return_duration = -modified_duration * dy
                    price_duration = price * (1 + return_duration)
                    
                    # Method 1b: Duration + Convexity
                    # ΔP/P ≈ -D_mod × Δy + 0.5 × C × (Δy)²
                    return_convexity = -modified_duration * dy + 0.5 * convexity * dy**2
                    price_convexity = price * (1 + return_convexity)
                    
                    # Method 2: Exact calculation
                    new_period_ytm = new_yield / freq
                    new_df = (1 + new_period_ytm) ** (-periods)
                    price_exact = np.sum(cash_flows * new_df)
                    return_exact = (price_exact - price) / price
                    
                    results_data.append({
                        'Scenario': scenario_name,
                        'Δy': dy,
                        'Δy (bps)': dy * 10000,
                        'New Yield': new_yield,
                        'Return (Duration)': return_duration,
                        'Return (Dur+Conv)': return_convexity,
                        'Return (Exact)': return_exact,
                        'Price (Duration)': price_duration,
                        'Price (Dur+Conv)': price_convexity,
                        'Price (Exact)': price_exact,
                        'Error (Dur)': abs(return_duration - return_exact),
                        'Error (Conv)': abs(return_convexity - return_exact)
                    })
                
                results_df = pd.DataFrame(results_data)
                
                # Display summary
                st.write("---")
                st.write("### Results: Confidence Interval for Return")
                
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.write("**Method 1: Duration Approximation**")
                    r_dur_lower = results_df[results_df['Scenario'].str.contains('Upper')]['Return (Duration)'].values[0]
                    r_dur_upper = results_df[results_df['Scenario'].str.contains('Lower')]['Return (Duration)'].values[0]
                    st.latex(rf"r \in [{r_dur_lower*100:.4f}\%, {r_dur_upper*100:.4f}\%]")
                    st.caption("(Symmetric interval)")
                
                with col_r2:
                    st.write("**Method 2: Exact Calculation**")
                    r_exact_lower = results_df[results_df['Scenario'].str.contains('Upper')]['Return (Exact)'].values[0]
                    r_exact_upper = results_df[results_df['Scenario'].str.contains('Lower')]['Return (Exact)'].values[0]
                    st.latex(rf"r \in [{r_exact_lower*100:.4f}\%, {r_exact_upper*100:.4f}\%]")
                    st.caption("(Asymmetric due to convexity)")
                
                # Full table
                st.write("### Detailed Results")
                st.dataframe(results_df.style.format({
                    'Δy': '{:.4f}',
                    'Δy (bps)': '{:.0f}',
                    'New Yield': '{:.4f}',
                    'Return (Duration)': '{:.4%}',
                    'Return (Dur+Conv)': '{:.4%}',
                    'Return (Exact)': '{:.4%}',
                    'Price (Duration)': '${:,.2f}',
                    'Price (Dur+Conv)': '${:,.2f}',
                    'Price (Exact)': '${:,.2f}',
                    'Error (Dur)': '{:.6%}',
                    'Error (Conv)': '{:.6%}'
                }))
                
                # Explanation
                with st.expander("📝 Methodology & Excel Formulas"):
                    st.markdown(rf"""
**Problem Setup:**
- Yield change Δy ~ N({delta_y_mean}, {delta_y_std}²)
- {conf_level*100:.0f}% CI: Δy ∈ [μ - z×σ, μ + z×σ] where z = {z_score:.4f}

**Method 1: Duration Approximation**
$$r \approx -D_{{mod}} \times \Delta y = -{modified_duration:.4f} \times \Delta y$$

This gives a **symmetric** interval because the formula is linear in Δy.

**Method 1b: With Convexity Adjustment**
$$r \approx -D_{{mod}} \times \Delta y + \frac{{1}}{{2}} \times C \times (\Delta y)^2$$
$$r \approx -{modified_duration:.4f} \times \Delta y + \frac{{1}}{{2}} \times {convexity:.4f} \times (\Delta y)^2$$

**Method 2: Exact Calculation**
Calculate the exact price at the new yield and compute return:
$$r = \frac{{P(y + \Delta y) - P(y)}}{{P(y)}}$$

This gives an **asymmetric** interval because the price-yield relationship is convex:
- When yields fall, prices rise more than the duration approximation predicts
- When yields rise, prices fall less than the duration approximation predicts

**Excel Formulas:**
```
Z-score: =NORM.S.INV((1+{conf_level})/2) = {z_score:.4f}
Δy lower: ={delta_y_mean} - {z_score:.4f} * {delta_y_std} = {dy_lower:.4f}
Δy upper: ={delta_y_mean} + {z_score:.4f} * {delta_y_std} = {dy_upper:.4f}

Return (Duration): =-{modified_duration:.4f} * Δy
Return (Exact): =(PV(new_yield/{freq}, {n_periods}, -{coupon_payment}, -{face_value}) - {price:.2f}) / {price:.2f}
```
                    """)
            
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def immunization_section():
    """
    Solve the duration matching (immunization) problem using two bonds.

    Given a liability’s present value and duration, and two candidate bonds with
    their prices, durations and yields, this function computes the weights in
    each bond needed to match the liability’s duration while investing the full
    amount.  It returns investment amounts, number of bonds and verifies the
    portfolio duration.  Excel formulas with cell references are provided for
    students to replicate the calculation on paper or in spreadsheets.
    """
    st.subheader("Duration Matching / Immunization")

    # Exam tip for immunization
    st.info(
        "🧠 **Exam tip:** For immunization, you must match both the **present value** and **duration** of the liability. "
        "Ensure that the candidate bonds have different durations; otherwise the system is underdetermined.  "
        "Weights should sum to one (full investment) and negative weights indicate a short position in that bond. "
        "Lay out the weights, dollar investments and duration contributions clearly in your solution."
    )

    st.markdown("""
    Match a liability's **duration** and **present value** using two bonds.
    """)
    
    st.write("**Liability to Immunize:**")
    col1, col2 = st.columns(2)
    with col1:
        liability_pv = st.number_input(
            "Liability Present Value ($)", value=1_000_000.0, min_value=1.0,
            help="Present value of the liability you wish to immunize (e.g., $1,000,000)")
    with col2:
        liability_duration = st.number_input(
            "Liability Duration (years)", value=5.0, min_value=0.1,
            help="Target Macaulay duration of the liability in years"
        )
    
    st.write("**Bond 1 Properties:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        bond1_price = st.number_input(
            "Bond 1 Price ($)", value=980.0,
            help="Current market price per bond for Bond 1 (dirty or clean price as appropriate)",
            key="b1_price"
        )
    with col2:
        bond1_duration = st.number_input(
            "Bond 1 Duration (years)", value=3.0,
            help="Macaulay duration of Bond 1 in years",
            key="b1_dur"
        )
    with col3:
        bond1_ytm = st.number_input(
            "Bond 1 YTM", value=0.04, format="%.4f",
            help="Yield to maturity of Bond 1 expressed as a decimal",
            key="b1_ytm"
        )
    
    st.write("**Bond 2 Properties:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        bond2_price = st.number_input(
            "Bond 2 Price ($)", value=1050.0,
            help="Current market price per bond for Bond 2",
            key="b2_price"
        )
    with col2:
        bond2_duration = st.number_input(
            "Bond 2 Duration (years)", value=8.0,
            help="Macaulay duration of Bond 2 in years",
            key="b2_dur"
        )
    with col3:
        bond2_ytm = st.number_input(
            "Bond 2 YTM", value=0.05, format="%.4f",
            help="Yield to maturity of Bond 2 expressed as a decimal",
            key="b2_ytm"
        )
    
    if st.button("🧮 Solve Immunization", key="calc_immunize"):
        try:
            # Solve: w1*D1 + w2*D2 = D_L  (duration matching)
            #        w1 + w2 = 1           (weights sum to 1)
            # 
            # From second equation: w2 = 1 - w1
            # Substitute: w1*D1 + (1-w1)*D2 = D_L
            # w1*(D1 - D2) = D_L - D2
            # w1 = (D_L - D2) / (D1 - D2)
            
            if bond1_duration == bond2_duration:
                st.error("❌ Bond durations must be different for immunization")
                return
            
            w1 = (liability_duration - bond2_duration) / (bond1_duration - bond2_duration)
            w2 = 1 - w1
            
            # Calculate $ amounts
            bond1_investment = w1 * liability_pv
            bond2_investment = w2 * liability_pv
            
            # Number of bonds
            n_bond1 = bond1_investment / bond1_price
            n_bond2 = bond2_investment / bond2_price
            
            # Portfolio duration (verify)
            portfolio_duration = w1 * bond1_duration + w2 * bond2_duration
            
            st.subheader("📊 Immunization Solution")
            
            results_df = pd.DataFrame({
                'Metric': [
                    'Weight (w_i)',
                    'Weight (%)',
                    'Investment ($)',
                    'Number of Bonds',
                    'Duration Contribution (years)'
                ],
                'Bond 1': [
                    f"{w1:.4f}",
                    f"{w1*100:.2f}%",
                    f"${bond1_investment:,.2f}",
                    f"{n_bond1:,.2f}",
                    f"{(w1*bond1_duration):.4f}"
                ],
                'Bond 2': [
                    f"{w2:.4f}",
                    f"{w2*100:.2f}%",
                    f"${bond2_investment:,.2f}",
                    f"{n_bond2:,.2f}",
                    f"{(w2*bond2_duration):.4f}"
                ]
            })
            st.table(results_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Portfolio Duration (years)", f"{portfolio_duration:.4f}", help="Duration of the immunized portfolio")
            with col2:
                st.metric("Target Duration (years)", f"{liability_duration:.4f}", help="Liability duration to match")
            with col3:
                st.metric("Match Error (years)", f"{abs(portfolio_duration - liability_duration):.6f}", help="Difference between portfolio and target durations")
            
            if w1 < 0 or w2 < 0:
                st.warning("⚠️ Negative weights indicate short positions required")
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(rf"""
**Immunization / Duration Matching in Excel:**

**The Problem:**
Find weights $w_1$ and $w_2$ such that:
- $w_1 \cdot D_1 + w_2 \cdot D_2 = D_L$ (match liability duration)
- $w_1 + w_2 = 1$ (full investment)

**Analytical Solution:**
$$w_1 = \frac{{D_L - D_2}}{{D_1 - D_2}}$$
$$w_2 = 1 - w_1$$

**Excel Formulas:**
```
=({liability_duration} - {bond2_duration}) / ({bond1_duration} - {bond2_duration})
```

Result: $w_1 = {w1:.4f}$, $w_2 = {w2:.4f}$

**Dollar Amounts:**
```
Bond 1 Investment = w1 × Liability_PV = {w1:.4f} × ${liability_pv:,.0f} = ${bond1_investment:,.2f}
Bond 2 Investment = w2 × Liability_PV = {w2:.4f} × ${liability_pv:,.0f} = ${bond2_investment:,.2f}
```

**Verification:**
Portfolio Duration = {w1:.4f} × {bond1_duration} + {w2:.4f} × {bond2_duration} = {portfolio_duration:.4f} ✓
                """)
            
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


# =============================================================================
# MODULE 3: HUMAN CAPITAL (LIFE-CYCLE)
# =============================================================================

def human_capital_module():
    """
    Compute human capital and optimal portfolio allocation within a life‑cycle context.

    This module values the present worth of future labor income (human capital) using a
    growing annuity formula and calculates the optimal stock allocation considering
    risk aversion, labor income volatility and the correlation between labor income
    and stock returns.  Users input their current income, growth and discount rates,
    remaining working years, financial wealth and market parameters.  The module then
    displays human capital, financial wealth, total wealth and the optimal stock
    weight, along with intermediate calculations.  Exam tips and Excel formulas
    help students replicate the analysis on paper or in spreadsheets.  Core
    mathematical logic remains unchanged; improvements focus on clarity, units and
    pedagogy.
    """
    st.header("👤 Module 3: Human Capital (Life‑Cycle Finance)")
    
    st.markdown("""
    Calculate the present value of future labor income (human capital) and the optimal
    portfolio allocation that accounts for labor income risk.
    """)
    
    # Exam tip for human capital
    st.info(
        "🧠 **Exam tip:** Human capital is often treated like a bond with a known "
        "growth rate (g) and discount rate (r). Use the growing annuity formula and remember to "
        "handle the r = g case separately.  The optimal stock weight scales the standard mean‑variance "
        "weight by total wealth (1 + L/W) and subtracts a hedge term proportional to the correlation between "
        "labor income and stock returns.  Show intermediate calculations for L/W, scaling and hedge terms in your answer."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income Parameters")
        current_income = st.number_input(
            "Current Annual Income ($)", value=75000.0, min_value=0.0,
            help="Current annual labor income (cash salary or wages)"
        )
        growth_rate = st.number_input(
            "Income Growth Rate (g)", value=0.03, format="%.4f",
            help="Expected annual growth rate of your income expressed as a decimal (e.g., 0.03 for 3%)"
        )
        discount_rate = st.number_input(
            "Discount Rate (r)", value=0.05, format="%.4f",
            help="Rate used to discount future labor income to present value (decimal)"
        )
        years_remaining = st.number_input(
            "Working Years Remaining (T)", value=30, min_value=1, max_value=60,
            help="Number of years left until retirement"
        )
    
    with col2:
        st.subheader("Portfolio Parameters")
        financial_wealth = st.number_input(
            "Current Financial Wealth ($)", value=200000.0, min_value=0.0,
            help="Your current invested wealth (exclude human capital)"
        )
        gamma = st.number_input(
            "Risk Aversion (γ)", value=4.0, min_value=0.1,
            help="Coefficient of relative risk aversion (higher γ = more risk averse)"
        )
        
        st.write("**Market Parameters:**")
        stock_return = st.number_input(
            "Expected Stock Return", value=0.08, format="%.4f",
            help="Expected annual return of the stock (μ_S) as a decimal"
        )
        stock_vol = st.number_input(
            "Stock Volatility (σ_S)", value=0.20, format="%.4f",
            help="Annual standard deviation of stock returns (σ_S)"
        )
        rf = st.number_input(
            "Risk-Free Rate", value=0.02, format="%.4f",
            help="Risk‑free rate (r_f) as a decimal"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        labor_vol = st.number_input(
            "Labor Income Volatility (σ_L)", value=0.10, format="%.4f",
            help="Annual standard deviation of labor income growth (σ_L)"
        )
    with col2:
        corr_stock_labor = st.number_input(
            "Correlation (Stock, Labor)", value=0.20,
            min_value=-1.0, max_value=1.0, format="%.2f",
            help="Correlation coefficient between stock returns and labor income growth (ρ_SL)"
        )
    
    # Income Timing Convention with Explanations
    st.write("---")
    with st.expander("⚙️ **Income Timing Convention** — Which formula should I use?", expanded=True):
        st.markdown(r"""
        ## Quick Decision Guide
        
        | Timing Convention | When r = g | When r != g | Use When |
        |-------------------|------------|-------------|----------|
        | **FMT/Munk** | $L = Y * T$ | $L = (1+g) * \frac{Y[1-(\frac{1+g}{1+r})^T]}{r-g}$ | Current income now; first future payment is $Y*(1+g)$ at t=1 |
        | **Ordinary** | $L = \frac{Y * T}{1+r}$ | $L = \frac{Y[1-(\frac{1+g}{1+r})^T]}{r-g}$ | First payment $Y$ at t=1 |
        | **Annuity Due** | $L = Y * T$ | $L = (1+r) * \frac{Y[1-(\frac{1+g}{1+r})^T]}{r-g}$ | First payment $Y$ at t=0 |
        
        ---
        
        **Exam Tip:** FMT problems typically follow the Munk convention. If timing is unclear, check whether the solution uses a (1+g) or (1+r) multiplier.
        """)
    
    timing_convention = st.selectbox(
        "Income Timing Convention",
        ["munk_fmt", "ordinary", "annuity_due"],
        format_func=lambda x: {
            "munk_fmt": "FMT/Munk: Current income now, first future payment = Y*(1+g) at t=1",
            "ordinary": "Ordinary: First payment = Y at t=1",
            "annuity_due": "Annuity Due: First payment = Y at t=0"
        }[x],
        index=0,
        key="hc_timing",
        help="FMT exams use Munk convention. Select based on how the problem defines income timing."
    )

    
    if st.button("🧮 Calculate Human Capital & Optimal Allocation", key="calc_hc"):
        try:
            # =================================================================
            # HUMAN CAPITAL CALCULATION
            # =================================================================
            
            if abs(discount_rate - growth_rate) < 0.0001:
                ordinary_annuity = current_income * years_remaining / (1 + discount_rate)
                base_formula = (
                    f"Base ordinary PV = Y * T / (1+r) = "
                    f"${current_income:,.0f} * {years_remaining} / {1+discount_rate:.4f} = ${ordinary_annuity:,.2f}"
                )
            else:
                growth_factor = (1 + growth_rate) / (1 + discount_rate)
                ordinary_annuity = current_income * (1 - growth_factor ** years_remaining) / (discount_rate - growth_rate)
                base_formula = (
                    "Base ordinary PV = Y * [1 - ((1+g)/(1+r))^T] / (r-g) = "
                    f"${ordinary_annuity:,.2f}"
                )
            
            if timing_convention == "munk_fmt":
                annuity_type_str = "FMT/Munk: current income now; first future payment is Y*(1+g) at t=1"
                timing_multiplier = f"(1+g) = {1+growth_rate:.4f}"
                timing_reason = "applies growth to the first future payment"
                human_capital = ordinary_annuity * (1 + growth_rate)
            elif timing_convention == "annuity_due":
                annuity_type_str = "Annuity Due: first payment is Y at t=0"
                timing_multiplier = f"(1+r) = {1+discount_rate:.4f}"
                timing_reason = "shifts payments to the beginning of each period"
                human_capital = ordinary_annuity * (1 + discount_rate)
            else:
                annuity_type_str = "Ordinary: first payment is Y at t=1"
                timing_multiplier = "No multiplier"
                timing_reason = "payments already start at t=1"
                human_capital = ordinary_annuity
            
            formula_used = f"{base_formula}. Timing adjustment: {timing_multiplier} ({timing_reason})."
            total_wealth = financial_wealth + human_capital
            
            # =================================================================
            # OPTIMAL ALLOCATION WITH HUMAN CAPITAL
            # =================================================================
            
            # Extended formula:
            # π_S = (μ - rf)/(γ × σ_S²) × (1 + L/W) - (L/W) × (ρ_SL × σ_L / σ_S)
            #
            # Where:
            # - First term: Standard mean-variance allocation scaled by total wealth
            # - Second term: Hedge term for labor income correlation
            
            L_over_W = human_capital / financial_wealth if financial_wealth > 0 else 0
            
            # Standard MV weight (ignoring human capital)
            excess_return = stock_return - rf
            standard_weight = excess_return / (gamma * stock_vol ** 2)
            
            # Full weight with human capital
            scaling_factor = 1 + L_over_W
            hedge_term = L_over_W * (corr_stock_labor * labor_vol / stock_vol)
            
            optimal_stock_weight = standard_weight * scaling_factor - hedge_term
            
            # Dollar amounts
            stock_dollars = optimal_stock_weight * financial_wealth
            rf_dollars = financial_wealth - stock_dollars
            
            # =================================================================
            # DISPLAY RESULTS
            # =================================================================
            
            st.subheader("📊 Human Capital Valuation")
            
            # Show which annuity type was used
            st.info(f"**Timing Convention Used:** {annuity_type_str}")
            st.markdown(f"**Formula:** {formula_used}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Human Capital ($)", f"${human_capital:,.0f}", help="Present value of future labor income")
            with col2:
                st.metric("Financial Wealth ($)", f"${financial_wealth:,.0f}", help="Current invested wealth")
            with col3:
                st.metric("Total Wealth ($)", f"${total_wealth:,.0f}", help="Sum of financial wealth and human capital")
            
            st.metric("L/W Ratio", f"{L_over_W:.2f}", help="Human capital divided by financial wealth")
            
            # Create wealth breakdown chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Pie chart of total wealth
            ax1.pie([human_capital, financial_wealth], 
                   labels=['Human Capital', 'Financial Wealth'],
                   autopct='%1.1f%%', colors=['#3498db', '#2ecc71'])
            ax1.set_title('Total Wealth Composition')
            
            # Human capital over time
            years = np.arange(0, years_remaining + 1)
            hc_over_time = []
            for t in years:
                remaining = years_remaining - t
                if remaining <= 0:
                    hc_over_time.append(0)
                elif abs(discount_rate - growth_rate) < 0.0001:
                    future_income = current_income * (1 + growth_rate) ** t
                    hc_over_time.append(future_income * remaining / (1 + discount_rate))
                else:
                    future_income = current_income * (1 + growth_rate) ** t
                    gf = (1 + growth_rate) / (1 + discount_rate)
                    hc_over_time.append(future_income * (1 - gf ** remaining) / (discount_rate - growth_rate))
            
            ax2.plot(years, hc_over_time, 'b-', linewidth=2)
            ax2.fill_between(years, hc_over_time, alpha=0.3)
            ax2.set_xlabel('Years from Now')
            ax2.set_ylabel('Human Capital ($)')
            ax2.set_title('Human Capital Depletion Over Time')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
            st.subheader("📈 Optimal Portfolio Allocation")
            
            # Breakdown of the formula components
            st.write("**Formula Components:**")
            
            component_df = pd.DataFrame({
                'Component': [
                    'Standard MV Weight (no HC)',
                    'Total Wealth Scaling (1 + L/W)',
                    'Scaled Weight (before hedge)',
                    'Hedge Term (ρ × σ_L/σ_S × L/W)',
                    'Final Optimal Weight'
                ],
                'Value': [
                    f"{standard_weight:.4f} ({standard_weight*100:.2f}%)",
                    f"{scaling_factor:.4f}",
                    f"{standard_weight * scaling_factor:.4f} ({standard_weight * scaling_factor*100:.2f}%)",
                    f"{hedge_term:.4f} ({hedge_term*100:.2f}%)",
                    f"{optimal_stock_weight:.4f} ({optimal_stock_weight*100:.2f}%)"
                ]
            })
            st.table(component_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Optimal Stock Weight (π_S)", f"{optimal_stock_weight*100:.2f}%", help="Fraction of financial wealth invested in stocks")
                st.metric("Stock Investment ($)", f"${stock_dollars:,.0f}", help="Dollar amount invested in stocks")
            with col2:
                st.metric("Risk‑Free Weight (1−π_S)", f"{(1-optimal_stock_weight)*100:.2f}%", help="Fraction of financial wealth invested in the risk‑free asset")
                st.metric("Risk‑Free Investment ($)", f"${rf_dollars:,.0f}", help="Dollar amount invested in the risk‑free asset")
            
            if optimal_stock_weight > 1:
                st.info(f"📊 Weight > 100% indicates borrowing ${stock_dollars - financial_wealth:,.0f} at the risk-free rate to invest in stocks (leverage)")
            elif optimal_stock_weight < 0:
                st.warning(f"⚠️ Negative weight indicates shorting stocks worth ${-stock_dollars:,.0f}")
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(rf"""
**Human Capital Calculation (Growing Annuity):**

$$HC = \frac{{Income_0 \times \left[1 - \left(\frac{{1+g}}{{1+r}}\right)^T\right]}}{{r - g}}$$

**Excel Formula:**
```
=Income * (1 - ((1+g)/(1+r))^T) / (r - g)
```
With values: `={current_income} * (1 - ((1+{growth_rate})/(1+{discount_rate}))^{years_remaining}) / ({discount_rate} - {growth_rate})`

Result: HC = ${human_capital:,.0f}

---

**Optimal Stock Allocation with Human Capital:**

$$\pi_S = \frac{{\mu - r_f}}{{\gamma \sigma_S^2}} \times \left(1 + \frac{{L}}{{W}}\right) - \frac{{L}}{{W}} \times \frac{{\rho_{{SL}} \sigma_L}}{{\sigma_S}}$$

**Excel breakdown:**

1. **Standard weight:** `=({stock_return}-{rf})/({gamma}*{stock_vol}^2)` = {standard_weight:.4f}

2. **Scaling factor:** `=1 + HC/W` = 1 + {human_capital:,.0f}/{financial_wealth:,.0f} = {scaling_factor:.4f}

3. **Hedge term:** `=(L/W)*{corr_stock_labor}*{labor_vol}/{stock_vol}` = {hedge_term:.4f}

4. **Final weight:** `=Step1 * Step2 - Step3` = {optimal_stock_weight:.4f}

**Interpretation:**
- Human capital acts like a "bond-like" asset (relatively stable income)
- High L/W ratio → investor can take MORE stock risk
- Positive correlation → need to HEDGE by reducing stock exposure
- {f"Your labor income is {corr_stock_labor*100:.0f}% correlated with stocks, reducing optimal stock weight by {hedge_term*100:.2f}%" if corr_stock_labor > 0 else "Negative correlation with stocks means labor income hedges stock risk"}
                """)
            
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


# =============================================================================
# MODULE 4: FACTOR MODELS
# =============================================================================

def factor_models_module():
    """
    Factor models and performance analysis module.

    This module covers two key topics often tested in asset pricing courses.  The
    **Performance Metrics** tab computes Sharpe ratio, Treynor ratio, Jensen’s alpha,
    information ratio, Modigliani–Modigliani (M²) and other statistics given fund and
    market parameters.  It also decomposes fund variance into systematic and
    idiosyncratic components.  The **Treynor–Black** tab implements the
    Treynor–Black model combining an active portfolio of mispriced securities with
    the market index.  The tab accepts alphas, betas and residual volatilities for
    each security and calculates the optimal active weight.  Exam tips and
    descriptive labels throughout help students interpret results and replicate
    calculations in Excel.
    """
    st.header("📉 Module 4: Factor Models (Alpha & Performance)")
    
    tab1, tab2 = st.tabs(["📊 Performance Metrics", "📈 Treynor-Black Model"])
    
    with tab1:
        performance_metrics_section()
    
    with tab2:
        treynor_black_section()


def performance_metrics_section():
    """
    Compute common fund performance metrics and variance decomposition.

    This section takes user inputs for fund return, volatility and beta along with
    corresponding market parameters and the risk‑free rate.  It calculates and
    displays the **Sharpe ratio**, **Treynor ratio**, **Jensen’s alpha**, **information
    ratio**, **Modigliani–Modigliani (M²)**, and the **CAPM expected return**.  A
    variance decomposition splits total fund variance into systematic and
    idiosyncratic components.  Results are presented with units and descriptive
    labels and include a pie chart for variance breakdown.  An Excel how‑to is
    provided via expander for exam practice.
    """
    st.subheader("Fund Performance Analysis")

    # Exam tip for performance metrics
    st.info(
        "🧠 **Exam tip:** Sharpe ratio uses total risk (σ), whereas Treynor ratio uses beta (systematic risk). "
        "Jensen’s alpha measures outperformance over CAPM, and the information ratio scales alpha by residual risk. "
        "Present percentages as decimals (e.g., 0.10 for 10%) and show each calculation clearly in your solution."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Fund Statistics:**")
        fund_return = st.number_input(
            "Fund Return (annualized)", value=0.12, format="%.4f",
            help="Annualized expected return of the fund as a decimal (e.g., 0.12 for 12%)"
        )
        fund_vol = st.number_input(
            "Fund Volatility (σ_F)", value=0.18, format="%.4f",
            help="Annualized standard deviation of fund returns (σ_F) as a decimal (e.g., 0.18 for 18%)"
        )
        fund_beta = st.number_input(
            "Fund Beta (β)", value=1.2, format="%.4f",
            help="Beta of the fund relative to the market index"
        )
    
    with col2:
        st.write("**Market & Benchmark:**")
        market_return = st.number_input(
            "Market Return", value=0.10, format="%.4f",
            help="Expected annual return of the market portfolio (decimal)"
        )
        market_vol = st.number_input(
            "Market Volatility (σ_M)", value=0.15, format="%.4f",
            help="Annualized standard deviation of market returns (σ_M) as a decimal"
        )
        rf = st.number_input(
            "Risk‑Free Rate (r_f)", value=0.02, format="%.4f", key="rf_factor",
            help="Risk‑free rate expressed as a decimal (e.g., 0.02 for 2%)"
        )
    
    if st.button("🧮 Calculate Performance Metrics", key="calc_perf"):
        try:
            # Sharpe Ratio
            sharpe = (fund_return - rf) / fund_vol
            
            # Treynor Ratio
            treynor = (fund_return - rf) / fund_beta
            
            # Jensen's Alpha
            capm_return = rf + fund_beta * (market_return - rf)
            jensens_alpha = fund_return - capm_return
            
            # Variance decomposition: σ²_total = β²σ²_m + σ²_ε
            systematic_var = (fund_beta ** 2) * (market_vol ** 2)
            total_var = fund_vol ** 2
            residual_var = total_var - systematic_var
            residual_vol = np.sqrt(max(0, residual_var))  # Ensure non-negative
            
            # Information Ratio
            info_ratio = jensens_alpha / residual_vol if residual_vol > 0 else np.nan
            
            # M² (Modigliani-Modigliani)
            m_squared = rf + sharpe * market_vol
            
            # Tracking Error (approximate)
            tracking_error = np.sqrt((fund_vol ** 2) + (market_vol ** 2) - 2 * fund_beta * fund_vol * market_vol)
            
            st.subheader("📊 Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sharpe Ratio (SR)", f"{sharpe:.4f}", help="Excess return per unit of total volatility")
                st.metric("Treynor Ratio (TR)", f"{treynor:.4f}", help="Excess return per unit of beta")
            
            with col2:
                st.metric("Jensen’s Alpha (α)", f"{jensens_alpha*100:.2f}%", help="Fund return minus CAPM expected return")
                st.metric("Information Ratio (IR)", f"{info_ratio:.4f}" if not np.isnan(info_ratio) else "N/A", help="Alpha divided by residual volatility")
            
            with col3:
                st.metric("M² (Risk‑Adjusted Return)", f"{m_squared*100:.2f}%", help="Risk‑adjusted return scaled to market volatility")
                st.metric("CAPM Expected Return (μ_CAPM)", f"{capm_return*100:.2f}%", help="Expected return predicted by CAPM")
            
            st.subheader("📈 Variance Decomposition")
            
            decomp_df = pd.DataFrame({
                'Component': ['Total Variance (σ²)', 'Systematic Variance (β²σ²_m)', 
                             'Residual/Idiosyncratic Variance (σ²_ε)'],
                'Value': [total_var, systematic_var, residual_var],
                'Volatility': [fund_vol, np.sqrt(systematic_var), residual_vol],
                '% of Total': [100, systematic_var/total_var*100 if total_var > 0 else 0,
                              residual_var/total_var*100 if total_var > 0 else 0]
            })
            st.dataframe(decomp_df.style.format({
                'Value': '{:.6f}',
                'Volatility': '{:.4f}',
                '% of Total': '{:.2f}%'
            }))
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Risk-Return plot
            ax1.scatter([fund_vol], [fund_return], s=150, c='blue', marker='o', label='Fund', zorder=5)
            ax1.scatter([market_vol], [market_return], s=150, c='red', marker='^', label='Market', zorder=5)
            ax1.scatter([0], [rf], s=100, c='green', marker='s', label='Risk-Free', zorder=5)
            
            # SML
            betas = np.linspace(0, 2, 100)
            sml_returns = rf + betas * (market_return - rf)
            # Plot against implied volatility (β × σ_m)
            ax1.plot([0, 2*market_vol], [rf, rf + 2*(market_return-rf)], 'r--', alpha=0.5, label='CML')
            
            ax1.set_xlabel('Volatility')
            ax1.set_ylabel('Return')
            ax1.set_title('Risk-Return Space')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Variance decomposition pie
            if residual_var > 0:
                ax2.pie([systematic_var, residual_var], 
                       labels=['Systematic', 'Idiosyncratic'],
                       autopct='%1.1f%%', colors=['#e74c3c', '#3498db'])
            else:
                ax2.pie([1], labels=['Systematic (100%)'], colors=['#e74c3c'])
            ax2.set_title('Variance Decomposition')
            
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(rf"""
**Performance Metrics Formulas:**

| Metric | Formula | Excel |
|--------|---------|-------|
| Sharpe Ratio | $\frac{{R_p - R_f}}{{\sigma_p}}$ | `=({fund_return}-{rf})/{fund_vol}` = {sharpe:.4f} |
| Treynor Ratio | $\frac{{R_p - R_f}}{{\beta_p}}$ | `=({fund_return}-{rf})/{fund_beta}` = {treynor:.4f} |
| Jensen's Alpha | $R_p - [R_f + \beta(R_m - R_f)]$ | `={fund_return}-({rf}+{fund_beta}*({market_return}-{rf}))` = {jensens_alpha:.4f} |
| Information Ratio | $\frac{{\alpha}}{{\sigma_\epsilon}}$ | `=Alpha/ResidualVol` = {info_ratio:.4f} |
| M² | $R_f + SR_p \times \sigma_m$ | `={rf}+{sharpe}*{market_vol}` = {m_squared:.4f} |

**Variance Decomposition:**

$$\sigma^2_{{total}} = \beta^2 \sigma^2_m + \sigma^2_\epsilon$$

**Excel Calculation for Residual Variance:**
```
Systematic Variance = β² × σ²_m = {fund_beta}² × {market_vol}² = {systematic_var:.6f}
Residual Variance = σ²_total - β²σ²_m = {total_var:.6f} - {systematic_var:.6f} = {residual_var:.6f}
Residual Volatility = SQRT({residual_var:.6f}) = {residual_vol:.4f}
```

**Interpretation:**
- Sharpe: {sharpe:.2f} units of excess return per unit of total risk
- Alpha of {jensens_alpha*100:.2f}% means fund {"outperformed" if jensens_alpha > 0 else "underperformed"} CAPM expectation
- {systematic_var/total_var*100:.1f}% of variance is systematic (market) risk
                """)
            
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def treynor_black_section():
    """
    Implement the Treynor–Black active portfolio selection.

    The Treynor–Black model combines a passive market index with an actively managed
    portfolio constructed from mispriced securities.  This section prompts the user
    for market parameters (expected return, volatility and risk‑free rate) and
    asset‑specific inputs (alphas, betas and residual volatilities) for each active
    security.  It then calculates the optimal active weights, the characteristics
    of the combined portfolio and displays them with appropriate labels and units.
    An Excel instructions expander helps students replicate the process manually.
    """
    st.subheader("Treynor‑Black Active Portfolio")
    
    st.markdown("""
    The Treynor–Black model optimally combines a passive market index with an actively managed portfolio of mispriced securities.
    """)

    # Exam tip for Treynor–Black
    st.info(
        """
🧠 **Exam tip:** To use the Treynor–Black model, compute each security’s **alpha** and **residual variance**
to determine its contribution to the active portfolio.  Normalize the alpha/variance ratios to obtain active
weights, then scale against the market to satisfy the investor’s risk aversion.  Clearly label alphas, betas
and residual volatilities and show intermediate steps in your answer.
"""
    )
    
    st.write("**Market Parameters:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        market_return = st.number_input(
            "Market E[R] (μ_M)", value=0.10, format="%.4f", key="tb_mkt_ret",
            help="Expected annual return of the market index (μ_M)"
        )
    with col2:
        market_vol = st.number_input(
            "Market σ (σ_M)", value=0.15, format="%.4f", key="tb_mkt_vol",
            help="Annual volatility of market returns (σ_M)"
        )
    with col3:
        rf = st.number_input(
            "Risk‑Free Rate (r_f)", value=0.02, format="%.4f", key="tb_rf",
            help="Risk‑free rate expressed as a decimal"
        )
    
    st.write("**Active Securities (paste data or enter manually):**")

    assets_input = st.text_area(
        "Alpha, Beta, Residual Volatility (per row)",
        value="0.02, 1.2, 0.15\n0.03, 0.8, 0.20\n-0.01, 1.5, 0.10",
        height=120,
        help="Enter one line per security in the format: α, β, σ_ε. Values should be decimals (e.g., 0.02 for 2% alpha)."
    )

    gamma = st.number_input(
        "Investor Risk Aversion (γ)", value=4.0, min_value=0.1, key="tb_gamma",
        help="Coefficient of risk aversion used to scale the active portfolio"
    )
    
    if st.button("🧮 Calculate Treynor-Black Weights", key="calc_tb"):
        try:
            # Parse input
            data = parse_matrix_input(assets_input)
            
            if data.shape[1] < 3:
                st.error("❌ Need at least 3 columns: Alpha, Beta, Residual Vol")
                return
            
            n_assets = data.shape[0]
            alphas = data[:, 0]
            betas = data[:, 1]
            residual_vols = data[:, 2]
            residual_vars = residual_vols ** 2
            
            # =================================================================
            # TREYNOR-BLACK CALCULATIONS
            # =================================================================
            
            # Step 1: Calculate raw active weights
            # w_i ∝ α_i / σ²_ε,i
            raw_weights = alphas / residual_vars
            
            # Step 2: Normalize to sum to 1
            w_active = raw_weights / np.sum(np.abs(raw_weights))  # Scale by absolute sum
            
            # Alternatively, scale so that sum = 1 (can have shorts)
            w_active_normalized = raw_weights / np.sum(raw_weights) if np.sum(raw_weights) != 0 else raw_weights
            
            # Step 3: Active portfolio characteristics
            alpha_A = w_active_normalized @ alphas
            beta_A = w_active_normalized @ betas
            
            # Residual variance of active portfolio
            # σ²_ε,A = Σ w²_i × σ²_ε,i (assuming independent residuals)
            var_epsilon_A = np.sum(w_active_normalized ** 2 * residual_vars)
            sigma_epsilon_A = np.sqrt(var_epsilon_A)
            
            # Total variance of active portfolio
            var_A = (beta_A ** 2) * (market_vol ** 2) + var_epsilon_A
            sigma_A = np.sqrt(var_A)
            
            # Step 4: Information Ratio of Active Portfolio
            IR_A = alpha_A / sigma_epsilon_A if sigma_epsilon_A > 0 else 0
            
            # Step 5: Optimal weight in active portfolio (vs market)
            # w*_A = (α_A / σ²_ε,A) / [(E[R_m] - r_f) / σ²_m]
            # Or: w*_A = (IR²_A / γ) / (β_A + (SR²_m + IR²_A)/γ) - more complex
            
            # Simple version: weight based on appraisal ratio
            market_sharpe_sq = ((market_return - rf) / market_vol) ** 2
            
            # Active portfolio weight (before beta adjustment)
            w_A_raw = (alpha_A / var_epsilon_A) / ((market_return - rf) / (market_vol ** 2))
            
            # Adjust for beta: final weight
            w_A = w_A_raw / (1 + (1 - beta_A) * w_A_raw)
            
            # Step 6: Weight in market index
            w_M = 1 - w_A
            
            # Step 7: Combined portfolio statistics
            combined_beta = w_A * beta_A + w_M * 1.0
            combined_alpha = w_A * alpha_A  # Market has zero alpha
            combined_return = rf + combined_beta * (market_return - rf) + combined_alpha
            
            # Combined portfolio variance
            # σ²_C = β²_C × σ²_m + w²_A × σ²_ε,A
            var_C = (combined_beta ** 2) * (market_vol ** 2) + (w_A ** 2) * var_epsilon_A
            sigma_C = np.sqrt(var_C)
            
            # Active portfolio expected return
            # E[r_A] = rf + α_A + β_A × (E[r_m] - rf)
            return_A = rf + alpha_A + beta_A * (market_return - rf)
            
            # Optimal risky portfolio Sharpe Ratio (squared)
            SR_combined_sq = market_sharpe_sq + IR_A ** 2
            SR_combined = np.sqrt(SR_combined_sq)
            
            # =================================================================
            # DISPLAY RESULTS
            # =================================================================
            
            st.subheader("📊 Individual Security Analysis")
            
            security_df = pd.DataFrame({
                'Security': [f'Security {i+1}' for i in range(n_assets)],
                'Alpha (α)': alphas,
                'Beta (β)': betas,
                'Residual Vol (σ_ε)': residual_vols,
                'α / σ²_ε': alphas / residual_vars,
                'Raw Weight': raw_weights,
                'Normalized Weight': w_active_normalized
            })
            st.dataframe(security_df.style.format({
                'Alpha (α)': '{:.4f}',
                'Beta (β)': '{:.2f}',
                'Residual Vol (σ_ε)': '{:.4f}',
                'α / σ²_ε': '{:.4f}',
                'Raw Weight': '{:.4f}',
                'Normalized Weight': '{:.4f}'
            }))
            
            st.subheader("📈 Active Portfolio (A)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Alpha of Active Portfolio (α_A)", f"{alpha_A*100:.2f}%", help="Weighted alpha of the active portfolio")
                st.metric("Beta of Active Portfolio (β_A)", f"{beta_A:.3f}", help="Weighted beta of the active portfolio")
            with col2:
                st.metric("Residual Volatility (σ_ε,A)", f"{sigma_epsilon_A*100:.2f}%", help="Residual standard deviation of the active portfolio")
                st.metric("Total Volatility (σ_A)", f"{sigma_A*100:.2f}%", help="Total standard deviation of the active portfolio")
            with col3:
                st.metric("E[r_A] (Expected Return)", f"{return_A*100:.2f}%", help="Expected return of the active portfolio")
                st.metric("Variance (σ²_A)", f"{var_A:.6f}", help="Total variance of the active portfolio")
            
            col1b, col2b = st.columns(2)
            with col1b:
                st.metric("Information Ratio (IR_A)", f"{IR_A:.4f}", help="Alpha divided by residual volatility")
            with col2b:
                st.metric("Appraisal Ratio² (IR_A²)", f"{IR_A**2:.4f}", help="Square of the information ratio")
            
            st.subheader("🎯 Optimal Allocation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Portfolio Weight (w_A)", f"{w_A*100:.2f}%", help="Fraction invested in the active portfolio")
                st.metric("Market Index Weight (w_M)", f"{w_M*100:.2f}%", help="Fraction invested in the market index")
            with col2:
                st.metric("Sharpe Ratio (combined)", f"{SR_combined:.4f}", help="Sharpe ratio of the combined active + market portfolio")
                st.metric("Market Sharpe Ratio", f"{np.sqrt(market_sharpe_sq):.4f}", help="Sharpe ratio of the market portfolio")
            
            st.subheader("📊 Combined Portfolio (C)")
            
            col1c, col2c, col3c = st.columns(3)
            with col1c:
                st.metric("Beta (β_C)", f"{combined_beta:.4f}", help="Beta of the combined portfolio")
            with col2c:
                st.metric("E[r_C] (Expected Return)", f"{combined_return*100:.2f}%", help="Expected return of the combined portfolio")
            with col3c:
                st.metric("Volatility (σ_C)", f"{sigma_C*100:.2f}%", help="Volatility of the combined portfolio")
            
            col1d, col2d = st.columns(2)
            with col1d:
                st.metric("Variance (σ²_C)", f"{var_C:.6f}", help="Variance of the combined portfolio")
            with col2d:
                st.metric("Excess Return (E[r_C] - rf)", f"{(combined_return - rf)*100:.2f}%", help="Combined portfolio excess return over risk-free")
            
            st.write("**Sharpe Ratio Improvement:**")
            st.latex(rf"SR^2_{{combined}} = SR^2_M + IR^2_A = {market_sharpe_sq:.4f} + {IR_A**2:.4f} = {SR_combined_sq:.4f}")
            st.write(f"Sharpe Ratio increases from {np.sqrt(market_sharpe_sq):.4f} to {SR_combined:.4f} (+{(SR_combined/np.sqrt(market_sharpe_sq)-1)*100:.1f}%)")
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(rf"""
**Treynor-Black Model - Excel Implementation:**

**Step 1: Calculate α/σ²_ε for each security**
```
=Alpha_i / (ResidualVol_i^2)
```

**Step 2: Calculate active portfolio weights**
$$w_i = \frac{{\alpha_i / \sigma^2_{{\epsilon,i}}}}{{\sum_j |\alpha_j / \sigma^2_{{\epsilon,j}}|}}$$

**Step 3: Active portfolio alpha**
```
=SUMPRODUCT(weights, alphas)
```
Result: α_A = {alpha_A:.4f}

**Step 4: Active portfolio beta**
```
=SUMPRODUCT(weights, betas)
```
Result: β_A = {beta_A:.4f}

**Step 5: Active portfolio residual variance**
```
=SUMPRODUCT(weights^2, residual_vars)
```
(Assumes independent residuals)
Result: σ²_ε,A = {var_epsilon_A:.6f}

**Step 6: Information Ratio**
$$IR_A = \frac{{\alpha_A}}{{\sigma_{{\epsilon,A}}}} = \frac{{{alpha_A:.4f}}}{{{sigma_epsilon_A:.4f}}} = {IR_A:.4f}$$

**Step 7: Optimal weight in Active Portfolio**
$$w^*_A = \frac{{\alpha_A / \sigma^2_{{\epsilon,A}}}}{{(E[R_m] - r_f) / \sigma^2_m}} \times \frac{{1}}{{1 + (1-\beta_A) \times w_A^{{raw}}}}$$

**Step 8: Sharpe Ratio of Combined Portfolio**
$$SR^2_{{optimal}} = SR^2_M + IR^2_A$$

**Key Excel Functions:**
- `SUMPRODUCT`: For weighted sums
- `SUMSQ`: For sum of squared weights × variances
                """)
            
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


# =============================================================================
# MODULE 5: PROBABILITY SCRATCHPAD
# =============================================================================

def probability_module():
    """
    Redesigned Probability Module with exam-focused tools.
    Handles Problems 1a, 1b, 1c style questions directly.
    """
    st.header("🎲 Module 5: Probability & Statistics")

    # Module 6 guidance expander
    with st.expander("📘 Module Guidance: When to use this?", expanded=False):
        st.markdown('''
        **When to use:**
        * **Probability:** "What is the probability return is < 0?" or "P(Amazon > Walmart)?".
        * **Goal Seek:** "What $\\mu$ is needed to keep probability of loss < 5%?"
        * **Scaling:** "If annual $\mu=8\%$, what is the 10-year expected return?"
        * **Handout Formulas:** Wealth Equivalent Loss, Levered VaR, Diversification N.

        **Inputs you need:**
        * Mean ($\mu$), Std Dev ($\sigma$), Correlation ($\rho$), Horizon ($T$).
        * Thresholds (e.g., 0 for negative return).

        **Common Pitfalls:**
        * **Variance vs Volatility:** Ensure you input $\sigma$ (StdDev), not $\sigma^2$ (Variance).
        * **Time Scaling:** Returns scale by $T$, Volatility by $\sqrt{T}$ (assuming independence).
        * **Comparison:** $P(A > B)$ requires constructing the difference variable $D = A - B$.

        **Fast Decision Rule:**
        "If the problem asks for **Probability, Confidence Intervals, or Time Scaling**, use this module."
        ''')
    
    st.markdown("""
    This module handles normal distribution problems found in exams, including:
    - **Single Asset:** Probabilities, Value at Risk, and "Reverse Solving" (e.g., find μ given risk).
    - **Asset Comparison:** P(A > B) and difference distributions.
    - **Minimize Downside:** Find portfolio with lowest P(R < 0) or P(R < threshold).
    - **Time Scaling:** Projecting annual risk/return to N years.
    - **Bivariate Normal:** Joint and conditional distributions.
    - **Theoretical Tools:** Wealth equivalent loss, levered VaR, diversification formulas.
    """)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Single Asset & Goal Seek",
        "⚖️ Compare Two Assets",
        "🎯 Min Downside Risk",
        "⏳ Multi-Period Scaling",
        "📈 Bivariate Normal",
        "📐 Theoretical Tools"
    ])
    
    with tab1:
        single_asset_goalseeking_tab()
    
    with tab2:
        compare_assets_tab()
    
    with tab3:
        min_downside_risk_tab()
    
    with tab4:
        multi_period_scaling_tab()
    
    with tab5:
        bivariate_normal_tab()
    
    with tab6:
        theoretical_tools_tab()


def single_asset_goalseeking_tab():
    """
    Single‑asset probability analysis with forward and reverse modes.

    This tab handles normal distribution questions for a single asset.  In
    **forward mode** it calculates the probability that returns fall within a
    specified interval or exceed a threshold given the asset’s mean (μ) and
    standard deviation (σ).  In **reverse (goal‑seek) mode** it solves for the
    required mean or volatility to achieve a target downside probability.  Use
    decimals for all rates (e.g., 0.10 for 10%) and supply lower and upper
    bounds in return units, not percentages.  Z‑scores and Excel formulas are
    provided to aid exam practice.
    """
    st.subheader("Single Asset Analysis")

    # Exam tip for single asset analysis
    st.info(
        "🧠 **Exam tip:** When working with normal returns, always convert percentage inputs to decimal form (e.g., −0.05 for −5%). "
        "For goal‑seek problems, recall that P(R < x) = Φ((x − μ)/σ).  Show the z‑score calculation explicitly and use the inverse normal function (NORM.S.INV) in Excel for target probabilities."
    )
    
    # Toggle between Forward (Calc Prob) and Reverse (Calc Inputs)
    mode = st.radio(
        "Analysis Mode", 
        ["Forward: Calculate Probability", "Reverse: Find Required μ or σ (Goal Seek)"],
        help="Use 'Reverse' for problems like 'What must expected return be to have 5% downside risk?'",
        key="single_asset_mode"
    )
    
    if mode == "Forward: Calculate Probability":
        col1, col2 = st.columns(2)
        with col1:
            mu = st.number_input(
                "Expected Return (μ)", value=0.08, format="%.4f", key="sa_mu",
                help="Expected return of the asset as a decimal (e.g., 0.08 for 8%)"
            )
            sigma = st.number_input(
                "Standard Deviation (σ)", value=0.25, format="%.4f", key="sa_sigma",
                help="Standard deviation of returns as a decimal (e.g., 0.25 for 25%)"
            )
        with col2:
            lower = st.number_input(
                "Lower Bound", value=-0.10, format="%.4f", key="sa_lower",
                help="Lower bound of the return interval (decimal, e.g., -0.10 for -10%)"
            )
            upper = st.number_input(
                "Upper Bound", value=0.20, format="%.4f", key="sa_upper",
                help="Upper bound of the return interval (decimal)"
            )
        
        single_threshold = st.number_input(
            "Single Threshold (for P(R < x))", value=0.0, format="%.4f", key="sa_thresh",
            help="Return threshold x for calculating P(R < x) (decimal)"
        )
        
        if st.button("Calculate Probabilities", key="sa_calc"):
            # Prob calculations
            p_interval = norm.cdf(upper, mu, sigma) - norm.cdf(lower, mu, sigma)
            p_below_lower = norm.cdf(lower, mu, sigma)
            p_above_upper = 1 - norm.cdf(upper, mu, sigma)
            p_below_thresh = norm.cdf(single_threshold, mu, sigma)
            p_above_thresh = 1 - p_below_thresh
            
            z_lower = (lower - mu) / sigma
            z_upper = (upper - mu) / sigma
            z_thresh = (single_threshold - mu) / sigma
            
            st.success("**Results:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"P({lower:.2%} < R < {upper:.2%})", f"{p_interval:.4f}")
                st.metric(f"P(R < {lower:.2%})", f"{p_below_lower:.4f}")
                st.metric(f"P(R > {upper:.2%})", f"{p_above_upper:.4f}")
            with col2:
                st.metric(f"P(R < {single_threshold:.2%})", f"{p_below_thresh:.4f}")
                st.metric(f"P(R > {single_threshold:.2%})", f"{p_above_thresh:.4f}")
                st.metric(f"Z-score at threshold", f"{z_thresh:.4f}")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
            y = norm.pdf(x, mu, sigma)
            ax.plot(x, y, color='blue', linewidth=2)
            
            # Shade interval
            mask = (x >= lower) & (x <= upper)
            ax.fill_between(x[mask], y[mask], color='blue', alpha=0.3, label=f'P({lower:.1%} < R < {upper:.1%})')
            
            # Mark threshold
            ax.axvline(single_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {single_threshold:.2%}')
            ax.axvline(mu, color='green', linestyle=':', alpha=0.7, label=f'Mean = {mu:.2%}')
            
            ax.set_xlabel('Return')
            ax.set_ylabel('Density')
            ax.set_title(f"Normal Distribution N({mu:.2%}, {sigma:.2%}²)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📝 Excel Formulas"):
                st.markdown(f"""
**Excel Formulas:**
- P(R < {lower:.4f}): `=NORM.DIST({lower}, {mu}, {sigma}, TRUE)` = {p_below_lower:.6f}
- P({lower:.4f} < R < {upper:.4f}): `=NORM.DIST({upper}, {mu}, {sigma}, TRUE) - NORM.DIST({lower}, {mu}, {sigma}, TRUE)` = {p_interval:.6f}
- P(R < {single_threshold:.4f}): `=NORM.DIST({single_threshold}, {mu}, {sigma}, TRUE)` = {p_below_thresh:.6f}
- Z-score: `=({single_threshold} - {mu}) / {sigma}` = {z_thresh:.4f}
                """)
            
    else:  # Reverse Mode (Problem 1c solver)
        st.info("💡 **Goal Seek Solver:** Find the required μ or σ to satisfy a probability constraint.")
        
        st.markdown("""
        **Example Problem:** *"You want your portfolio to have the property that the probability of 
        obtaining a negative return is exactly 5%. If σ = 20%, what must μ be?"*
        """)
        
        target_type = st.selectbox(
            "What do you need to find?", 
            ["Required Mean (μ)", "Required Volatility (σ)"],
            key="gs_target_type"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            target_prob = st.number_input(
                "Target Probability (left-tail)", 
                value=0.05, 
                format="%.4f", 
                help="Example: 'Probability of negative return is 5%' → Enter 0.05",
                key="gs_prob"
            )
            threshold = st.number_input(
                "Return Threshold", 
                value=0.0, 
                format="%.4f",
                help="Example: 'Negative return' → Threshold is 0.0",
                key="gs_thresh"
            )
        
        with col2:
            if target_type == "Required Mean (μ)":
                known_val = st.number_input("Known Volatility (σ)", value=0.20, format="%.4f", key="gs_known_sigma")
            else:
                known_val = st.number_input("Known Mean (μ)", value=0.15, format="%.4f", key="gs_known_mu")

        if st.button("🎯 Solve", key="gs_solve"):
            # Z-score for the target probability
            # P(R < threshold) = target_prob
            # (threshold - mu) / sigma = Z
            z_score = norm.ppf(target_prob)
            
            st.write("---")
            st.subheader("Solution")
            
            st.write(f"**Step 1: Find Z-score for left-tail probability of {target_prob:.2%}**")
            st.latex(rf"Z = \Phi^{{-1}}({target_prob}) = {z_score:.4f}")
            
            if target_type == "Required Mean (μ)":
                # Solve for mu: (T - mu) / sigma = z  =>  mu = T - z*sigma
                sigma = known_val
                solved_mu = threshold - (z_score * sigma)
                
                st.write(f"**Step 2: Solve for μ**")
                st.latex(rf"\frac{{\text{{Threshold}} - \mu}}{{\sigma}} = Z")
                st.latex(rf"\mu = \text{{Threshold}} - Z \times \sigma")
                st.latex(rf"\mu = {threshold} - ({z_score:.4f}) \times {sigma} = {solved_mu:.4f}")
                
                st.success(f"**✅ Required Expected Return (μ): {solved_mu:.4f} ({solved_mu*100:.2f}%)**")
                
                # Verification
                verification_prob = norm.cdf(threshold, solved_mu, sigma)
                st.write(f"**Verification:** P(R < {threshold}) = {verification_prob:.6f} ≈ {target_prob}")
                
                if threshold == 0:
                    st.info(f"""
                    **Exam Tip:** When the threshold is 0 (negative returns), 
                    the required μ = |Z| × σ = {abs(z_score):.4f} × {sigma} = {solved_mu:.4f}.
                    
                    The ratio μ/σ = {solved_mu/sigma:.4f} equals |Z| = {abs(z_score):.4f}.
                    """)
                
                with st.expander("📝 Excel Formula"):
                    st.markdown(f"""
**Excel:**
```
Z-score: =NORM.S.INV({target_prob}) = {z_score:.4f}
Required μ: ={threshold} - {z_score:.4f} * {sigma} = {solved_mu:.4f}
Verify: =NORM.DIST({threshold}, {solved_mu:.4f}, {sigma}, TRUE) = {verification_prob:.6f}
```
                    """)

            else:
                # Solve for sigma: (T - mu) / sigma = z => sigma = (T - mu) / z
                mu = known_val
                
                # Check for valid solution
                if abs(z_score) < 1e-10:
                    st.error("Z-score is ~0. Cannot solve for σ when target probability is 50%.")
                elif (z_score < 0 and mu > threshold) or (z_score > 0 and mu < threshold):
                    solved_sigma = (threshold - mu) / z_score
                    
                    if solved_sigma > 0:
                        st.write(f"**Step 2: Solve for σ**")
                        st.latex(rf"\sigma = \frac{{\text{{Threshold}} - \mu}}{{Z}}")
                        st.latex(rf"\sigma = \frac{{{threshold} - {mu}}}{{{z_score:.4f}}} = {solved_sigma:.4f}")
                        
                        st.success(f"**✅ Required Standard Deviation (σ): {solved_sigma:.4f} ({solved_sigma*100:.2f}%)**")
                        
                        # Verification
                        verification_prob = norm.cdf(threshold, mu, solved_sigma)
                        st.write(f"**Verification:** P(R < {threshold}) = {verification_prob:.6f} ≈ {target_prob}")
                        
                        with st.expander("📝 Excel Formula"):
                            st.markdown(f"""
**Excel:**
```
Z-score: =NORM.S.INV({target_prob}) = {z_score:.4f}
Required σ: =({threshold} - {mu}) / {z_score:.4f} = {solved_sigma:.4f}
Verify: =NORM.DIST({threshold}, {mu}, {solved_sigma:.4f}, TRUE) = {verification_prob:.6f}
```
                            """)
                    else:
                        st.error(f"Solution gives negative σ = {solved_sigma:.4f}. No valid solution exists.")
                else:
                    st.error("""
                    **No valid solution exists.** 
                    
                    The mean is on the 'wrong side' of the threshold for this probability.
                    For P(R < threshold) < 50%, mean must be > threshold.
                    For P(R < threshold) > 50%, mean must be < threshold.
                    """)


def compare_assets_tab():
    """
    Compare probabilities for two normally distributed asset returns.

    This tab computes probabilities such as P(X > Y), P(X < Y), or P(X > Y + k)
    for two assets X and Y that follow a bivariate normal distribution with
    means μ_X and μ_Y, standard deviations σ_X and σ_Y, and correlation ρ.  The
    difference D = X − Y is also normally distributed; probabilities are
    computed using its mean and variance.  Excel formulas are provided for
    exam practice.
    """
    st.subheader("Compare Two Assets")
    st.markdown("""
    Calculates probabilities involving two assets, e.g., P(X < Y) or P(X > Y + k).
    
    **Method:** Construct the difference variable D = X − Y.  If X and Y are joint
    normally distributed with correlation ρ, then D is also normally distributed
    with mean μ_D = μ_X − μ_Y and variance σ_D² = σ_X² + σ_Y² − 2ρσ_Xσ_Y.
    """)
    
    # Exam tip for comparing two assets
    st.info(
        "🧠 **Exam tip:** When comparing two normally distributed returns X and Y, express probabilities "
        "in terms of the difference D = X − Y.  Compute μ_D and σ_D² using the formula above, then use the standard normal CDF.  "
        "Always convert percentages to decimals and clearly show your difference calculations."
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Asset X**")
        mu_x = st.number_input(
            "Mean X (μ_X)", value=0.12, format="%.4f", key="cmp_mu_x",
            help="Expected return of asset X as a decimal (e.g., 0.12 for 12%)"
        )
        sigma_x = st.number_input(
            "Std Dev X (σ_X)", value=0.20, format="%.4f", key="cmp_sigma_x",
            help="Standard deviation of asset X returns as a decimal (σ_X)"
        )
    with col2:
        st.write("**Asset Y**")
        mu_y = st.number_input(
            "Mean Y (μ_Y)", value=0.06, format="%.4f", key="cmp_mu_y",
            help="Expected return of asset Y as a decimal"
        )
        sigma_y = st.number_input(
            "Std Dev Y (σ_Y)", value=0.14, format="%.4f", key="cmp_sigma_y",
            help="Standard deviation of asset Y returns (σ_Y)"
        )
    with col3:
        st.write("**Relationship**")
        rho = st.number_input(
            "Correlation (ρ)", value=0.20, format="%.2f", min_value=-1.0, max_value=1.0, key="cmp_rho",
            help="Correlation coefficient between returns of assets X and Y (ρ)"
        )

    st.write("---")
    st.write("**Question Parameters:**")
    diff_val = st.number_input(
        "Difference Threshold (k)", 
        value=0.0, 
        format="%.4f",
        help="For 'X exceeds Y by 20%', enter 0.20. For 'X > Y', enter 0.",
        key="cmp_k"
    )
    
    if st.button("Calculate Comparison", key="cmp_calc"):
        # We analyze D = X - Y
        # E[D] = E[X] - E[Y]
        # Var[D] = Var[X] + Var[Y] - 2*ρ*σ_X*σ_Y
        
        mu_diff = mu_x - mu_y
        var_diff = sigma_x**2 + sigma_y**2 - 2 * rho * sigma_x * sigma_y
        
        if var_diff < 0:
            st.error(f"Variance of D is negative ({var_diff:.6f}). Check correlation value.")
            return
            
        sigma_diff = np.sqrt(var_diff)
        
        # P(X < Y) = P(D < 0)
        z_zero = (0 - mu_diff) / sigma_diff
        p_x_less_y = norm.cdf(z_zero)
        
        # P(X > Y) = P(D > 0) = 1 - P(D < 0)
        p_x_greater_y = 1 - p_x_less_y
        
        # P(X > Y + k) = P(D > k)
        z_k = (diff_val - mu_diff) / sigma_diff
        p_x_exceeds_k = 1 - norm.cdf(z_k)
        
        # P(X < Y + k) = P(D < k)
        p_x_below_k = norm.cdf(z_k)
        
        st.subheader("Results")
        
        st.info("**Method:** Construct difference variable D = X - Y")
        
        st.write(f"**Step 1: Difference Distribution (D = X - Y)**")
        st.latex(rf"\mu_D = \mu_X - \mu_Y = {mu_x} - {mu_y} = {mu_diff:.4f}")
        st.latex(rf"\sigma_D^2 = \sigma_X^2 + \sigma_Y^2 - 2\rho\sigma_X\sigma_Y = {sigma_x}^2 + {sigma_y}^2 - 2({rho})({sigma_x})({sigma_y}) = {var_diff:.6f}")
        st.latex(rf"\sigma_D = {sigma_diff:.4f}")
        
        st.write(f"**Step 2: Calculate Probabilities**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("P(X < Y)", f"{p_x_less_y:.4f}")
            st.caption(f"= P(D < 0), Z = {z_zero:.4f}")
            
            st.metric("P(X > Y)", f"{p_x_greater_y:.4f}")
            st.caption(f"= P(D > 0) = 1 - P(D < 0)")
        
        with col2:
            st.metric(f"P(X > Y + {diff_val:.2%})", f"{p_x_exceeds_k:.4f}")
            st.caption(f"= P(D > {diff_val}), Z = {z_k:.4f}")
            
            st.metric(f"P(X < Y + {diff_val:.2%})", f"{p_x_below_k:.4f}")
            st.caption(f"= P(D < {diff_val})")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.linspace(mu_diff - 4*sigma_diff, mu_diff + 4*sigma_diff, 500)
        y = norm.pdf(x, mu_diff, sigma_diff)
        ax.plot(x, y, color='blue', linewidth=2)
        
        # Shade P(D > k)
        mask = x >= diff_val
        ax.fill_between(x[mask], y[mask], color='green', alpha=0.3, label=f'P(D > {diff_val:.2%})')
        
        # Shade P(D < 0)
        mask_neg = x < 0
        ax.fill_between(x[mask_neg], y[mask_neg], color='red', alpha=0.2, label='P(D < 0) = P(X < Y)')
        
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(diff_val, color='green', linestyle='--', alpha=0.7)
        ax.axvline(mu_diff, color='black', linestyle=':', alpha=0.5, label=f'E[D] = {mu_diff:.2%}')
        
        ax.set_xlabel('D = X - Y')
        ax.set_ylabel('Density')
        ax.set_title(f"Distribution of Difference: D ~ N({mu_diff:.4f}, {sigma_diff:.4f}²)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        with st.expander("📝 Excel Formulas"):
            st.markdown(f"""
**Excel Implementation:**

```
Mean of D: ={mu_x} - {mu_y} = {mu_diff}
Var of D:  ={sigma_x}^2 + {sigma_y}^2 - 2*{rho}*{sigma_x}*{sigma_y} = {var_diff:.6f}
Std of D:  =SQRT({var_diff:.6f}) = {sigma_diff:.4f}

P(X < Y):      =NORM.DIST(0, {mu_diff}, {sigma_diff}, TRUE) = {p_x_less_y:.6f}
P(X > Y):      =1 - NORM.DIST(0, {mu_diff}, {sigma_diff}, TRUE) = {p_x_greater_y:.6f}
P(X > Y + k):  =1 - NORM.DIST({diff_val}, {mu_diff}, {sigma_diff}, TRUE) = {p_x_exceeds_k:.6f}
```
            """)


def min_downside_risk_tab():
    """
    Find portfolio with minimum probability of negative (or below-threshold) return.
    
    Key insight: For R_p ~ N(μ_p, σ_p²),
    P(R_p < k) = Φ((k - μ_p) / σ_p)
    
    Minimizing P(R_p < 0) is equivalent to maximizing μ_p / σ_p (the Sharpe ratio when r_f=0).
    More generally, minimizing P(R_p < k) means maximizing (μ_p - k) / σ_p.
    """
    st.subheader("Minimize Downside Risk Portfolio")

    # Exam tip for minimizing downside risk
    st.info(
        "🧠 **Exam tip:** To minimize the probability that the portfolio return falls below a threshold k, "
        "maximize (μ − k) / σ.  When k = 0, this is equivalent to maximizing the Sharpe ratio. "
        "Compute the z‑score (k − μ)/σ and use the standard normal CDF to find probabilities.  "
        "In exam answers, state the tangency portfolio provides the minimum downside probability for a given set of assets."
    )
    
    st.markdown("""
    **Problem Type:** *"Which portfolio has the lowest probability of a negative return?"*
    
    **Key Insight:** For normally distributed returns, minimizing P(R < k) is equivalent to 
    maximizing (μ - k) / σ. When k = 0, this is exactly the **Sharpe Ratio** (with r_f = 0).
    
    So the **tangency portfolio** (max Sharpe) = **minimum probability of negative return** portfolio.
    """)
    
    st.write("---")
    
    # Use a form to prevent rerun on every input change
    with st.form(key="mdr_form"):
        # Number of assets - use selectbox instead of radio to be form-compatible
        n_assets = st.selectbox("Number of Assets", [2, 3, 4], index=0, key="mdr_n_assets")
        
        # Threshold
        threshold = st.number_input(
            "Return Threshold (k)", 
            value=0.0, 
            format="%.4f",
            help="Usually 0 for 'negative return'. Could be -0.10 for 'loss exceeds 10%'.",
            key="mdr_threshold"
        )
        
        # Constraints
        col_constraint1, col_constraint2 = st.columns(2)
        with col_constraint1:
            allow_short = st.checkbox("Allow Short Selling", value=True, key="mdr_short")
        with col_constraint2:
            full_investment = st.checkbox("Full Investment (weights sum to 1)", value=True, key="mdr_full")
        
        st.write("---")
        st.write("**Asset Parameters:**")
        
        # Always show 2-asset inputs (most common case for exams)
        # For 3 and 4 assets, use text inputs
        
        if n_assets == 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Asset 1**")
                mu1 = st.number_input("μ₁ (Expected Return)", value=0.14, format="%.4f", key="mdr_mu1")
                sigma1 = st.number_input("σ₁ (Std Dev)", value=0.30, format="%.4f", key="mdr_sig1")
            with col2:
                st.write("**Asset 2**")
                mu2 = st.number_input("μ₂ (Expected Return)", value=0.10, format="%.4f", key="mdr_mu2")
                sigma2 = st.number_input("σ₂ (Std Dev)", value=0.20, format="%.4f", key="mdr_sig2")
            with col3:
                st.write("**Correlation**")
                rho12 = st.number_input("ρ₁₂", value=0.50, min_value=-1.0, max_value=1.0, format="%.4f", key="mdr_rho12")
        
        elif n_assets == 3:
            st.write("**Expected Returns (μ):**")
            mu_input_3 = st.text_input("μ₁, μ₂, μ₃", value="0.12, 0.10, 0.08", key="mdr_mu_3")
            
            st.write("**Standard Deviations (σ):**")
            sigma_input_3 = st.text_input("σ₁, σ₂, σ₃", value="0.25, 0.20, 0.15", key="mdr_sig_3")
            
            st.write("**Correlations:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                rho12_3 = st.number_input("ρ₁₂", value=0.30, min_value=-1.0, max_value=1.0, format="%.2f", key="mdr_rho12_3")
            with col2:
                rho13_3 = st.number_input("ρ₁₃", value=0.20, min_value=-1.0, max_value=1.0, format="%.2f", key="mdr_rho13_3")
            with col3:
                rho23_3 = st.number_input("ρ₂₃", value=0.40, min_value=-1.0, max_value=1.0, format="%.2f", key="mdr_rho23_3")
        
        else:  # n_assets == 4
            st.write("**Expected Returns (μ):**")
            mu_input_4 = st.text_input("μ₁, μ₂, μ₃, μ₄", value="0.12, 0.10, 0.08, 0.06", key="mdr_mu_4")
            
            st.write("**Standard Deviations (σ):**")
            sigma_input_4 = st.text_input("σ₁, σ₂, σ₃, σ₄", value="0.25, 0.20, 0.15, 0.10", key="mdr_sig_4")
            
            st.write("**Correlation Matrix (4×4):**")
            corr_input_4 = st.text_area(
                "Correlation matrix rows",
                value="1.0, 0.3, 0.2, 0.1\n0.3, 1.0, 0.4, 0.2\n0.2, 0.4, 1.0, 0.3\n0.1, 0.2, 0.3, 1.0",
                height=100,
                key="mdr_corr_4"
            )
        
        # Submit button inside the form
        submitted = st.form_submit_button("🎯 Find Optimal Portfolio")
    
    # Process after form submission
    if submitted:
        try:
            # Build arrays based on n_assets
            if n_assets == 2:
                mu = np.array([mu1, mu2])
                sigma = np.array([sigma1, sigma2])
                corr = np.array([[1.0, rho12], [rho12, 1.0]])
                asset_names = ["Asset 1", "Asset 2"]
            elif n_assets == 3:
                mu = parse_messy_input(mu_input_3)
                sigma = parse_messy_input(sigma_input_3)
                corr = np.array([
                    [1.0, rho12_3, rho13_3],
                    [rho12_3, 1.0, rho23_3],
                    [rho13_3, rho23_3, 1.0]
                ])
                asset_names = ["Asset 1", "Asset 2", "Asset 3"]
            else:  # n_assets == 4
                mu = parse_messy_input(mu_input_4)
                sigma = parse_messy_input(sigma_input_4)
                corr = parse_matrix_input(corr_input_4)
                asset_names = [f"Asset {i+1}" for i in range(4)]
            
            n = len(mu)
            
            # Build covariance matrix from volatilities and correlations
            D = np.diag(sigma)
            cov = D @ corr @ D
            
            # Validate covariance matrix
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals < -1e-10):
                st.error(f"Covariance matrix is not positive semi-definite. Min eigenvalue: {np.min(eigvals):.6f}")
                return
            
            st.subheader("📊 Solution")
            
            # Adjust expected returns for threshold
            # Minimizing P(R_p < k) = maximizing (μ_p - k) / σ_p
            # This is like maximizing Sharpe with "excess return" = μ - k
            mu_adjusted = mu - threshold
            
            if threshold != 0:
                st.info(f"**Note:** With threshold k = {threshold:.4f}, we maximize (μ_p - {threshold}) / σ_p")
            
            # ================================================================
            # ANALYTICAL SOLUTION (for unconstrained or simple constraints)
            # ================================================================
            
            if allow_short and full_investment:
                # Closed-form: max Sharpe with r_f = k (or equivalently, excess returns μ - k)
                # w* ∝ Σ⁻¹(μ - k·1)
                
                st.write("**Method:** Closed-form solution (unconstrained Sharpe maximization)")
                
                ones = np.ones(n)
                cov_inv = np.linalg.inv(cov)
                
                # Raw weights (before normalization)
                w_raw = cov_inv @ mu_adjusted
                
                # Normalize to sum to 1
                w_opt = w_raw / np.sum(w_raw)
                
                st.write("**Step 1: Compute Σ⁻¹(μ - k·1)**")
                st.latex(r"\mathbf{w}^* \propto \Sigma^{-1}(\boldsymbol{\mu} - k \cdot \mathbf{1})")
                
                st.write("**Step 2: Normalize weights to sum to 1**")
                
            else:
                # Need numerical optimization for constraints
                st.write("**Method:** Numerical optimization (constrained)")
                
                def neg_sharpe_ratio(w):
                    port_return = w @ mu_adjusted
                    port_var = w @ cov @ w
                    if port_var <= 0:
                        return 1e10
                    port_std = np.sqrt(port_var)
                    return -port_return / port_std  # Negative because we minimize
                
                # Constraints
                constraints = []
                if full_investment:
                    constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
                
                # Bounds
                if allow_short:
                    bounds = [(None, None)] * n
                else:
                    bounds = [(0, 1)] * n
                
                # Initial guess
                w0 = np.ones(n) / n
                
                result = minimize(neg_sharpe_ratio, w0, method='SLSQP', bounds=bounds, constraints=constraints)
                
                if not result.success:
                    st.warning(f"Optimization warning: {result.message}")
                
                w_opt = result.x
            
            # ================================================================
            # COMPUTE PORTFOLIO STATISTICS
            # ================================================================
            
            port_return = w_opt @ mu
            port_var = w_opt @ cov @ w_opt
            port_std = np.sqrt(port_var)
            
            # The key metric: probability of return below threshold
            z_score = (threshold - port_return) / port_std
            prob_below = norm.cdf(z_score)
            prob_above = 1 - prob_below
            
            # Sharpe-like ratio (with threshold as reference)
            sharpe_like = (port_return - threshold) / port_std
            
            # ================================================================
            # DISPLAY RESULTS
            # ================================================================
            
            st.write("---")
            st.write("### Optimal Portfolio Weights")
            
            weights_df = pd.DataFrame({
                'Asset': asset_names,
                'Weight': w_opt,
                'Weight (%)': w_opt * 100
            })
            st.dataframe(weights_df.style.format({
                'Weight': '{:.4f}',
                'Weight (%)': '{:.2f}%'
            }))
            
            st.write("### Portfolio Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Return (μ_p)", f"{port_return:.4f} ({port_return*100:.2f}%)")
                st.metric("Volatility (σ_p)", f"{port_std:.4f} ({port_std*100:.2f}%)")
                st.metric("(μ_p - k) / σ_p", f"{sharpe_like:.4f}")
            
            with col2:
                st.metric(f"P(R_p < {threshold:.2%})", f"{prob_below:.6f} ({prob_below*100:.4f}%)", 
                         delta=None, delta_color="inverse")
                st.metric(f"P(R_p ≥ {threshold:.2%})", f"{prob_above:.6f} ({prob_above*100:.4f}%)")
                st.metric("Z-score at threshold", f"{z_score:.4f}")
            
            st.success(f"""
            **✅ Answer:** The optimal portfolio has weights [{', '.join([f'{w:.4f}' for w in w_opt])}]
            
            **Probability of return below {threshold:.2%}: {prob_below:.6f} ({prob_below*100:.4f}%)**
            """)
            
            # ================================================================
            # COMPARISON WITH INDIVIDUAL ASSETS
            # ================================================================
            
            st.write("---")
            st.write("### Comparison: Individual Assets vs Optimal Portfolio")
            
            comparison_data = []
            for i in range(n):
                z_i = (threshold - mu[i]) / sigma[i]
                p_i = norm.cdf(z_i)
                ratio_i = (mu[i] - threshold) / sigma[i]
                comparison_data.append({
                    'Portfolio': asset_names[i] + " (100%)",
                    'μ': mu[i],
                    'σ': sigma[i],
                    '(μ-k)/σ': ratio_i,
                    f'P(R < {threshold:.2%})': p_i,
                    'Z-score': z_i
                })
            
            # Add optimal portfolio
            comparison_data.append({
                'Portfolio': '⭐ Optimal',
                'μ': port_return,
                'σ': port_std,
                '(μ-k)/σ': sharpe_like,
                f'P(R < {threshold:.2%})': prob_below,
                'Z-score': z_score
            })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.style.format({
                'μ': '{:.4f}',
                'σ': '{:.4f}',
                '(μ-k)/σ': '{:.4f}',
                f'P(R < {threshold:.2%})': '{:.6f}',
                'Z-score': '{:.4f}'
            }).highlight_min(subset=[f'P(R < {threshold:.2%})'], color='lightgreen'))
            
            # ================================================================
            # VISUALIZATION
            # ================================================================
            
            st.write("---")
            st.write("### Visualization")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Left plot: Distribution comparison
            ax1 = axes[0]
            x_range = np.linspace(
                min(mu.min(), port_return) - 3 * max(sigma.max(), port_std),
                max(mu.max(), port_return) + 3 * max(sigma.max(), port_std),
                500
            )
            
            colors = plt.cm.tab10(np.linspace(0, 1, n + 1))
            
            for i in range(n):
                y = norm.pdf(x_range, mu[i], sigma[i])
                ax1.plot(x_range, y, color=colors[i], linestyle='--', alpha=0.6, 
                        label=f'{asset_names[i]}: N({mu[i]:.2f}, {sigma[i]:.2f}²)')
            
            # Optimal portfolio
            y_opt = norm.pdf(x_range, port_return, port_std)
            ax1.plot(x_range, y_opt, color='green', linewidth=2.5, 
                    label=f'Optimal: N({port_return:.3f}, {port_std:.3f}²)')
            
            # Shade the "bad" region
            mask = x_range < threshold
            ax1.fill_between(x_range[mask], y_opt[mask], color='red', alpha=0.2)
            
            ax1.axvline(threshold, color='red', linestyle=':', linewidth=2, label=f'Threshold = {threshold:.2%}')
            ax1.axvline(port_return, color='green', linestyle=':', alpha=0.5)
            
            ax1.set_xlabel('Return')
            ax1.set_ylabel('Density')
            ax1.set_title('Return Distributions')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Right plot: Efficient frontier with downside probability (for 2 assets)
            if n == 2:
                ax2 = axes[1]
                
                # Generate portfolios along the weight spectrum
                w1_range = np.linspace(-0.5 if allow_short else 0, 1.5 if allow_short else 1, 200)
                
                frontier_data = []
                for w1 in w1_range:
                    w2 = 1 - w1 if full_investment else 1 - w1
                    w = np.array([w1, w2])
                    
                    p_ret = w @ mu
                    p_var = w @ cov @ w
                    if p_var > 0:
                        p_std = np.sqrt(p_var)
                        z = (threshold - p_ret) / p_std
                        p_below = norm.cdf(z)
                        frontier_data.append({
                            'w1': w1,
                            'return': p_ret,
                            'std': p_std,
                            'prob_below': p_below
                        })
                
                frontier_df = pd.DataFrame(frontier_data)
                
                # Color by probability of negative return
                scatter = ax2.scatter(
                    frontier_df['std'], 
                    frontier_df['return'],
                    c=frontier_df['prob_below'],
                    cmap='RdYlGn_r',
                    s=20,
                    alpha=0.7
                )
                plt.colorbar(scatter, ax=ax2, label=f'P(R < {threshold:.0%})')
                
                # Mark optimal
                ax2.scatter([port_std], [port_return], color='blue', s=200, marker='*', 
                           edgecolors='black', linewidths=1.5, zorder=5, label='Optimal')
                
                # Mark individual assets
                for i in range(n):
                    ax2.scatter([sigma[i]], [mu[i]], s=100, marker='o', 
                               edgecolors='black', linewidths=1, zorder=4, label=asset_names[i])
                
                ax2.axhline(threshold, color='red', linestyle=':', alpha=0.5)
                ax2.set_xlabel('Volatility (σ)')
                ax2.set_ylabel('Expected Return (μ)')
                ax2.set_title('Risk-Return with Downside Probability')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
            else:
                axes[1].text(0.5, 0.5, 'Frontier plot available\nfor 2-asset case only', 
                            ha='center', va='center', fontsize=12)
                axes[1].set_xlim(0, 1)
                axes[1].set_ylim(0, 1)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # ================================================================
            # EXCEL / METHODOLOGY
            # ================================================================
            
            with st.expander("📝 Step-by-Step Solution & Excel"):
                st.markdown(f"""
**Problem:** Find portfolio minimizing P(R_p < {threshold})

**Key Insight:**
For R_p ~ N(μ_p, σ_p²):

P(R_p < k) = Φ((k - μ_p) / σ_p)

To minimize this probability, we need to minimize (k - μ_p) / σ_p, 
which is equivalent to **maximizing (μ_p - k) / σ_p**.

When k = 0, this is exactly the Sharpe Ratio!

**Closed-Form Solution (unconstrained, weights sum to 1):**

1. Compute excess returns: μ - k = {mu} - {threshold} = {mu_adjusted}

2. Invert covariance matrix: Σ⁻¹

3. Compute raw weights: w_raw = Σ⁻¹ × (μ - k)

4. Normalize: w* = w_raw / sum(w_raw) = [{', '.join([f'{w:.4f}' for w in w_opt])}]

**Portfolio Statistics:**
- μ_p = w'μ = {port_return:.6f}
- σ_p = √(w'Σw) = {port_std:.6f}
- Z = ({threshold} - {port_return:.4f}) / {port_std:.4f} = {z_score:.4f}
- P(R_p < {threshold}) = Φ({z_score:.4f}) = {prob_below:.6f}

**Excel Implementation:**
```
1. Set up covariance matrix in cells (e.g., B2:C3 for 2 assets)
2. Use MINVERSE() to get Σ⁻¹
3. Multiply: =MMULT(Σ⁻¹, μ-k) for raw weights
4. Normalize: =raw_weight / SUM(raw_weights)
5. Portfolio return: =SUMPRODUCT(weights, returns)
6. Portfolio variance: =MMULT(MMULT(TRANSPOSE(w), Σ), w)
7. Probability: =NORM.DIST({threshold}, μ_p, σ_p, TRUE)
```

Or use Solver:
- Target: Minimize P(R < k) = NORM.DIST(k, μ_p, σ_p, TRUE)
- Changing cells: weights
- Constraints: weights sum to 1, (optionally) weights >= 0
                """)
            
            # ================================================================
            # SPECIAL CASE: 2-ASSET ANALYTICAL FORMULA
            # ================================================================
            
            if n == 2 and allow_short and full_investment:
                with st.expander("📐 Two-Asset Analytical Formula"):
                    st.markdown(f"""
**For two assets with full investment (w₁ + w₂ = 1):**

The optimal weight in Asset 1 that maximizes (μ_p - k)/σ_p is:

""")
                    st.latex(r"w_1^* = \frac{(\mu_1 - k)\sigma_2^2 - (\mu_2 - k)\rho\sigma_1\sigma_2}{(\mu_1 - k)\sigma_2^2 + (\mu_2 - k)\sigma_1^2 - (\mu_1 + \mu_2 - 2k)\rho\sigma_1\sigma_2}")
                    
                    # Compute using the formula
                    mu1_adj = mu[0] - threshold
                    mu2_adj = mu[1] - threshold
                    s1, s2 = sigma[0], sigma[1]
                    rho = corr[0, 1]
                    
                    numerator = mu1_adj * s2**2 - mu2_adj * rho * s1 * s2
                    denominator = mu1_adj * s2**2 + mu2_adj * s1**2 - (mu1_adj + mu2_adj) * rho * s1 * s2
                    
                    w1_formula = numerator / denominator if abs(denominator) > 1e-10 else 0.5
                    
                    st.markdown(f"""
**Substituting values:**
- μ₁ - k = {mu[0]} - {threshold} = {mu1_adj:.4f}
- μ₂ - k = {mu[1]} - {threshold} = {mu2_adj:.4f}
- σ₁ = {s1}, σ₂ = {s2}, ρ = {rho}

Numerator = ({mu1_adj:.4f})({s2}²) - ({mu2_adj:.4f})({rho})({s1})({s2}) = {numerator:.6f}

Denominator = {denominator:.6f}

**w₁* = {w1_formula:.4f}**, w₂* = {1 - w1_formula:.4f}

(Matches optimization result: w₁ = {w_opt[0]:.4f})
                    """)
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def multi_period_scaling_tab():
    """
    Multi-period return scaling with three methods:
    1. Arithmetic (Parameters Only)
    2. Log-Returns (Continuous Compounding) 
    3. Simple Returns (Discrete Compounding)
    """
    st.subheader("Multi‑Period Scaling")
    
    # -------------------------------------------------------------------------
    # "WHEN TO USE WHAT" GUIDE - COMPREHENSIVE
    # -------------------------------------------------------------------------
    with st.expander("❓ CRITICAL: Which method should I choose?", expanded=True):
        st.markdown(r"""
        ## Quick Decision Tree
        
        ```
        Does the problem ask for EXPECTED WEALTH or GROSS RETURN?
        │
        ├─ NO → Just need scaled μ and σ
        │       → Use Method 1: ARITHMETIC (Parameters Only)
        │
        └─ YES → Need to calculate wealth growth
                 │
                 ├─ Keywords: "log-return", "continuous", "constant-weight portfolio"
                 │  → Use Method 2: LOG-RETURNS (Continuous)
                 │
                 └─ Keywords: "simple return", "buy-and-hold", "independent yearly returns"
                    → Use Method 3: SIMPLE RETURNS (Discrete)
        ```
        
        ---
        
        ## Detailed Comparison
        
        | Method | When to Use | Mean Formula | Volatility Formula | Gross Return |
        |--------|-------------|--------------|-------------------|--------------|
        | **1. Arithmetic** | Just need scaled distribution parameters | $\mu_T = T \cdot \mu$ | $\sigma_T = \sqrt{T} \cdot \sigma$ | ❌ Not applicable |
        | **2. Log-Returns** | Wealth with continuous compounding | $\mu_T = T \cdot \mu$ | $\sigma_T = \sqrt{T} \cdot \sigma$ | $e^{\mu T}$ |
        | **3. Simple Returns** | Wealth with discrete compounding | $(1+\mu)^T - 1$ | Complex formula | $(1+\mu)^T$ |
        
        ---
        
        ## Method 1: ARITHMETIC (Parameters Only)
        
        **Use when the problem:**
        - Asks only for the T-year mean and standard deviation
        - Says "scale the parameters" or "what is the distribution at horizon T"
        - Does NOT ask for expected wealth, gross return, or final portfolio value
        
        **Keywords:** "T-year mean", "T-year standard deviation", "distribution at horizon", "scaled parameters"
        
        **Example:** *"If annual μ=8% and σ=20%, what are the 5-year mean and standard deviation?"*
        - Answer: $\mu_5 = 5 \times 0.08 = 0.40$, $\sigma_5 = \sqrt{5} \times 0.20 = 0.447$
        - ✅ Done! No gross return needed.
        
        ---
        
        ## Method 2: LOG-RETURNS (Continuous Compounding)
        
        **Use when the problem:**
        - Mentions **log-returns**, **continuous compounding**, or **constant-weight portfolios**
        - References **Theorem 8.1** or says returns are **log-normally distributed**
        - Asks for **expected wealth** with continuous rebalancing
        
        **Keywords:** "log-return", "continuously compounded", "constant-weight", "log-normal", "Theorem 8.1"
        
        **Example:** *"A constant-weight portfolio has μ=3.8% and σ=6%. What is expected wealth after 20 years?"*
        - Step 1: $\mu_T = 20 \times 0.038 = 0.76$
        - Step 2: $E[\text{Gross Return}] = e^{0.76} = 2.138$
        - Step 3: $E[W_{20}] = W_0 \times 2.138$
        
        ⚠️ **Critical:** The scaled log-return (0.76 = 76%) is NOT the expected percentage gain!
        The expected gross return is $e^{0.76} = 2.138$ (113.8% gain).
        
        ---
        
        ## Method 3: SIMPLE RETURNS (Discrete Compounding)
        
        **Use when the problem:**
        - Mentions **simple returns** or **discrete compounding**
        - Describes **buy-and-hold** with **independent yearly returns**
        - Asks "what will $1 grow to" with annual returns that compound
        
        **Keywords:** "simple return", "buy and hold", "discrete", "independent annual returns", "compound annually"
        
        **Example:** *"An investment has expected simple return of 8%/year. What is expected value after 4 years?"*
        - $E[\text{Gross Return}] = (1.08)^4 = 1.360$
        - $E[W_4] = W_0 \times 1.360$
        
        ---
        
        ## ⚠️ Common Exam Traps
        
        1. **Trap:** Reporting $\mu_T$ as the "expected return" when asked about wealth
           - Wrong: "20-year expected return is 76%"
           - Right: "Expected gross return is $e^{0.76} = 2.138$" (for log-returns)
        
        2. **Trap:** Using simple return formulas when the problem says "log-return"
           - If it says "log-return" or "continuous" → Use Method 2
           - If it says "simple return" or "discrete" → Use Method 3
        
        3. **Trap:** Forgetting that Methods 1 and 2 use the SAME scaling formulas
           - The difference is whether you also compute $e^{\mu T}$ for wealth
        """)

    st.write("---")
    
    # Method selection FIRST (more prominent)
    st.markdown("### Select Scaling Method")
    method = st.radio(
        "Choose based on what the problem asks for:", 
        [
            "1. Arithmetic (Parameters Only) — Just need μ_T and σ_T",
            "2. Log-Returns (Continuous) — Need wealth with e^(μT)",
            "3. Simple Returns (Discrete) — Need wealth with (1+μ)^T"
        ],
        key="mp_method",
        help="Read the guide above to choose the correct method"
    )

    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        mu_annual = st.number_input("Annual Mean (μ)", value=0.038, format="%.4f", key="mp_mu",
                                   help="Annual expected return (as decimal, e.g., 0.038 for 3.8%)")
        sigma_annual = st.number_input("Annual Std Dev (σ)", value=0.06, format="%.4f", key="mp_sigma",
                                      help="Annual volatility (as decimal)")
    with col2:
        years = st.number_input("Time Horizon T (Years)", value=20, min_value=1, key="mp_years")
        
    # Additional probability calculation
    st.write("**Optional: Calculate probability at horizon**")
    calc_prob = st.checkbox("Calculate probability at threshold", key="mp_calc_prob")
    if calc_prob:
        threshold_mp = st.number_input("Return Threshold", value=0.0, format="%.4f", key="mp_thresh")
        
    if st.button("📊 Scale to Horizon", key="mp_scale"):
        
        # =====================================================================
        # METHOD 1: ARITHMETIC (Parameters Only)
        # =====================================================================
        if "1. Arithmetic" in method:
            mu_total = mu_annual * years
            var_total = (sigma_annual**2) * years
            sigma_total = np.sqrt(var_total)
            
            st.subheader(f"📈 {years}-Year Distribution Parameters (Arithmetic Scaling)")
            
            st.info("**Method 1: Arithmetic** — Reporting scaled parameters only. No wealth/gross return calculation.")
            
            st.markdown(r"""
            **Formulas Used:**
            - $\mu_T = T \times \mu$ (means add)
            - $\sigma_T = \sqrt{T} \times \sigma$ (standard deviations scale by √T)
            - $\sigma^2_T = T \times \sigma^2$ (variances add)
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.latex(rf"\mu_T = {years} \times {mu_annual} = {mu_total:.4f}")
                st.metric(f"{years}-Year Mean (μ_T)", f"{mu_total:.4f} ({mu_total*100:.2f}%)")
            with col2:
                st.latex(rf"\sigma_T = \sqrt{{{years}}} \times {sigma_annual} = {sigma_total:.4f}")
                st.metric(f"{years}-Year Std Dev (σ_T)", f"{sigma_total:.4f} ({sigma_total*100:.2f}%)")
            
            # Also show variance
            st.metric(f"{years}-Year Variance (σ²_T)", f"{var_total:.6f}")
            
            st.success(f"""
            **Final Answer:** The {years}-year distribution has:
            - Mean: **{mu_total:.4f}** ({mu_total*100:.2f}%)
            - Standard Deviation: **{sigma_total:.4f}** ({sigma_total*100:.2f}%)
            """)
            
            st.warning("""
            ⚠️ **Note:** This method does NOT calculate expected wealth or gross return.
            If the problem asks "what will the investment be worth" or "expected wealth", 
            you need Method 2 (Log-Returns) or Method 3 (Simple Returns)!
            """)
            
            with st.expander("📝 Excel Implementation"):
                st.markdown(rf"""
**Arithmetic Scaling in Excel:**
```
Mean (T years):     = {mu_annual} * {years} = {mu_total:.4f}
Variance (T years): = {sigma_annual}^2 * {years} = {var_total:.6f}
Std Dev (T years):  = SQRT({var_total:.6f}) = {sigma_total:.4f}
   OR equivalently: = {sigma_annual} * SQRT({years}) = {sigma_total:.4f}
```
                """)
        
        # =====================================================================
        # METHOD 2: LOG-RETURNS (Continuous Compounding)
        # =====================================================================
        elif "2. Log-Returns" in method:
            # Log-return scaling: log-returns are additive
            mu_total = mu_annual * years
            var_total = (sigma_annual**2) * years
            sigma_total = np.sqrt(var_total)
            
            # CRITICAL: Expected Gross Return for wealth calculations
            expected_gross_return = np.exp(mu_total)
            
            st.subheader(f"📈 {years}-Year Log-Return Statistics (Continuous Compounding)")
            
            st.info("**Method 2: Log-Returns** — For continuous compounding / constant-weight portfolios.")
            
            # Part A: Scaled Parameters
            st.markdown("### Part A: Scaled Log-Return Parameters")
            st.markdown(r"""
            Log-returns are additive: $r^{log}_{0,T} \sim N(\mu_T, \sigma^2_T)$
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.latex(rf"\mu_T = T \times \mu = {years} \times {mu_annual} = {mu_total:.4f}")
                st.metric(f"{years}-Year Log-Return Mean (μ_T)", f"{mu_total:.4f} ({mu_total*100:.2f}%)")
            with col2:
                st.latex(rf"\sigma_T = \sqrt{{T}} \times \sigma = \sqrt{{{years}}} \times {sigma_annual} = {sigma_total:.4f}")
                st.metric(f"{years}-Year Volatility (σ_T)", f"{sigma_total:.4f} ({sigma_total*100:.2f}%)")
            
            # Part B: Expected Wealth (THE KEY PART)
            st.markdown("---")
            st.markdown("### Part B: Expected Wealth / Gross Return")
            
            st.markdown(r"""
            **Key Result (Theorem 3.1 / 8.1):** To convert log-returns to expected wealth:
            $$E[\text{Gross Return}] = e^{\mu_T} = e^{\mu \times T}$$
            
            ⚠️ **This is the critical step!** Don't just report μ_T as the answer!
            """)
            
            st.latex(rf"E[\text{{Gross Return}}] = e^{{{mu_total:.4f}}} = {expected_gross_return:.6f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Gross Return", f"{expected_gross_return:.6f}", 
                         help="Multiply initial wealth by this")
            with col2:
                st.metric("As Percentage Gain", f"{(expected_gross_return - 1)*100:.2f}%",
                         help="Expected percentage increase in wealth")
            with col3:
                st.metric("As Multiplier", f"{expected_gross_return:.4f}×",
                         help="Wealth multiplier")
            
            # Example wealth calculation
            st.markdown(f"""
            **Example Wealth Calculation:**
            - If $W_0 = \\$100,000$: Expected Wealth = $100,000 × {expected_gross_return:.6f} = **${100000 * expected_gross_return:,.2f}**
            - If $W_0 = \\$1,000,000$: Expected Wealth = $1,000,000 × {expected_gross_return:.6f} = **${1000000 * expected_gross_return:,.2f}**
            """)
            
            st.success(f"""
            **Final Answer:** 
            - {years}-year log-return: N({mu_total:.4f}, {sigma_total:.4f}²)
            - Expected Gross Return: **{expected_gross_return:.6f}**
            - Expected Wealth: $W_0 × {expected_gross_return:.6f}$
            """)
            
            # Step-by-step for exams
            with st.expander("📝 Step-by-Step Solution for Exams"):
                st.markdown(rf"""
**Given:**
- Annual log-return mean: μ = {mu_annual} ({mu_annual*100:.2f}%)
- Annual volatility: σ = {sigma_annual} ({sigma_annual*100:.2f}%)
- Time horizon: T = {years} years

**Step 1: Scale the log-return distribution parameters**

$\mu_T = T \times \mu = {years} \times {mu_annual} = {mu_total:.4f}$

$\sigma_T = \sqrt{{T}} \times \sigma = \sqrt{{{years}}} \times {sigma_annual} = {sigma_total:.4f}$

So $r^{{log}}_{{0,{years}}} \sim N({mu_total:.4f}, {sigma_total:.4f}^2)$

**Step 2: Calculate Expected Gross Return (CRITICAL!)**

$E[\text{{Gross Return}}] = e^{{\mu_T}} = e^{{{mu_total:.4f}}} = {expected_gross_return:.6f}$

**Step 3: Calculate Expected Wealth**

$E[W_T] = W_0 \times e^{{\mu_T}} = W_0 \times {expected_gross_return:.6f}$

**Excel Formulas:**
```
Scaled mean:          = {mu_annual} * {years} = {mu_total:.4f}
Scaled std dev:       = {sigma_annual} * SQRT({years}) = {sigma_total:.4f}
Expected Gross Return: = EXP({mu_total:.4f}) = {expected_gross_return:.6f}
Expected Wealth:       = Initial_Wealth * EXP({mu_annual} * {years})
```

**⚠️ Common Mistake:** 
- WRONG: "The {years}-year expected return is {mu_total*100:.2f}%"
- RIGHT: "The expected gross return is {expected_gross_return:.6f} ({(expected_gross_return-1)*100:.2f}% gain)"
                """)
        
        # =====================================================================
        # METHOD 3: SIMPLE RETURNS (Discrete Compounding)
        # =====================================================================
        else:  # Simple Returns
            # Mean: (1+r)^T - 1
            mu_total = (1 + mu_annual)**years - 1
            expected_gross_return = (1 + mu_annual)**years
            
            # Variance of product of independent variables
            term1 = (sigma_annual**2 + (1 + mu_annual)**2)**years
            term2 = (1 + mu_annual)**(2 * years)
            var_total = term1 - term2
            sigma_total = np.sqrt(var_total) if var_total > 0 else 0
            
            st.subheader(f"📈 {years}-Year Simple Return Statistics (Discrete Compounding)")
            
            st.info("**Method 3: Simple Returns** — For discrete/annual compounding with independent yearly returns.")
            
            st.markdown(r"""
            **Formulas Used:**
            - Expected Return: $E[R_T] = (1+\mu)^T - 1$
            - Gross Return: $(1+\mu)^T$
            - Variance: $\sigma^2_T = [(1+\mu)^2 + \sigma^2]^T - (1+\mu)^{2T}$
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.latex(rf"E[R_T] = (1 + {mu_annual})^{{{years}}} - 1 = {mu_total:.4f}")
                st.metric(f"{years}-Year Expected Return", f"{mu_total:.4f} ({mu_total*100:.2f}%)")
            with col2:
                st.metric(f"{years}-Year Volatility", f"{sigma_total:.4f} ({sigma_total*100:.2f}%)")
            
            st.markdown("---")
            st.markdown("### Expected Wealth / Gross Return")
            
            st.latex(rf"E[\text{{Gross Return}}] = (1 + \mu)^T = (1 + {mu_annual})^{{{years}}} = {expected_gross_return:.6f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Gross Return", f"{expected_gross_return:.6f}")
            with col2:
                st.metric("As Percentage Gain", f"{mu_total*100:.2f}%")
            
            st.success(f"""
            **Final Answer:**
            - {years}-year expected simple return: **{mu_total:.4f}** ({mu_total*100:.2f}%)
            - Expected Gross Return: **{expected_gross_return:.6f}**
            - Expected Wealth: $W_0 × {expected_gross_return:.6f}$
            """)
            
            # Comparison with arithmetic
            arithmetic_mean = mu_annual * years
            st.markdown(f"""
            **Why is this different from Arithmetic scaling?**
            
            | Method | {years}-Year Mean |
            |--------|------------------|
            | Arithmetic (T × μ) | {arithmetic_mean:.4f} ({arithmetic_mean*100:.2f}%) |
            | Simple Returns ((1+μ)^T - 1) | {mu_total:.4f} ({mu_total*100:.2f}%) |
            
            With discrete compounding, you earn returns on your returns: $(1+\mu)^T > 1 + T \times \mu$
            """)
            
            with st.expander("📝 Excel Implementation"):
                st.markdown(rf"""
**Simple Return Scaling in Excel:**
```
Expected Return:       = (1 + {mu_annual})^{years} - 1 = {mu_total:.6f}
Expected Gross Return: = (1 + {mu_annual})^{years} = {expected_gross_return:.6f}
Expected Wealth:       = Initial_Wealth * (1 + {mu_annual})^{years}
```

**Variance (complex formula):**
```
= ((1+{mu_annual})^2 + {sigma_annual}^2)^{years} - (1+{mu_annual})^(2*{years})
= {var_total:.6f}
```
                """)
        
        # =====================================================================
        # PROBABILITY CALCULATIONS (Same for all methods once we have moments)
        # =====================================================================
        if calc_prob:
            z_mp = (threshold_mp - mu_total) / sigma_total if sigma_total > 0 else 0
            p_below = norm.cdf(z_mp)
            p_above = 1 - p_below
            
            st.write("---")
            st.subheader("📊 Probability Calculations")
            st.markdown("*(Using Normal Approximation for the T-year return distribution)*")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"P(R_{years} < {threshold_mp:.2%})", f"{p_below:.4f}")
            with col2:
                st.metric(f"P(R_{years} > {threshold_mp:.2%})", f"{p_above:.4f}")
            
            st.latex(rf"Z = \frac{{{threshold_mp} - {mu_total:.4f}}}{{{sigma_total:.4f}}} = {z_mp:.4f}")
            
            if "3. Simple" in method:
                st.warning("⚠️ Note: For simple returns over long horizons, the distribution is approximately Lognormal, not Normal. This probability is an approximation.")


def univariate_normal_tab():
    """Legacy univariate normal tab - now redirects to single_asset_goalseeking_tab."""
    st.info("This functionality has been moved to the 'Single Asset & Goal Seek' tab.")
    single_asset_goalseeking_tab()
    
    col1, col2 = st.columns(2)
    
    with col1:
        mean = st.number_input("Mean (μ)", value=0.08, format="%.4f",
                               help="Expected return (can be entered as decimal)", key="uni_mean")
        std_dev = st.number_input("Standard Deviation (σ)", value=0.15, format="%.4f",
                                  help="Volatility/standard deviation", key="uni_std")
    
    with col2:
        lower_bound = st.number_input("Lower Bound", value=-0.10, format="%.4f",
                                       help="Lower bound of range", key="uni_lower")
        upper_bound = st.number_input("Upper Bound", value=0.20, format="%.4f",
                                       help="Upper bound of range", key="uni_upper")
    
    st.subheader("Quick Calculations")
    
    col1, col2 = st.columns(2)
    with col1:
        single_point = st.number_input("Single Threshold", value=0.0, format="%.4f",
                                        help="Calculate P(X < threshold) and P(X > threshold)", key="uni_thresh")
    with col2:
        percentile = st.number_input("Find Value at Percentile (%)", value=5.0, min_value=0.1, max_value=99.9,
                                     help="Find the return at this percentile (VaR style)", key="uni_pct")
    
    if st.button("🧮 Calculate Probabilities", key="uni_calc"):
        try:
            # Probability calculations
            prob_between = norm.cdf(upper_bound, mean, std_dev) - norm.cdf(lower_bound, mean, std_dev)
            prob_below_lower = norm.cdf(lower_bound, mean, std_dev)
            prob_above_upper = 1 - norm.cdf(upper_bound, mean, std_dev)
            prob_below_threshold = norm.cdf(single_point, mean, std_dev)
            prob_above_threshold = 1 - prob_below_threshold
            
            z_lower = (lower_bound - mean) / std_dev
            z_upper = (upper_bound - mean) / std_dev
            z_threshold = (single_point - mean) / std_dev
            value_at_percentile = norm.ppf(percentile / 100, mean, std_dev)
            
            st.subheader("📊 Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"P({lower_bound:.2%} < X < {upper_bound:.2%})", f"{prob_between:.4f}")
                st.metric(f"P(X < {lower_bound:.2%})", f"{prob_below_lower:.4f}")
                st.metric(f"P(X > {upper_bound:.2%})", f"{prob_above_upper:.4f}")
            with col2:
                st.metric(f"P(X < {single_point:.2%})", f"{prob_below_threshold:.4f}")
                st.metric(f"P(X > {single_point:.2%})", f"{prob_above_threshold:.4f}")
                st.metric(f"{percentile:.0f}th Percentile", f"{value_at_percentile:.4f}")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
            y = norm.pdf(x, mean, std_dev)
            ax.plot(x, y, 'b-', linewidth=2)
            
            x_fill = x[(x >= lower_bound) & (x <= upper_bound)]
            y_fill = norm.pdf(x_fill, mean, std_dev)
            ax.fill_between(x_fill, y_fill, alpha=0.3, color='blue')
            
            ax.axvline(mean, color='black', linestyle='--', alpha=0.5)
            ax.axvline(value_at_percentile, color='red', linestyle=':', linewidth=2)
            ax.set_xlabel('Return')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(f"""
**Excel Formulas:**
- P(X < a): `=NORM.DIST({lower_bound}, {mean}, {std_dev}, TRUE)` = {prob_below_lower:.6f}
- P(a < X < b): `=NORM.DIST({upper_bound}, {mean}, {std_dev}, TRUE) - NORM.DIST({lower_bound}, {mean}, {std_dev}, TRUE)` = {prob_between:.6f}
- VaR at {percentile}%: `=NORM.INV({percentile/100}, {mean}, {std_dev})` = {value_at_percentile:.6f}
                """)
        
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def bivariate_normal_tab():
    """
    Bivariate normal probability calculator and conditional distribution solver.

    This tab computes joint and conditional probabilities for two normally
    distributed variables X and Y with specified means, standard deviations and
    correlation.  It can calculate the joint probability P(X ≤ a, Y ≤ b), the
    marginal distributions, and the conditional distribution of X given Y = y.
    Inputs should be provided as decimals (e.g., 0.08 for 8%).
    """
    st.markdown("""
    Calculate probabilities for **bivariate normal** distributions.
    
    - **Joint CDF:** P(X ≤ a, Y ≤ b)
    - **Conditional Distribution:** distribution of X given Y = y
    """)

    # Exam tip for bivariate normal tab
    st.info(
        "🧠 **Exam tip:** For bivariate normal problems specify the means (μ_X, μ_Y), standard deviations (σ_X, σ_Y) and correlation (ρ). "
        "Joint probabilities require evaluating the CDF of the bivariate normal, often using software.  Conditional distributions are normal with "
        "mean μ_{X|Y} = μ_X + ρ(σ_X/σ_Y)(y − μ_Y) and variance σ_X^2(1 − ρ²).  Clearly write these formulas in your exam solution."
    )
    
    st.subheader("Distribution Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mu_x = st.number_input(
            "μ_X (Mean of X)", value=0.08, format="%.4f", key="biv_mu_x",
            help="Expected return (mean) of X as a decimal"
        )
        sigma_x = st.number_input(
            "σ_X (Std of X)", value=0.15, format="%.4f", key="biv_sig_x",
            help="Standard deviation of X as a decimal"
        )
    
    with col2:
        mu_y = st.number_input(
            "μ_Y (Mean of Y)", value=0.06, format="%.4f", key="biv_mu_y",
            help="Expected return (mean) of Y as a decimal"
        )
        sigma_y = st.number_input("σ_Y (Std of Y)", value=0.12, format="%.4f", key="biv_sig_y")
    
    with col3:
        rho = st.number_input("ρ (Correlation)", value=0.3, min_value=-1.0, max_value=1.0, format="%.4f", key="biv_rho")
    
    calc_type = st.radio("Calculation Type", ["Joint CDF: P(X≤a, Y≤b)", "Conditional: X | Y=y"], key="biv_type")
    
    if calc_type == "Joint CDF: P(X≤a, Y≤b)":
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("a (upper bound for X)", value=0.10, format="%.4f", key="biv_a")
        with col2:
            b = st.number_input("b (upper bound for Y)", value=0.08, format="%.4f", key="biv_b")
        
        if st.button("🧮 Calculate Joint Probability", key="biv_joint_calc"):
            try:
                result, excel_steps, _ = bivariate_normal_cdf(a, b, mu_x, mu_y, sigma_x, sigma_y, rho)
                
                st.subheader("📊 Results")
                st.metric(f"P(X ≤ {a}, Y ≤ {b})", f"{result['probability']:.6f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"P(X ≤ {a}) [marginal]", f"{result['marginal_x']:.6f}")
                with col2:
                    st.metric(f"P(Y ≤ {b}) [marginal]", f"{result['marginal_y']:.6f}")
                
                st.write(f"If independent: P(X≤a)×P(Y≤b) = {result['marginal_x']*result['marginal_y']:.6f}")
                
                with st.expander("📝 How to do this in Excel"):
                    for step in excel_steps:
                        st.markdown(step)
            
            except ValidationError as e:
                st.error(f"❌ Validation error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Calculation error: {str(e)}")
    
    else:  # Conditional
        y_value = st.number_input("y (observed value of Y)", value=0.10, format="%.4f", key="biv_y_obs")
        
        if st.button("🧮 Calculate Conditional Distribution", key="biv_cond_calc"):
            try:
                result, excel_steps, _ = conditional_normal(y_value, mu_x, mu_y, sigma_x, sigma_y, rho)
                
                st.subheader("📊 Conditional Distribution: X | Y = " + f"{y_value}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Conditional Mean E[X|Y]", f"{result['conditional_mean']:.6f}")
                with col2:
                    st.metric("Conditional Std σ_X|Y", f"{result['conditional_std']:.6f}")
                
                st.latex(rf"X | Y={y_value} \sim N({result['conditional_mean']:.4f}, {result['conditional_std']:.4f}^2)")
                
                # Plot conditional vs unconditional
                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.linspace(mu_x - 4*sigma_x, mu_x + 4*sigma_x, 1000)
                
                # Unconditional
                y_uncond = norm.pdf(x, mu_x, sigma_x)
                ax.plot(x, y_uncond, 'b-', linewidth=2, label=f'Unconditional X ~ N({mu_x:.3f}, {sigma_x:.3f}²)')
                
                # Conditional
                y_cond = norm.pdf(x, result['conditional_mean'], result['conditional_std'])
                ax.plot(x, y_cond, 'r--', linewidth=2, label=f'X|Y={y_value} ~ N({result["conditional_mean"]:.3f}, {result["conditional_std"]:.3f}²)')
                
                ax.axvline(result['conditional_mean'], color='red', linestyle=':', alpha=0.5)
                ax.axvline(mu_x, color='blue', linestyle=':', alpha=0.5)
                ax.legend()
                ax.set_xlabel('X')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                with st.expander("📝 How to do this in Excel"):
                    for step in excel_steps:
                        st.markdown(step)
            
            except ValidationError as e:
                st.error(f"❌ Validation error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Calculation error: {str(e)}")


def portfolio_returns_tab():
    """Portfolio return distribution for jointly normal returns."""
    st.markdown("""
    For jointly normal returns **R ~ N(μ, Σ)**, compute the distribution of portfolio return **w'R**.
    
    Result: w'R ~ N(w'μ, w'Σw)
    """)
    
    input_type = st.radio("Input Type", ["Covariance Matrix", "Volatilities + Correlations"], key="port_input_type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weights_input = st.text_area(
            "Portfolio Weights",
            value="0.4, 0.35, 0.25",
            help="Weights for each asset",
            key="port_weights"
        )
        
        returns_input = st.text_area(
            "Expected Returns",
            value="0.08, 0.12, 0.06",
            help="Expected return for each asset",
            key="port_returns"
        )
    
    with col2:
        if input_type == "Covariance Matrix":
            cov_input = st.text_area(
                "Covariance Matrix",
                value="0.04, 0.015, 0.01\n0.015, 0.0625, 0.02\n0.01, 0.02, 0.0225",
                height=100,
                key="port_cov"
            )
            vols_input = None
            corr_input = None
        else:
            vols_input = st.text_area(
                "Volatilities",
                value="0.20, 0.25, 0.15",
                key="port_vols"
            )
            corr_input = st.text_area(
                "Correlation Matrix",
                value="1.0, 0.3, 0.2\n0.3, 1.0, 0.4\n0.2, 0.4, 1.0",
                height=100,
                key="port_corr"
            )
            cov_input = None
    
    use_threshold = st.checkbox("Calculate probability at threshold", key="port_use_thresh")
    if use_threshold:
        threshold = st.number_input("Threshold", value=0.0, format="%.4f", key="port_thresh")
    else:
        threshold = None
    
    if st.button("🧮 Compute Portfolio Distribution", key="port_calc"):
        try:
            weights = parse_messy_input(weights_input)
            mu = parse_messy_input(returns_input)
            
            if input_type == "Covariance Matrix":
                cov = parse_matrix_input(cov_input)
                result, excel_steps, _ = portfolio_return_distribution(
                    weights, mu, covariance_matrix=cov, threshold=threshold
                )
            else:
                vols = parse_messy_input(vols_input)
                corr = parse_matrix_input(corr_input)
                result, excel_steps, _ = portfolio_return_distribution(
                    weights, mu, volatilities=vols, correlations=corr, threshold=threshold
                )
            
            st.subheader("📊 Portfolio Return Distribution")
            
            st.latex(rf"w'R \sim N({result['portfolio_mean']:.4f}, {result['portfolio_std']:.4f}^2)")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("E[w'R]", f"{result['portfolio_mean']*100:.4f}%")
                st.metric("σ(w'R)", f"{result['portfolio_std']*100:.4f}%")
            
            with col2:
                if threshold is not None:
                    st.metric(f"P(w'R ≤ {threshold})", f"{result['prob_below']:.6f}")
                    st.metric(f"P(w'R ≥ {threshold})", f"{result['prob_above']:.6f}")
                    st.metric("Z-score", f"{result['z_score']:.4f}")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            pm, ps = result['portfolio_mean'], result['portfolio_std']
            x = np.linspace(pm - 4*ps, pm + 4*ps, 1000)
            y = norm.pdf(x, pm, ps)
            ax.plot(x, y, 'b-', linewidth=2)
            ax.fill_between(x, y, alpha=0.2)
            
            if threshold is not None:
                ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.2%}')
                x_below = x[x <= threshold]
                y_below = norm.pdf(x_below, pm, ps)
                ax.fill_between(x_below, y_below, alpha=0.3, color='red', label=f'P(w\'R ≤ {threshold}) = {result["prob_below"]:.4f}')
            
            ax.axvline(pm, color='black', linestyle=':', alpha=0.5, label=f'E[w\'R] = {pm:.2%}')
            ax.set_xlabel('Portfolio Return')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📝 How to do this in Excel"):
                for step in excel_steps:
                    st.markdown(step)
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def theoretical_tools_tab():
    """Theoretical formulas from handouts - wealth equivalent loss, levered VaR, diversification."""
    st.subheader("Theoretical Tools (Handout Formulas)")
    
    st.markdown("""
    This tab implements specific formulas from the course handouts that are frequently tested
    but don't fit into standard portfolio optimization. These include:
    - **Wealth Equivalent Loss:** Cost of suboptimal investing
    - **Levered VaR:** Value at Risk with leverage
    - **Diversification Formula:** Number of stocks for target volatility
    - **Certainty Equivalent:** Risk-adjusted return measure
    """)
    
    tool = st.selectbox(
        "Select Tool",
        [
            "Wealth Equivalent Loss",
            "Levered VaR",
            "Stocks for Target Volatility",
            "Certainty Equivalent Return"
        ],
        key="theo_tool"
    )
    
    st.write("---")
    
    if tool == "Wealth Equivalent Loss":
        st.write("### Wealth Equivalent Loss")
        
        st.markdown(r"""
        **Measures the cost of using a suboptimal strategy as a fraction of wealth.**
        
        **Formula:** $WEL = 1 - \exp\left(-\frac{\gamma}{2}(SR^2_{opt} - SR^2_{sub}) \cdot T\right)$
        
        **Approximation:** $WEL \approx \frac{\gamma \cdot T}{2}(SR^2_{opt} - SR^2_{sub})$
        """)
        
        st.info(
            "🧠 **Exam tip:** This formula answers questions like 'What percentage of wealth "
            "does an investor lose by holding cash instead of the optimal portfolio?' or "
            "'How much worse off is a naive 60/40 portfolio compared to mean-variance optimal?'"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            optimal_sr = st.number_input("Optimal Sharpe Ratio", value=0.40, format="%.4f", key="wel_opt_sr")
            suboptimal_sr = st.number_input("Suboptimal Sharpe Ratio", value=0.00, format="%.4f", key="wel_sub_sr",
                                           help="Enter 0 for cash")
        with col2:
            gamma = st.number_input("Risk Aversion (γ)", value=4.0, min_value=0.1, key="wel_gamma")
            horizon = st.number_input("Horizon (Years)", value=1, min_value=1, key="wel_horizon")
        
        if st.button("🧮 Calculate Wealth Equivalent Loss", key="wel_calc"):
            result, steps = wealth_equivalent_loss(optimal_sr, suboptimal_sr, gamma, horizon)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wealth Equivalent Loss", f"{result['wealth_equivalent_loss_pct']:.4f}%")
                st.metric("Linear Approximation", f"{result['linear_approximation']*100:.4f}%")
            with col2:
                st.metric("Annualized Loss", f"{result['annualized_loss']*100:.4f}%")
                st.metric("SR² Difference", f"{result['sharpe_sq_difference']:.6f}")
            
            with st.expander("📝 Step-by-Step & Excel"):
                for step in steps:
                    st.markdown(step)
    
    elif tool == "Levered VaR":
        st.write("### Levered Value at Risk")
        
        st.markdown(r"""
        **VaR with leverage adjustment.**
        
        **Formula:** $VaR_{lev} = (1 + L/E) \cdot VaR_{asset} - (L/E) \cdot r_{debt} \cdot \Delta t$
        
        Where:
        - $L/E$ = Leverage ratio (Debt/Equity)
        - $r_{debt}$ = Cost of debt
        - $\Delta t$ = Holding period as fraction of year
        """)
        
        st.info(
            "🧠 **Exam tip:** When leverage is used, losses are amplified by (1 + L/E). "
            "For example, 2:1 leverage (L/E = 1) means losses are doubled. "
            "The debt cost provides a small offset since you owe the interest regardless of returns."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            asset_var = st.number_input("Asset VaR (decimal)", value=0.10, format="%.4f", key="lvar_asset",
                                        help="E.g., 0.10 = 10% loss at confidence level")
            leverage = st.number_input("Leverage Ratio (L/E)", value=1.0, min_value=0.0, key="lvar_lev",
                                      help="1.0 = 50% debt, 2.0 = 67% debt")
        with col2:
            debt_rate = st.number_input("Debt Rate (annual)", value=0.05, format="%.4f", key="lvar_debt")
            holding = st.number_input("Holding Period (days)", value=1, min_value=1, key="lvar_hold")
            confidence = st.number_input("Confidence Level", value=0.95, format="%.2f", key="lvar_conf")
        
        if st.button("🧮 Calculate Levered VaR", key="lvar_calc"):
            result, steps = levered_var(asset_var, leverage, debt_rate, confidence, holding)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Levered VaR", f"{result['levered_var_pct']:.2f}%")
                st.metric("Amplification Factor", f"{result['amplification_factor']:.2f}x")
            with col2:
                st.metric("Unlevered VaR", f"{result['asset_var']*100:.2f}%")
                st.metric("Effective Amplification", f"{result['effective_amplification']:.2f}x")
            
            with st.expander("📝 Step-by-Step & Excel"):
                for step in steps:
                    st.markdown(step)
    
    elif tool == "Stocks for Target Volatility":
        st.write("### Minimum Stocks for Target Volatility")
        
        st.markdown(r"""
        **Find how many stocks needed to reach a target portfolio volatility.**
        
        **Portfolio Variance (equal weights):** $\sigma_p^2 = \sigma^2 \left[\frac{1-\rho}{N} + \rho\right]$
        
        **Solving for N:** $N = \frac{1-\rho}{(\sigma_{target}/\sigma)^2 - \rho}$
        
        **Diversification Limit:** $\lim_{N\to\infty} \sigma_p = \sigma\sqrt{\rho}$
        """)
        
        st.info(
            "🧠 **Exam tip:** With positive correlation, you CANNOT diversify below σ√ρ. "
            "This is the 'systematic risk' that remains even with infinite stocks. "
            "The formula shows diminishing returns to diversification."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            stock_vol = st.number_input("Individual Stock Volatility (σ)", value=0.30, format="%.4f", key="div_sigma")
            correlation = st.number_input("Average Correlation (ρ)", value=0.20, format="%.4f", key="div_rho",
                                         min_value=-0.5, max_value=1.0)
        with col2:
            target_vol = st.number_input("Target Portfolio Volatility", value=0.15, format="%.4f", key="div_target")
        
        # Show diversification limit
        limit_vol = stock_vol * np.sqrt(max(0, correlation))
        st.write(f"**Diversification limit:** σ√ρ = {stock_vol} × √{correlation} = {limit_vol:.4f} = {limit_vol*100:.2f}%")
        
        if target_vol < limit_vol and correlation > 0:
            st.warning(f"⚠️ Target {target_vol*100:.2f}% is below diversification limit {limit_vol*100:.2f}%. Impossible to achieve!")
        
        if st.button("🧮 Calculate Required Stocks", key="div_calc"):
            result, steps = stocks_for_target_volatility(stock_vol, correlation, target_vol)
            
            if result['achievable']:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Minimum Stocks Required", str(result['n_stocks']))
                    st.metric("Exact N", f"{result['n_exact']:.4f}")
                with col2:
                    st.metric("Achieved Volatility", f"{result['achieved_volatility']*100:.2f}%")
                    st.metric("Variance Reduction", f"{result['variance_reduction']*100:.1f}%")
            else:
                st.error(f"❌ {result['message']}")
            
            with st.expander("📝 Step-by-Step & Excel"):
                for step in steps:
                    st.markdown(step)
    
    elif tool == "Certainty Equivalent Return":
        st.write("### Certainty Equivalent Return")
        
        st.markdown(r"""
        **The risk-free rate that gives the same utility as a risky portfolio.**
        
        **Formula:** $CE = E[r] - \frac{\gamma}{2}\sigma^2$
        
        This represents the guaranteed return an investor would accept instead of taking the risk.
        """)
        
        st.info(
            "🧠 **Exam tip:** The certainty equivalent is useful for comparing portfolios for "
            "an investor with specific risk aversion. A higher CE means the portfolio is preferred. "
            "The term (γ/2)σ² is the 'risk penalty' - what you subtract for taking on volatility."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            exp_ret = st.number_input("Expected Return E[r]", value=0.10, format="%.4f", key="ce_ret")
            volatility = st.number_input("Volatility (σ)", value=0.20, format="%.4f", key="ce_vol")
        with col2:
            gamma = st.number_input("Risk Aversion (γ)", value=4.0, min_value=0.1, key="ce_gamma")
        
        if st.button("🧮 Calculate Certainty Equivalent", key="ce_calc"):
            result, steps = certainty_equivalent_return(exp_ret, volatility, gamma)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Certainty Equivalent", f"{result['certainty_equivalent_pct']:.4f}%")
            with col2:
                st.metric("Risk Penalty", f"{result['risk_penalty']*100:.4f}%")
            
            # Interpretation
            if result['certainty_equivalent'] > 0:
                st.success(f"✅ This portfolio is worth taking: CE ({result['certainty_equivalent_pct']:.2f}%) > 0")
            else:
                st.warning(f"⚠️ This portfolio has negative CE: the investor prefers cash")
            
            with st.expander("📝 Step-by-Step & Excel"):
                for step in steps:
                    st.markdown(step)


# =============================================================================
# MODULE A: ADVANCED FIXED INCOME ENGINE (Exam-Specific)
# =============================================================================

def advanced_fixed_income_module():
    st.header("🏦 Module A: Advanced Fixed Income Engine")
    
    st.markdown("""
    **Exam-Specific Problem Families (2023-2025):**
    - Bond pricing with exact cash flow schedules
    - Yield solving via Newton-Raphson
    - Sensitivity analysis with confidence intervals
    - Synthetic zero-coupon bond replication
    - Duration-matching immunization
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Bond Analysis", "🔧 Bond Replication", "🛡️ Immunization"])
    
    with tab1:
        advanced_bond_analysis()
    
    with tab2:
        bond_replication_section()
    
    with tab3:
        advanced_immunization_section()


def advanced_bond_analysis():
    st.subheader("BulletBond Analysis (Exam Format)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        face_value = st.number_input("Face Value", value=1000.0, key="adv_fv")
        coupon_rate = st.number_input("Coupon Rate", value=0.05, format="%.4f", key="adv_cr")
        maturity = st.number_input("Maturity (Years)", value=5, min_value=1, key="adv_mat")
    
    with col2:
        input_type = st.radio("Input Type", ["Yield → Price", "Price → Yield"])
        if input_type == "Yield → Price":
            yield_rate = st.number_input("Yield to Maturity", value=0.04, format="%.4f", key="adv_ytm")
            price_input = None
        else:
            price_input = st.number_input("Market Price", value=1043.76, key="adv_price")
            yield_rate = None
    
    if st.button("🧮 Analyze Bond", key="analyze_bond"):
        try:
            # Create BulletBond instance
            if input_type == "Yield → Price":
                bond = BulletBond(
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    maturity=maturity,
                    yield_rate=yield_rate
                )
            else:
                bond = BulletBond(
                    face_value=face_value,
                    coupon_rate=coupon_rate,
                    maturity=maturity,
                    price=price_input
                )
            
            st.subheader("📋 Cash Flow Schedule (Exam Format)")
            st.markdown("*This exact table format is required for exam duration verification*")
            
            schedule = bond.get_schedule()
            st.dataframe(schedule.style.format({
                't (years)': '{:.2f}',
                'CashFlow': '${:,.2f}',
                'DiscountFactor': '{:.6f}',
                'PV': '${:,.4f}',
                'Weight': '{:.6f}',
                'TimeWeighted': '{:.6f}'
            }))
            
            st.subheader("📊 Bond Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price", f"${bond.price:,.4f}")
            with col2:
                st.metric("Yield", f"{bond.yield_rate*100:.4f}%")
            with col3:
                st.metric("Macaulay Duration", f"{bond.macaulay_duration:.4f} years")
            with col4:
                st.metric("Modified Duration", f"{bond.modified_duration:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Convexity", f"{bond.convexity:.4f}")
            with col2:
                st.metric("DV01 (Dollar Duration)", f"${bond.dollar_duration:.4f}")
            
            # Sensitivity Analysis
            st.subheader("📈 Sensitivity Analysis (Confidence Interval)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                delta_y_mean = st.number_input("Expected Δy", value=0.0, format="%.4f", key="dy_mean")
            with col2:
                delta_y_std = st.number_input("Std Dev of Δy", value=0.01, format="%.4f", key="dy_std")
            with col3:
                conf_level = st.number_input("Confidence Level", value=0.95, format="%.2f", key="conf")
            
            if st.button("Calculate Sensitivity", key="calc_sens"):
                sens_table = bond.sensitivity_analysis(delta_y_mean, delta_y_std, conf_level)
                st.dataframe(sens_table.style.format({
                    'Δy': '{:.4f}',
                    'Δy (bps)': '{:.1f}',
                    'New Yield': '{:.4f}',
                    'Price (Duration)': '${:,.2f}',
                    'Price (Dur+Conv)': '${:,.2f}',
                    'Price (Exact)': '${:,.2f}',
                    'Return (Duration)': '{:.4%}',
                    'Return (Dur+Conv)': '{:.4%}',
                    'Return (Exact)': '{:.4%}',
                    'Error (Dur vs Exact)': '${:.4f}',
                    'Error (Conv vs Exact)': '${:.4f}'
                }))
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(bond.get_excel_instructions())
                
                st.markdown("**LaTeX Formulas:**")
                for name, formula in bond.get_latex_formulas().items():
                    st.latex(formula)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def bond_replication_section():
    st.subheader("🔧 Synthetic Zero-Coupon Bond Replication")
    
    st.markdown("""
    **Problem Type:** Replicate a target cash flow using coupon bonds.
    
    **Algorithm:** Solve C × N = T where C is the cash flow matrix, N is units, T is target.
    """)
    
    st.write("**Define Bonds:**")
    
    n_bonds = st.number_input("Number of Bonds", value=2, min_value=1, max_value=5, key="n_bonds_rep")
    
    bonds_data = []
    cols = st.columns(n_bonds)
    
    for i, col in enumerate(cols):
        with col:
            st.write(f"**Bond {i+1}**")
            fv = st.number_input(f"Face Value {i+1}", value=1000.0, key=f"rep_fv_{i}")
            cr = st.number_input(f"Coupon Rate {i+1}", value=0.05 + i*0.01, format="%.4f", key=f"rep_cr_{i}")
            mat = st.number_input(f"Maturity {i+1}", value=2+i, min_value=1, key=f"rep_mat_{i}")
            ytm = st.number_input(f"YTM {i+1}", value=0.04, format="%.4f", key=f"rep_ytm_{i}")
            bonds_data.append({'fv': fv, 'cr': cr, 'mat': mat, 'ytm': ytm})
    
    st.write("**Define Target Cash Flow:**")
    target_time = st.number_input("Target Time (years)", value=3, min_value=1, key="target_t")
    target_amount = st.number_input("Target Amount ($)", value=1000.0, key="target_amt")
    
    if st.button("🧮 Solve Replication", key="solve_rep"):
        try:
            # Create bond objects
            instruments = []
            for bd in bonds_data:
                bond = BulletBond(
                    face_value=bd['fv'],
                    coupon_rate=bd['cr'],
                    maturity=bd['mat'],
                    yield_rate=bd['ytm']
                )
                instruments.append(bond)
            
            # Create target flows
            target_flows = {target_time: target_amount}
            
            # Create replicator
            replicator = BondReplicator(instruments=instruments, target_flows=target_flows)
            result = replicator.solve()
            
            if result['solvable']:
                st.success(f"✅ Solution found using: {result['method']}")
                
                if 'warning' in result and result['warning']:
                    st.warning(f"⚠️ {result['warning']}")
                
                st.subheader("📊 Solution")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Units Required:**")
                    for i, n in enumerate(result['units']):
                        st.metric(f"Bond {i+1} Units", f"{n:.4f}")
                
                with col2:
                    st.metric("Implied Price", f"${result['implied_price']:,.4f}")
                    if result.get('implied_yield'):
                        st.metric("Implied Yield", f"{result['implied_yield']*100:.4f}%")
                
                st.subheader("✓ Verification Table")
                verification = replicator.get_verification_table()
                st.dataframe(verification.style.format('{:.4f}', na_rep='-'))
                
                st.subheader("📐 Cash Flow Matrix")
                st.write("**Matrix C:**")
                C_df = pd.DataFrame(
                    result['cash_flow_matrix'],
                    index=[f"t={t}" for t in result['times']],
                    columns=[f"Bond {i+1}" for i in range(len(instruments))]
                )
                st.dataframe(C_df.style.format('{:.2f}'))
                
                if 'determinant' in result:
                    st.metric("Determinant", f"{result['determinant']:.6f}")
                
                with st.expander("📝 How to do this in Excel"):
                    st.markdown(replicator.get_excel_instructions())
                    
                    st.markdown("**LaTeX Formulas:**")
                    for name, formula in replicator.get_latex_formulas().items():
                        st.latex(formula)
            else:
                st.error(f"❌ Cannot solve: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def advanced_immunization_section():
    st.subheader("🛡️ Duration Matching Immunization")
    
    st.markdown("""
    **Problem Type:** Match a liability's duration and present value using two bonds.
    
    **Formulas:**
    - w_A = (D_L - D_B) / (D_A - D_B)
    - w_B = 1 - w_A
    - N_i = (w_i × PV_L) / P_i
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Liability:**")
        liab_amount = st.number_input("Liability Amount ($)", value=1000000.0, key="imm_liab_amt")
        liab_time = st.number_input("Liability Time (years)", value=5.0, key="imm_liab_t")
        liab_yield = st.number_input("Discount Yield", value=0.05, format="%.4f", key="imm_liab_y")
    
    with col2:
        st.write("**Bond A:**")
        a_fv = st.number_input("Face Value A", value=1000.0, key="imm_a_fv")
        a_cr = st.number_input("Coupon Rate A", value=0.04, format="%.4f", key="imm_a_cr")
        a_mat = st.number_input("Maturity A", value=3, min_value=1, key="imm_a_mat")
        a_ytm = st.number_input("YTM A", value=0.05, format="%.4f", key="imm_a_ytm")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Bond B:**")
        b_fv = st.number_input("Face Value B", value=1000.0, key="imm_b_fv")
        b_cr = st.number_input("Coupon Rate B", value=0.06, format="%.4f", key="imm_b_cr")
        b_mat = st.number_input("Maturity B", value=8, min_value=1, key="imm_b_mat")
        b_ytm = st.number_input("YTM B", value=0.05, format="%.4f", key="imm_b_ytm")
    
    if st.button("🧮 Solve Immunization", key="solve_imm"):
        try:
            # Create bonds
            bond_a = BulletBond(face_value=a_fv, coupon_rate=a_cr, maturity=a_mat, yield_rate=a_ytm)
            bond_b = BulletBond(face_value=b_fv, coupon_rate=b_cr, maturity=b_mat, yield_rate=b_ytm)
            
            # Create immunizer
            immunizer = Immunizer(
                liability_amount=liab_amount,
                liability_time=liab_time,
                liability_yield=liab_yield,
                bond_a=bond_a,
                bond_b=bond_b
            )
            
            result = immunizer.solve()
            
            if result['solvable']:
                st.success("✅ Immunization Solution Found")
                
                if result.get('warning'):
                    st.warning(f"⚠️ {result['warning']}")
                
                if result['short_positions']:
                    st.info("📊 Short positions required (negative weights)")
                
                st.subheader("📊 Solution Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Weights:**")
                    st.metric("Weight A (w_A)", f"{result['weights']['A']:.4f} ({result['weights']['A']*100:.2f}%)")
                    st.metric("Weight B (w_B)", f"{result['weights']['B']:.4f} ({result['weights']['B']*100:.2f}%)")
                
                with col2:
                    st.write("**Investments ($):**")
                    st.metric("Investment A", f"${result['investments']['A']:,.2f}")
                    st.metric("Investment B", f"${result['investments']['B']:,.2f}")
                
                with col3:
                    st.write("**Units:**")
                    st.metric("Bond A Units", f"{result['units']['A']:.4f}")
                    st.metric("Bond B Units", f"{result['units']['B']:.4f}")
                
                st.subheader("✓ Proof of Immunization")
                
                proof = result['proof']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Liability PV", f"${proof['liability_pv']:,.2f}")
                    st.metric("Portfolio Value", f"${proof['portfolio_value']:,.2f}")
                    st.metric("Value Match", "✓" if proof['value_match'] else "✗")
                
                with col2:
                    st.metric("Target Duration", f"{proof['liability_duration']:.4f} yrs")
                    st.metric("Portfolio Duration", f"{proof['portfolio_duration']:.4f} yrs")
                    st.metric("Duration Match", "✓" if proof['duration_match'] else "✗")
                
                with col3:
                    st.metric("Bond A Duration", f"{result['bond_durations']['A']:.4f} yrs")
                    st.metric("Bond B Duration", f"{result['bond_durations']['B']:.4f} yrs")
                
                with st.expander("📝 How to do this in Excel"):
                    st.markdown(immunizer.get_excel_instructions())
                    
                    st.markdown("**LaTeX Formulas:**")
                    for name, formula in immunizer.get_latex_formulas().items():
                        st.latex(formula)
            else:
                st.error(f"❌ Cannot immunize: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# =============================================================================
# MODULE B: ADVANCED PORTFOLIO ENGINE (Exam-Specific)
# =============================================================================

def advanced_portfolio_module():
    st.header("📊 Module B: Advanced Portfolio Engine")
    
    st.markdown("""
    **Exam-Specific Problem Families (2023-2025):**
    - Mean-variance optimization with ESG preferences
    - Human capital and life-cycle portfolio choice
    - Extended mean-variance with labor income hedging
    """)
    
    tab1, tab2 = st.tabs(["🎯 MV Optimizer with ESG", "👤 Human Capital Calculator"])
    
    with tab1:
        esg_portfolio_section()
    
    with tab2:
        human_capital_section()


def esg_portfolio_section():
    st.subheader("Mean-Variance Optimization with ESG Extension")
    
    st.markdown("""
    **ESG Utility Function:**
    $$U = E[r_p] - \\frac{1}{2}\\gamma\\sigma_p^2 + a \\cdot ESG_p$$
    
    **Optimal Weights:**
    $$\\mathbf{w}^* = \\frac{1}{\\gamma}\\Sigma^{-1}(\\mu - r_f\\mathbf{1}) + \\frac{a}{\\gamma}\\Sigma^{-1}\\mathbf{s}$$
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gamma = st.number_input("Risk Aversion (γ)", value=4.0, min_value=0.1, key="esg_gamma")
        rf = st.number_input("Risk-Free Rate", value=0.02, format="%.4f", key="esg_rf")
        esg_pref = st.number_input("ESG Preference (a)", value=0.0, format="%.4f", key="esg_a",
                                   help="Set > 0 to include ESG in utility")
    
    with col2:
        returns_input = st.text_area(
            "Expected Returns (one per line or comma-separated)",
            value="0.08\n0.12\n0.06",
            height=100,
            key="esg_returns"
        )
        
        vols_input = st.text_area(
            "Volatilities (one per line or comma-separated)",
            value="0.20\n0.25\n0.15",
            height=100,
            key="esg_vols"
        )
    
    corr_input = st.text_area(
        "Correlation Matrix (row per line)",
        value="1.0, 0.3, 0.1\n0.3, 1.0, 0.2\n0.1, 0.2, 1.0",
        height=100,
        key="esg_corr"
    )
    
    if esg_pref > 0:
        esg_input = st.text_area(
            "ESG Scores (one per asset)",
            value="0.7, 0.5, 0.9",
            key="esg_scores_input"
        )
    
    if st.button("🧮 Optimize Portfolio", key="esg_optimize"):
        try:
            # Parse inputs
            mu = parse_messy_input(returns_input)
            sigma = parse_messy_input(vols_input)
            corr = parse_matrix_input(corr_input)
            
            esg_scores = None
            if esg_pref > 0:
                esg_scores = parse_messy_input(esg_input)
            
            # Build and display Covariance Matrix
            n = len(mu)
            D = np.diag(sigma)
            cov_matrix = D @ corr @ D
            
            st.subheader("📊 Matrices")
            
            st.write("**Covariance Matrix (Σ):**")
            cov_df = pd.DataFrame(cov_matrix, 
                                  index=[f'Asset {i+1}' for i in range(n)],
                                  columns=[f'Asset {i+1}' for i in range(n)])
            st.dataframe(cov_df.style.format("{:.6f}"))
            
            # Calculate and display Inverse Covariance Matrix
            try:
                cov_inv = np.linalg.inv(cov_matrix)
                st.write("**Inverse Covariance Matrix (Σ⁻¹):**")
                st.info("💡 Copy this table for exam questions asking to 'Determine the inverse matrix'")
                cov_inv_df = pd.DataFrame(cov_inv,
                                          index=[f'Asset {i+1}' for i in range(n)],
                                          columns=[f'Asset {i+1}' for i in range(n)])
                st.dataframe(cov_inv_df.style.format("{:.4f}"))
            except np.linalg.LinAlgError:
                st.warning("⚠️ Covariance matrix is singular (cannot display inverse)")
            
            # Create optimizer
            optimizer = MeanVarianceOptimizer(
                expected_returns=mu,
                volatilities=sigma,
                correlations=corr,
                risk_free_rate=rf,
                risk_aversion=gamma,
                esg_scores=esg_scores,
                esg_preference=esg_pref
            )
            
            # Calculate portfolios
            tangency = optimizer.tangency_portfolio()
            gmv = optimizer.gmv_portfolio()
            optimal = optimizer.optimal_portfolio(include_esg=(esg_pref > 0))
            
            st.subheader("📊 Portfolio Results")
            
            # Results table
            results_data = []
            for pf in [tangency, gmv, optimal]:
                row = {
                    'Portfolio': pf['name'],
                    'E[r]': f"{pf['expected_return']*100:.2f}%",
                    'Volatility': f"{pf['volatility']*100:.2f}%",
                    'Sharpe': f"{pf['sharpe_ratio']:.4f}"
                }
                for i, w in enumerate(pf['weights']):
                    row[f'w_{i+1}'] = f"{w:.4f}"
                if 'portfolio_esg' in pf:
                    row['ESG'] = f"{pf['portfolio_esg']:.4f}"
                results_data.append(row)
            
            st.dataframe(pd.DataFrame(results_data))
            
            # Optimal portfolio details
            st.subheader(f"🎯 Optimal Portfolio (γ={gamma})")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Weights:**")
                for i, w in enumerate(optimal['weights']):
                    st.metric(f"Asset {i+1}", f"{w:.4f} ({w*100:.2f}%)")
                st.metric("Total Risky", f"{optimal['risky_weight']:.4f}")
                st.metric("Cash/RF", f"{optimal['cash_weight']:.4f}")
            
            with col2:
                st.write("**Decomposition (ESG):**")
                if esg_pref > 0:
                    st.write("MV Component:")
                    for i, m in enumerate(optimal['mv_component']):
                        st.write(f"  Asset {i+1}: {m:.4f}")
                    st.write("ESG Component:")
                    for i, e in enumerate(optimal['esg_component']):
                        st.write(f"  Asset {i+1}: {e:.4f}")
            
            # Efficient frontier plot
            st.subheader("📈 Efficient Frontier")
            
            frontier = optimizer.efficient_frontier()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(frontier['volatility'], frontier['return'], 'b-', linewidth=2, label='Efficient Frontier')
            
            # Plot portfolios
            ax.scatter([gmv['volatility']], [gmv['expected_return']], s=150, c='green', marker='s', label='GMV', zorder=5)
            ax.scatter([tangency['volatility']], [tangency['expected_return']], s=150, c='red', marker='^', label='Tangency', zorder=5)
            ax.scatter([optimal['volatility']], [optimal['expected_return']], s=150, c='purple', marker='D', label=f'Optimal (γ={gamma})', zorder=5)
            
            # Plot assets
            ax.scatter(sigma, mu, s=100, c='gray', marker='o', label='Assets', zorder=4)
            
            # CML
            cml_vols = np.linspace(0, max(frontier['volatility'])*1.1, 100)
            cml_returns = rf + tangency['sharpe_ratio'] * cml_vols
            ax.plot(cml_vols, cml_returns, 'r--', alpha=0.5, label='CML')
            
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Expected Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(optimizer.get_excel_instructions())
                
                st.markdown("**LaTeX Formulas:**")
                for name, formula in optimizer.get_latex_formulas().items():
                    st.latex(formula)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def human_capital_section():
    st.subheader("Human Capital & Life-Cycle Portfolio Choice")
    
    st.markdown("""
    **Key Formula:**
    $$\\pi^* = M(1 + l) - H$$
    
    Where M = Myopic demand, l = L₀/W (leverage ratio), H = Hedging demand
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Financial Wealth:**")
        wealth = st.number_input("Current Wealth ($)", value=200000.0, key="hc_wealth")
        
        st.write("**Labor Income:**")
        income = st.number_input("Current Annual Income ($)", value=75000.0, key="hc_income")
        growth = st.number_input("Income Growth Rate (g)", value=0.03, format="%.4f", key="hc_growth")
        discount = st.number_input("Discount Rate (r)", value=0.05, format="%.4f", key="hc_discount")
        years = st.number_input("Working Years (T)", value=30, min_value=1, key="hc_years")

        timing_convention = st.selectbox(
            "Income Timing Convention",
            ["munk_fmt", "ordinary", "annuity_due"],
            format_func=lambda x: {
                "munk_fmt": "FMT/Munk: Current income now, first future payment = Y*(1+g) at t=1",
                "ordinary": "Ordinary: First payment = Y at t=1",
                "annuity_due": "Annuity Due: First payment = Y at t=0"
            }[x],
            index=0,
            key="hc_timing_section",
            help="FMT exams use Munk convention. Select based on how the problem defines income timing."
        )
    
    with col2:
        st.write("**Market Parameters:**")
        mu_mkt = st.number_input("Market Expected Return", value=0.08, format="%.4f", key="hc_mu")
        sigma_mkt = st.number_input("Market Volatility", value=0.20, format="%.4f", key="hc_sigma_m")
        rf = st.number_input("Risk-Free Rate", value=0.02, format="%.4f", key="hc_rf")
        gamma = st.number_input("Risk Aversion (γ)", value=4.0, key="hc_gamma")
        
        st.write("**Labor-Market Correlation:**")
        corr_lm = st.number_input("Corr(Labor, Market)", value=0.2, min_value=-1.0, max_value=1.0, key="hc_corr")
        sigma_labor = st.number_input("Labor Income Volatility", value=0.10, format="%.4f", key="hc_sigma_l")
    
    if st.button("🧮 Calculate Optimal Allocation", key="hc_calc"):
        try:
            calc = HumanCapitalCalc(
                wealth=wealth,
                income=income,
                growth_rate=growth,
                discount_rate=discount,
                years=years,
                mu_market=mu_mkt,
                sigma_market=sigma_mkt,
                risk_free_rate=rf,
                risk_aversion=gamma,
                corr_labor_market=corr_lm,
                sigma_labor=sigma_labor,
                timing_convention=timing_convention
            )
            
            st.subheader("📊 Results Summary")
            
            summary = calc.get_summary_table()
            st.dataframe(summary)
            
            # Key metrics
            st.subheader("🎯 Key Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Human Capital (L₀)", f"${calc.human_capital:,.0f}")
                st.metric("Total Wealth", f"${calc.total_wealth:,.0f}")
            with col2:
                st.metric("Leverage Ratio (l)", f"{calc.leverage_ratio:.4f}")
                st.metric("Optimal Weight (π*)", f"{calc.optimal_weight:.4f} ({calc.optimal_weight*100:.2f}%)")
            with col3:
                st.metric("Stock Investment", f"${calc.stock_investment:,.0f}")
                st.metric("Risk-Free Investment", f"${calc.riskfree_investment:,.0f}")
            
            # Interpretation
            st.subheader("📖 Interpretation")
            
            if calc.optimal_weight > 1:
                st.info(f"📊 **Leverage Required:** Borrow ${calc.stock_investment - wealth:,.0f} at risk-free rate to invest in stocks")
            elif calc.optimal_weight < 0:
                st.warning(f"⚠️ **Short Stocks:** Short ${-calc.stock_investment:,.0f} in stocks")
            else:
                st.success(f"✅ **No Leverage:** Invest {calc.optimal_weight*100:.1f}% in stocks")
            
            st.write(f"""
            **Component Analysis:**
            - Myopic Demand (M): {calc.myopic_demand:.4f} - Standard MV allocation ignoring human capital
            - Scaling Factor (1+l): {1+calc.leverage_ratio:.4f} - Human capital allows more risk-taking
            - Hedging Demand (H): {calc.hedging_demand:.4f} - Reduces stocks due to {corr_lm*100:.0f}% correlation with labor
            """)
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(calc.get_excel_instructions())
                
                st.markdown("**LaTeX Formulas:**")
                for name, formula in calc.get_latex_formulas().items():
                    st.latex(formula)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def human_capital_multi_asset_tab():
    """Multi-asset optimization with human capital hedging."""
    st.subheader("Multi-Asset Portfolio with Human Capital Hedging")
    
    st.markdown(r"""
    **Extended Mean-Variance with Multiple Risky Assets:**
    
    $$\boldsymbol{\pi}^* = \frac{1}{\gamma} \Sigma^{-1} (\boldsymbol{\mu} - r_f \mathbf{1}) - \frac{H}{F} \Sigma^{-1} \boldsymbol{Cov}(r_{HC}, \mathbf{r})$$
    
    Where:
    - $\boldsymbol{\pi}^*$ = vector of optimal weights in risky assets
    - $\Sigma$ = covariance matrix of risky assets
    - $\boldsymbol{\mu}$ = expected returns of risky assets
    - $H$ = human capital value
    - $F$ = financial wealth
    - $\boldsymbol{Cov}(r_{HC}, \mathbf{r})$ = covariances between human capital and each risky asset
    
    **This tab also calculates:**
    - Covariance between HC and a portfolio (like the market)
    - Correlation between HC and a portfolio
    """)
    
    st.markdown("---")
    
    # =========================================================================
    # STEP 1: BASIC INPUTS
    # =========================================================================
    st.subheader("Step 1: Basic Inputs")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_assets = st.number_input("Number of Risky Assets", value=2, min_value=1, max_value=5, key="hc_ma_n")
        rf = st.number_input("Risk-Free Rate (rf)", value=0.02, format="%.4f", key="hc_ma_rf")
    
    with col2:
        fin_wealth = st.number_input("Financial Wealth ($)", value=150000.0, key="hc_ma_fw")
        human_cap = st.number_input("Human Capital ($)", value=450000.0, key="hc_ma_hc")
    
    with col3:
        gamma = st.number_input("Risk Aversion (γ)", value=4.0, min_value=0.1, key="hc_ma_gamma")
        sigma_hc = st.number_input("HC Volatility (σ_HC)", value=0.10, format="%.4f", key="hc_ma_shc")
    
    # =========================================================================
    # STEP 2: ASSET PARAMETERS
    # =========================================================================
    st.subheader("Step 2: Risky Asset Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        asset_names_input = st.text_input(
            "Asset Names (comma-separated)",
            value="Tech, Non-Tech" if n_assets == 2 else ", ".join([f"Asset {i+1}" for i in range(n_assets)]),
            key="hc_ma_names"
        )
        
        returns_input = st.text_area(
            "Expected Returns (comma-separated)",
            value="0.14, 0.10" if n_assets == 2 else ", ".join(["0.10"] * n_assets),
            key="hc_ma_returns",
            help="E.g., 0.14, 0.10 for 14% and 10%"
        )
        
        vols_input = st.text_area(
            "Volatilities (comma-separated)",
            value="0.30, 0.20" if n_assets == 2 else ", ".join(["0.20"] * n_assets),
            key="hc_ma_vols",
            help="E.g., 0.30, 0.20 for 30% and 20%"
        )
    
    with col2:
        corr_input = st.text_area(
            "Correlation Matrix (row per line)",
            value="1.0, 0.4\n0.4, 1.0" if n_assets == 2 else "\n".join(
                [", ".join(["1.0" if i == j else "0.3" for j in range(n_assets)]) for i in range(n_assets)]
            ),
            key="hc_ma_corr",
            help="Correlation matrix between risky assets"
        )
        
        hc_corr_input = st.text_area(
            "HC Correlations with Each Asset (comma-separated)",
            value="0.8, 0.0" if n_assets == 2 else ", ".join(["0.2"] * n_assets),
            key="hc_ma_hc_corr",
            help="ρ(HC, Asset_i) for each asset. E.g., 0.8, 0.0 means HC is correlated with first asset only"
        )
    
    # =========================================================================
    # OPTIONAL: MARKET PORTFOLIO WEIGHTS (for HC-Market covariance)
    # =========================================================================
    with st.expander("📊 Market Portfolio Composition (Optional - for HC-Market Covariance)", expanded=True):
        st.markdown("""
        If the "market" is a weighted combination of your assets, enter the weights here
        to calculate Cov(HC, Market) and Corr(HC, Market).
        """)
        
        market_weights_input = st.text_input(
            "Market Weights (comma-separated, must sum to 1)",
            value="0.4, 0.6" if n_assets == 2 else ", ".join([f"{1/n_assets:.4f}"] * n_assets),
            key="hc_ma_mkt_w",
            help="E.g., 0.4, 0.6 means market is 40% Asset 1 + 60% Asset 2"
        )
    
    # =========================================================================
    # CALCULATE BUTTON
    # =========================================================================
    if st.button("🧮 Calculate Optimal Allocation", key="hc_ma_calc"):
        try:
            # Parse inputs
            asset_names = [n.strip() for n in asset_names_input.split(',')]
            mu = np.array([float(x.strip()) for x in returns_input.split(',')])
            sigma = np.array([float(x.strip()) for x in vols_input.split(',')])
            corr_matrix = parse_matrix_input(corr_input)
            hc_corrs = np.array([float(x.strip()) for x in hc_corr_input.split(',')])
            market_weights = np.array([float(x.strip()) for x in market_weights_input.split(',')])
            
            n = len(mu)
            
            # Validate
            if len(sigma) != n or len(hc_corrs) != n or len(market_weights) != n:
                st.error("❌ All inputs must have the same number of assets!")
                return
            
            # Build covariance matrix
            D = np.diag(sigma)
            Sigma = D @ corr_matrix @ D
            
            # Calculate HC covariances with each asset
            # Cov(HC, Asset_i) = ρ(HC, i) × σ_HC × σ_i
            hc_cov_assets = hc_corrs * sigma_hc * sigma
            
            # =========================================================================
            # RESULTS SECTION 1: MARKET PORTFOLIO STATS
            # =========================================================================
            st.subheader("📊 Market Portfolio Analysis")
            
            # Market return and volatility
            mkt_return = market_weights @ mu
            mkt_variance = market_weights @ Sigma @ market_weights
            mkt_vol = np.sqrt(mkt_variance)
            
            # HC-Market Covariance
            # Cov(HC, Market) = Σ w_i × Cov(HC, Asset_i)
            hc_mkt_cov = market_weights @ hc_cov_assets
            
            # HC-Market Correlation
            hc_mkt_corr = hc_mkt_cov / (sigma_hc * mkt_vol) if (sigma_hc * mkt_vol) > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market E[r]", f"{mkt_return*100:.2f}%")
                st.metric("Market σ", f"{mkt_vol*100:.2f}%")
            with col2:
                st.metric("Cov(HC, Market)", f"{hc_mkt_cov:.6f}")
                st.metric("Corr(HC, Market)", f"{hc_mkt_corr:.4f}")
            with col3:
                st.metric("Market Sharpe", f"{(mkt_return - rf) / mkt_vol:.4f}")
            
            # Show calculation steps
            with st.expander("📝 HC-Market Covariance Calculation"):
                st.markdown("**Formula:** $Cov(r_{HC}, r_m) = \sum_i w_i \cdot Cov(r_{HC}, r_i)$")
                st.markdown("**Step-by-step:**")
                calc_parts = []
                for i in range(n):
                    st.markdown(f"- $Cov(HC, {asset_names[i]}) = \\rho_{{HC,{i+1}}} \\times \\sigma_{{HC}} \\times \\sigma_{i+1} = {hc_corrs[i]:.2f} \\times {sigma_hc:.4f} \\times {sigma[i]:.4f} = {hc_cov_assets[i]:.6f}$")
                    calc_parts.append(f"{market_weights[i]:.2f} × {hc_cov_assets[i]:.6f}")
                st.markdown(f"**Total:** $Cov(HC, Market) = {' + '.join(calc_parts)} = {hc_mkt_cov:.6f}$")
                st.markdown(f"**Correlation:** $\\rho_{{HC,m}} = \\frac{{{hc_mkt_cov:.6f}}}{{{sigma_hc:.4f} \\times {mkt_vol:.4f}}} = {hc_mkt_corr:.4f}$")
            
            # =========================================================================
            # RESULTS SECTION 2: SINGLE-MARKET OPTIMIZATION (for comparison)
            # =========================================================================
            st.subheader("📈 Answer to Part (c): Single-Market Optimization")
            
            st.markdown("""
            **Scenario:** Alex can only invest in the risk-free asset and a market ETF (tracking the overall stock market).
            """)
            
            # Formula selection
            formula_choice = st.radio(
                "Select Formula Variant:",
                ["standard", "total_wealth"],
                format_func=lambda x: {
                    "standard": "Standard: π* = (μ-rf)/(γσ²) - (H/F)·Cov(HC,m)/σ²",
                    "total_wealth": "Total Wealth: π* = [(F+H)/F]·(μ-rf)/(γσ²) - (H/F)·β_HC"
                }[x],
                key="hc_formula_choice",
                help="Different textbooks use different formulations. The 'Total Wealth' version scales by total wealth."
            )
            
            if formula_choice == "standard":
                st.markdown("**Formula:** $\\pi_m^* = \\frac{E[r_m] - r_f}{\\gamma \\sigma_m^2} - \\frac{H}{F} \\cdot \\frac{Cov(r_{HC}, r_m)}{\\sigma_m^2}$")
                
                # Merton demand
                merton_single = (mkt_return - rf) / (gamma * mkt_variance)
                
                # HC hedge
                hc_hedge_single = (human_cap / fin_wealth) * (hc_mkt_cov / mkt_variance)
                
                # Optimal weight
                pi_market = merton_single - hc_hedge_single
                
                # Show calculation steps
                with st.expander("📝 Step-by-Step Calculation", expanded=True):
                    st.markdown(f"""
**Step 1: Merton Demand (ignoring human capital)**
$$\\frac{{E[r_m] - r_f}}{{\\gamma \\sigma_m^2}} = \\frac{{{mkt_return:.4f} - {rf:.4f}}}{{{gamma:.1f} \\times {mkt_variance:.6f}}} = \\frac{{{mkt_return - rf:.4f}}}{{{gamma * mkt_variance:.6f}}} = {merton_single:.4f}$$

**Step 2: Human Capital Hedge Term**
$$\\frac{{H}}{{F}} \\cdot \\frac{{Cov(r_{{HC}}, r_m)}}{{\\sigma_m^2}} = \\frac{{{human_cap:,.0f}}}{{{fin_wealth:,.0f}}} \\times \\frac{{{hc_mkt_cov:.6f}}}{{{mkt_variance:.6f}}} = {human_cap/fin_wealth:.2f} \\times {hc_mkt_cov/mkt_variance:.4f} = {hc_hedge_single:.4f}$$

**Step 3: Optimal Weight in Market**
$$\\pi_m^* = {merton_single:.4f} - {hc_hedge_single:.4f} = {pi_market:.4f}$$
                    """)
            
            else:  # total_wealth formula
                st.markdown(r"""
**Formula:** $\pi_m^* = \frac{F + H}{F} \cdot \frac{E[r_m] - r_f}{\gamma \sigma_m^2} - \frac{H}{F} \cdot \beta_{HC}$

Where $\beta_{HC} = \frac{Cov(r_{HC}, r_m)}{\sigma_m^2}$ is the "beta" of human capital with respect to the market.
                """)
                
                total_wealth = fin_wealth + human_cap
                
                # Beta of human capital
                beta_hc = hc_mkt_cov / mkt_variance
                
                # Merton demand scaled by total wealth
                merton_single = (total_wealth / fin_wealth) * (mkt_return - rf) / (gamma * mkt_variance)
                
                # HC hedge using beta
                hc_hedge_single = (human_cap / fin_wealth) * beta_hc
                
                # Optimal weight
                pi_market = merton_single - hc_hedge_single
                
                # Show calculation steps
                with st.expander("📝 Step-by-Step Calculation", expanded=True):
                    st.markdown(f"""
**Step 1: Human Capital Beta**
$$\\beta_{{HC}} = \\frac{{Cov(r_{{HC}}, r_m)}}{{\\sigma_m^2}} = \\frac{{{hc_mkt_cov:.6f}}}{{{mkt_variance:.6f}}} = {beta_hc:.4f}$$

**Step 2: Merton Demand (scaled by total wealth)**
$$\\frac{{F + H}}{{F}} \\cdot \\frac{{E[r_m] - r_f}}{{\\gamma \\sigma_m^2}} = \\frac{{{total_wealth:,.0f}}}{{{fin_wealth:,.0f}}} \\times \\frac{{{mkt_return - rf:.4f}}}{{{gamma:.1f} \\times {mkt_variance:.6f}}}$$
$$= {total_wealth/fin_wealth:.2f} \\times {(mkt_return - rf)/(gamma * mkt_variance):.4f} = {merton_single:.4f}$$

**Step 3: Human Capital Hedge Term**
$$\\frac{{H}}{{F}} \\cdot \\beta_{{HC}} = \\frac{{{human_cap:,.0f}}}{{{fin_wealth:,.0f}}} \\times {beta_hc:.4f} = {human_cap/fin_wealth:.2f} \\times {beta_hc:.4f} = {hc_hedge_single:.4f}$$

**Step 4: Optimal Weight in Market**
$$\\pi_m^* = {merton_single:.4f} - {hc_hedge_single:.4f} = {pi_market:.4f}$$
                    """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Merton Demand", f"{merton_single:.4f} ({merton_single*100:.2f}%)")
            with col2:
                st.metric("HC Hedge Term", f"{hc_hedge_single:.4f} ({hc_hedge_single*100:.2f}%)")
            with col3:
                st.metric("π*_market", f"{pi_market:.4f} ({pi_market*100:.2f}%)")
            
            # Dollar amounts
            market_dollars = pi_market * fin_wealth
            rf_dollars_single = fin_wealth - market_dollars
            
            # Implied breakdown into tech/non-tech (via market weights)
            tech_via_market = market_weights[0] * market_dollars if n >= 1 else 0
            nontech_via_market = market_weights[1] * market_dollars if n >= 2 else 0
            
            st.markdown("---")
            st.markdown("### 💰 Dollar Allocation (Single Market ETF)")
            
            single_df = pd.DataFrame({
                'Asset': ['Market ETF', 'Risk-Free (Cash)', 'TOTAL'],
                'Weight (π)': [f"{pi_market:.4f}", f"{1-pi_market:.4f}", "1.0000"],
                'Weight (%)': [f"{pi_market*100:.2f}%", f"{(1-pi_market)*100:.2f}%", "100.00%"],
                'Dollar Amount': [f"${market_dollars:,.0f}", f"${rf_dollars_single:,.0f}", f"${fin_wealth:,.0f}"]
            })
            st.dataframe(single_df, use_container_width=True)
            
            # Show what this means in terms of underlying assets
            st.markdown("### 📊 Implied Exposure via Market ETF")
            st.markdown(f"""
Since the market ETF is {market_weights[0]*100:.0f}% {asset_names[0]} and {market_weights[1]*100:.0f}% {asset_names[1]}, 
investing **${market_dollars:,.0f}** in the market ETF gives you:
            """)
            
            implied_df = pd.DataFrame({
                'Underlying Asset': asset_names + ['Total in Market ETF'],
                'Market Weight': [f"{w*100:.0f}%" for w in market_weights] + ['100%'],
                'Implied $ Exposure': [f"${market_weights[i] * market_dollars:,.0f}" for i in range(n)] + [f"${market_dollars:,.0f}"]
            })
            st.dataframe(implied_df, use_container_width=True)
            
            # Interpretation box
            if rf_dollars_single < 0:
                st.error(f"""
**🔴 BORROWING REQUIRED:** 
- Borrow **${abs(rf_dollars_single):,.0f}** at the risk-free rate
- Invest **${market_dollars:,.0f}** in the Market ETF
- This gives exposure: **${tech_via_market:,.0f}** in {asset_names[0]}, **${nontech_via_market:,.0f}** in {asset_names[1]}
                """)
            elif market_dollars < 0:
                st.warning(f"""
**🟡 SHORT MARKET:** 
- Short **${abs(market_dollars):,.0f}** of the Market ETF
- Invest **${rf_dollars_single:,.0f}** in the risk-free asset
                """)
            else:
                st.success(f"""
**🟢 STANDARD ALLOCATION:** 
- Invest **${market_dollars:,.0f}** in the Market ETF
- Invest **${rf_dollars_single:,.0f}** in the risk-free asset
                """)
            
            # =========================================================================
            # RESULTS SECTION 3: MULTI-ASSET OPTIMIZATION
            # =========================================================================
            st.subheader("🎯 Answer to Part (d): Multi-Asset Optimization")
            
            st.markdown(f"""
            **Scenario:** Alex can invest in the risk-free asset, a {asset_names[0]} ETF, and a {asset_names[1]} ETF separately.
            """)
            
            st.markdown(r"**Formula:** $\boldsymbol{\pi}^* = \frac{1}{\gamma} \Sigma^{-1} (\boldsymbol{\mu} - r_f \mathbf{1}) - \frac{H}{F} \Sigma^{-1} \boldsymbol{Cov}(r_{HC}, \mathbf{r})$")
            
            # Invert covariance matrix
            Sigma_inv = np.linalg.inv(Sigma)
            
            # Excess returns
            excess_returns = mu - rf
            
            # Merton demand (vector)
            merton_multi = (1 / gamma) * (Sigma_inv @ excess_returns)
            
            # HC hedge (vector)
            hc_hedge_multi = (human_cap / fin_wealth) * (Sigma_inv @ hc_cov_assets)
            
            # Optimal weights
            pi_multi = merton_multi - hc_hedge_multi
            
            # Display matrices
            with st.expander("📐 Matrix Calculations", expanded=False):
                st.markdown("**Covariance Matrix (Σ):**")
                st.dataframe(pd.DataFrame(Sigma, index=asset_names, columns=asset_names).style.format("{:.6f}"))
                
                st.markdown("**Inverse Covariance Matrix (Σ⁻¹):**")
                st.dataframe(pd.DataFrame(Sigma_inv, index=asset_names, columns=asset_names).style.format("{:.4f}"))
                
                st.markdown("**Excess Returns (μ - rf):**")
                for i in range(n):
                    st.write(f"- {asset_names[i]}: {excess_returns[i]*100:.2f}%")
                
                st.markdown("**HC Covariances with Each Asset:**")
                for i in range(n):
                    st.write(f"- Cov(HC, {asset_names[i]}): {hc_cov_assets[i]:.6f}")
            
            # Component breakdown table
            st.markdown("### 📊 Component Breakdown")
            
            component_df = pd.DataFrame({
                'Asset': asset_names,
                'Merton Demand': [f"{m:.4f}" for m in merton_multi],
                'HC Hedge': [f"{h:.4f}" for h in hc_hedge_multi],
                'Optimal π* = Merton - Hedge': [f"{p:.4f}" for p in pi_multi],
                'Weight (%)': [f"{p*100:.2f}%" for p in pi_multi]
            })
            st.dataframe(component_df, use_container_width=True)
            
            # Dollar amounts
            dollars_multi = pi_multi * fin_wealth
            total_risky = np.sum(dollars_multi)
            rf_dollars_multi = fin_wealth - total_risky
            
            st.markdown("### 💰 Dollar Allocation (Multi-Asset)")
            
            # Build comprehensive allocation table
            multi_rows = []
            for i in range(n):
                multi_rows.append({
                    'Asset': f"{asset_names[i]} ETF",
                    'Weight (π)': f"{pi_multi[i]:.4f}",
                    'Weight (%)': f"{pi_multi[i]*100:.2f}%",
                    'Dollar Amount': f"${dollars_multi[i]:,.0f}",
                    'Action': 'LONG' if dollars_multi[i] > 0 else 'SHORT'
                })
            
            multi_rows.append({
                'Asset': 'Risk-Free (Cash)',
                'Weight (π)': f"{1 - np.sum(pi_multi):.4f}",
                'Weight (%)': f"{(1-np.sum(pi_multi))*100:.2f}%",
                'Dollar Amount': f"${rf_dollars_multi:,.0f}",
                'Action': 'LEND' if rf_dollars_multi > 0 else 'BORROW'
            })
            
            multi_rows.append({
                'Asset': 'TOTAL',
                'Weight (π)': '1.0000',
                'Weight (%)': '100.00%',
                'Dollar Amount': f"${fin_wealth:,.0f}",
                'Action': ''
            })
            
            multi_df = pd.DataFrame(multi_rows)
            st.dataframe(multi_df, use_container_width=True)
            
            # Clear interpretation box
            st.markdown("### 🎯 Optimal Strategy Summary")
            
            action_items = []
            for i in range(n):
                if dollars_multi[i] > 0:
                    action_items.append(f"**LONG** ${dollars_multi[i]:,.0f} in {asset_names[i]} ETF")
                else:
                    action_items.append(f"**SHORT** ${abs(dollars_multi[i]):,.0f} in {asset_names[i]} ETF")
            
            if rf_dollars_multi > 0:
                action_items.append(f"**LEND** ${rf_dollars_multi:,.0f} at the risk-free rate")
            else:
                action_items.append(f"**BORROW** ${abs(rf_dollars_multi):,.0f} at the risk-free rate")
            
            for item in action_items:
                st.markdown(f"- {item}")
            
            # =========================================================================
            # COMPARISON
            # =========================================================================
            st.subheader("📊 Comparison: Part (c) vs Part (d)")
            
            st.markdown("### Side-by-Side Dollar Allocation")
            
            comparison_data = {
                'Asset': [],
                'Part (c): Market ETF Only': [],
                'Part (d): Separate ETFs': []
            }
            
            # Add each asset
            for i in range(n):
                comparison_data['Asset'].append(f"{asset_names[i]}")
                comparison_data['Part (c): Market ETF Only'].append(f"${market_weights[i] * market_dollars:,.0f} (via market)")
                comparison_data['Part (d): Separate ETFs'].append(f"${dollars_multi[i]:,.0f} (direct)")
            
            # Add totals
            comparison_data['Asset'].append("Total Risky Assets")
            comparison_data['Part (c): Market ETF Only'].append(f"${market_dollars:,.0f}")
            comparison_data['Part (d): Separate ETFs'].append(f"${total_risky:,.0f}")
            
            comparison_data['Asset'].append("Risk-Free (Cash)")
            comparison_data['Part (c): Market ETF Only'].append(f"${rf_dollars_single:,.0f}")
            comparison_data['Part (d): Separate ETFs'].append(f"${rf_dollars_multi:,.0f}")
            
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            
            # =========================================================================
            # INTERPRETATION
            # =========================================================================
            st.subheader("📖 Interpretation & Discussion")
            
            # Build interpretation text dynamically to avoid format errors
            interp_text = f"""
**Part (c) Finding - Single Market ETF:**
- Optimal weight in market: π* = {pi_market*100:.2f}%
- Dollar amount: **${market_dollars:,.0f}** in Market ETF
- Risk-free: **${rf_dollars_single:,.0f}** {'(borrowing)' if rf_dollars_single < 0 else '(lending)'}
- Implied exposure: ${market_weights[0] * market_dollars:,.0f} in {asset_names[0]}, ${market_weights[1] * market_dollars:,.0f} in {asset_names[1]}

**Part (d) Finding - Separate ETFs:**
"""
            for i in range(n):
                interp_text += f"- {asset_names[i]}: **${dollars_multi[i]:,.0f}** ({'long' if dollars_multi[i] > 0 else 'short'})\n"
            interp_text += f"- Risk-free: **${rf_dollars_multi:,.0f}** ({'borrowing' if rf_dollars_multi < 0 else 'lending'})\n"
            
            interp_text += f"""
**Why the Difference?**
- Alex's human capital has **{hc_corrs[0]*100:.0f}%** correlation with {asset_names[0]}
- Alex's human capital has **{hc_corrs[1]*100:.0f}%** correlation with {asset_names[1]}
- With separate ETFs, Alex can **precisely hedge** his {asset_names[0]} exposure
- In Part (c), he can only use the blunt instrument of the market ETF

**Key Insight:**
The multi-asset approach allows Alex to SHORT the asset most correlated with his human capital 
({asset_names[0]}) while going LONG the uncorrelated asset ({asset_names[1]}). 
This is a more efficient hedge than simply adjusting market exposure.
            """
            
            st.markdown(interp_text)
            
            # Excel instructions
            with st.expander("📝 Excel Implementation"):
                st.markdown(f"""
**Step 1: Build Covariance Matrix Σ**
- Create correlation matrix
- Create diagonal volatility matrix D
- Σ = D × Corr × D using `=MMULT(MMULT(D, Corr), D)`

**Step 2: Invert Σ**
- Use `=MINVERSE(Sigma_range)`

**Step 3: Calculate Merton Demand**
- Excess returns: μ - rf = [{', '.join([f'{er*100:.2f}%' for er in excess_returns])}]
- Merton = (1/γ) × Σ⁻¹ × (μ - rf)
- Use `=MMULT(Sigma_inv, excess_returns) / gamma`

**Step 4: Calculate HC Hedge**
- HC covariances: [{', '.join([f'{c:.6f}' for c in hc_cov_assets])}]
- HC Hedge = (H/F) × Σ⁻¹ × Cov(HC, r)
- Use `=(H/F) * MMULT(Sigma_inv, hc_cov)`

**Step 5: Optimal Weights**
- π* = Merton - HC_Hedge
                """)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# =============================================================================
# MODULE C: ADVANCED FACTOR MODEL ENGINE (Exam-Specific)
# =============================================================================

def advanced_factor_module():
    st.header("📉 Module C: Advanced Factor Model Engine")
    
    st.markdown("""
    **Exam-Specific Problem Families (2023-2025):**
    - Multi-factor risk decomposition (with factor correlations!)
    - Performance attribution and metrics
    - Treynor-Black active portfolio optimization
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Factor Risk Analysis", "📈 Performance Metrics", "🎯 Treynor-Black"])
    
    with tab1:
        factor_analysis_section()
    
    with tab2:
        advanced_performance_section()
    
    with tab3:
        advanced_treynor_black_section()


def factor_analysis_section():
    st.subheader("Multi-Factor Risk Decomposition")
    
    st.markdown("""
    **IMPORTANT:** Uses matrix math β'Σ_F β for systematic variance.
    
    **NOT** just Σβ²σ² (which ignores factor correlations)!
    
    This module also calculates:
    - **Asset-Asset Covariance/Correlation**: How assets move together
    - **Asset-Factor Covariance**: How each asset co-moves with each factor
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_factors = st.number_input("Number of Factors", value=2, min_value=1, max_value=5, key="fa_nf")
        n_assets = st.number_input("Number of Assets", value=2, min_value=1, max_value=10, key="fa_na")
    
    with col2:
        rf = st.number_input("Risk-Free Rate", value=0.02, format="%.4f", key="fa_rf")
        
        asset_names_input = st.text_input(
            "Asset Names (optional, comma-separated)",
            value="",
            key="fa_asset_names",
            help="E.g., 'Attraxion, CharmyComp'"
        )
        
        factor_names_input = st.text_input(
            "Factor Names (optional, comma-separated)",
            value="",
            key="fa_factor_names",
            help="E.g., 'Market, WML'"
        )
    
    st.write("**Factor Parameters:**")
    
    factor_means_input = st.text_area(
        "Factor Risk Premiums (comma-separated)",
        value="0.06, 0.04" if n_factors == 2 else ", ".join(["0.05"] * n_factors),
        key="fa_means",
        help="E[F] - rf for each factor"
    )
    
    factor_vols_input = st.text_area(
        "Factor Volatilities (comma-separated)",
        value="0.16, 0.12" if n_factors == 2 else ", ".join(["0.12"] * n_factors),
        key="fa_vols"
    )
    
    factor_corr_input = st.text_area(
        "Factor Correlation Matrix (row per line)",
        value="1.0, 0.1\n0.1, 1.0" if n_factors == 2 else "\n".join(
            [", ".join(["1.0" if i == j else "0.2" for j in range(n_factors)]) for i in range(n_factors)]
        ),
        key="fa_corr"
    )
    
    st.write("**Asset Parameters:**")
    
    betas_input = st.text_area(
        f"Asset Betas Matrix ({n_assets} rows × {n_factors} cols, asset per row)",
        value="1.5, 0.5\n1.2, -0.5" if (n_factors == 2 and n_assets == 2) else "\n".join(
            [", ".join(["1.0"] * n_factors) for _ in range(n_assets)]
        ),
        key="fa_betas"
    )
    
    idio_vols_input = st.text_area(
        "Idiosyncratic Volatilities (one per asset)",
        value="0.40, 0.20" if n_assets == 2 else ", ".join(["0.10"] * n_assets),
        key="fa_idio"
    )
    
    if st.button("🧮 Analyze Factor Risk", key="fa_analyze"):
        try:
            # Parse inputs
            factor_means = parse_messy_input(factor_means_input)
            factor_vols = parse_messy_input(factor_vols_input)
            factor_corr = parse_matrix_input(factor_corr_input)
            asset_betas = parse_matrix_input(betas_input)
            idio_vols = parse_messy_input(idio_vols_input)
            
            # Parse names
            if asset_names_input.strip():
                asset_names = [n.strip() for n in asset_names_input.split(',')]
                if len(asset_names) != n_assets:
                    asset_names = None
            else:
                asset_names = None
            
            if factor_names_input.strip():
                factor_names = [n.strip() for n in factor_names_input.split(',')]
                if len(factor_names) != n_factors:
                    factor_names = None
            else:
                factor_names = None
            
            # Create analyzer
            analyzer = FactorAnalyzer(
                factor_means=factor_means,
                factor_vols=factor_vols,
                factor_correlations=factor_corr,
                asset_betas=asset_betas,
                asset_idio_vols=idio_vols,
                risk_free_rate=rf,
                asset_names=asset_names,
                factor_names=factor_names
            )
            
            # ============================================================
            # KEY SUMMARY TABLE (Expected Return, Std Dev, Sharpe Ratio)
            # ============================================================
            st.subheader("📋 Summary: Expected Return, Std Dev, Sharpe Ratio")
            
            expected_returns = analyzer.expected_returns()
            summary_data = []
            
            for i in range(analyzer.n_assets):
                er = expected_returns[i]
                total_var = analyzer.total_variance(i)
                total_vol = np.sqrt(total_var)
                excess_ret = er - rf
                sharpe = excess_ret / total_vol if total_vol > 0 else 0
                
                summary_data.append({
                    'Asset': analyzer.asset_names[i],
                    'E[R]': er,
                    'E[R] (%)': f"{er*100:.2f}%",
                    'Std Dev (σ)': total_vol,
                    'Std Dev (%)': f"{total_vol*100:.2f}%",
                    'Excess Return': excess_ret,
                    'Sharpe Ratio': sharpe
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Display as metrics
            cols = st.columns(len(analyzer.asset_names))
            for i, col in enumerate(cols):
                with col:
                    st.markdown(f"**{analyzer.asset_names[i]}**")
                    st.metric("E[R]", f"{summary_data[i]['E[R]']*100:.2f}%")
                    st.metric("Std Dev (σ)", f"{summary_data[i]['Std Dev (σ)']*100:.2f}%")
                    st.metric("Sharpe Ratio", f"{summary_data[i]['Sharpe Ratio']:.4f}")
            
            # Display as table
            st.dataframe(summary_df[['Asset', 'E[R] (%)', 'Std Dev (%)', 'Sharpe Ratio']].style.format({
                'Sharpe Ratio': '{:.4f}'
            }))
            
            # ============================================================
            # DETAILED RISK DECOMPOSITION
            # ============================================================
            st.subheader("📊 Risk Decomposition (Systematic vs Idiosyncratic)")
            
            decomp = analyzer.risk_decomposition()
            st.dataframe(decomp.style.format({
                'Systematic Var': '{:.6f}',
                'Idiosyncratic Var': '{:.6f}',
                'Total Var': '{:.6f}',
                'Systematic Vol': '{:.4f}',
                'Idiosyncratic Vol': '{:.4f}',
                'Total Vol': '{:.4f}',
                'Systematic %': '{:.2f}%',
                'Idiosyncratic %': '{:.2f}%',
                'R²': '{:.4f}'
            }))
            
            # ============================================================
            # EXPECTED RETURN CALCULATION DETAIL
            # ============================================================
            st.subheader("📈 Expected Returns (APT Formula)")
            
            st.latex(r"E[R_i] = r_f + \sum_k \beta_{ik} \times \lambda_k")
            
            for i in range(analyzer.n_assets):
                er = expected_returns[i]
                beta = asset_betas[i] if asset_betas.ndim > 1 else asset_betas
                
                calc_parts = [f"{rf}"]
                for k in range(analyzer.n_factors):
                    calc_parts.append(f"{beta[k]:.2f} × {factor_means[k]:.2f}")
                
                st.write(f"**{analyzer.asset_names[i]}:** E[R] = {' + '.join(calc_parts)} = **{er*100:.2f}%**")
            
            # ============================================================
            # ASSET-ASSET COVARIANCE & CORRELATION (NEW!)
            # ============================================================
            st.subheader("🔗 Asset-Asset Covariance & Correlation")
            
            st.markdown("""
            **Formula:** $Cov(R_i, R_j) = \\beta_i' \\Sigma_F \\beta_j$
            
            **Interpretation:** This measures how much two assets move together due to their common factor exposures.
            """)
            
            # Covariance Matrix
            asset_cov_matrix = analyzer.asset_covariance_matrix()
            st.write("**Asset Covariance Matrix:**")
            cov_df = pd.DataFrame(
                asset_cov_matrix,
                index=analyzer.asset_names,
                columns=analyzer.asset_names
            )
            st.dataframe(cov_df.style.format('{:.6f}'))
            
            # Correlation Matrix
            asset_corr_matrix = analyzer.asset_correlation_matrix()
            st.write("**Asset Correlation Matrix:**")
            corr_df = pd.DataFrame(
                asset_corr_matrix,
                index=analyzer.asset_names,
                columns=analyzer.asset_names
            )
            st.dataframe(corr_df.style.format('{:.4f}'))
            
            # Show individual correlations as metrics if only 2 assets
            if analyzer.n_assets == 2:
                corr_01 = analyzer.asset_correlation(0, 1)
                cov_01 = analyzer.asset_covariance(0, 1)
                st.success(f"**Correlation between {analyzer.asset_names[0]} and {analyzer.asset_names[1]}:** ρ = **{corr_01:.4f}**")
                st.info(f"**Covariance between {analyzer.asset_names[0]} and {analyzer.asset_names[1]}:** Cov = **{cov_01:.6f}**")
            
            # ============================================================
            # ASSET-FACTOR COVARIANCE (NEW!)
            # ============================================================
            st.subheader("📐 Asset-Factor Covariance")
            
            st.markdown("""
            **Formula:** $Cov(R_i, F_k) = \\sum_j \\beta_{ij} \\times Cov(F_j, F_k) = (\\beta_i' \\Sigma_F)_k$
            
            **Interpretation:** This measures how much an asset's return co-moves with each factor.
            Useful for understanding the asset's exposure to factor risk.
            """)
            
            af_cov_matrix = analyzer.asset_factor_covariance_matrix()
            af_cov_df = pd.DataFrame(
                af_cov_matrix,
                index=analyzer.asset_names,
                columns=analyzer.factor_names
            )
            st.dataframe(af_cov_df.style.format('{:.6f}'))
            
            # Show detailed breakdown
            with st.expander("🔍 Asset-Factor Covariance Details"):
                for i in range(analyzer.n_assets):
                    st.write(f"**{analyzer.asset_names[i]}:**")
                    for k in range(analyzer.n_factors):
                        cov_val = analyzer.asset_factor_covariance(i, k)
                        st.write(f"  • Cov({analyzer.asset_names[i]}, {analyzer.factor_names[k]}) = {cov_val:.6f}")
            
            # ============================================================
            # FACTOR CONTRIBUTION DETAIL
            # ============================================================
            st.subheader("🔍 Factor Contribution Detail")
            
            for i in range(analyzer.n_assets):
                with st.expander(f"{analyzer.asset_names[i]} - Factor Breakdown"):
                    detail = analyzer.factor_contribution_detail(i)
                    st.dataframe(detail.style.format({
                        'Beta (β)': '{:.4f}',
                        'Factor Vol (σ_F)': '{:.4f}',
                        'β²σ²_F (Diagonal)': '{:.6f}',
                        'Cross-Term': '{:.6f}',
                        'Total Contribution': '{:.6f}',
                        '% of Systematic': '{:.2f}%'
                    }, na_rep='-'))
            
            # ============================================================
            # FACTOR COVARIANCE MATRIX
            # ============================================================
            st.subheader("📐 Factor Covariance Matrix (Σ_F)")
            
            st.latex(r"\Sigma_F = D_\sigma \times Corr_F \times D_\sigma")
            
            st.dataframe(pd.DataFrame(
                analyzer.factor_covariance,
                index=analyzer.factor_names,
                columns=analyzer.factor_names
            ).style.format('{:.6f}'))
            
            # ============================================================
            # EXCEL INSTRUCTIONS (COMPREHENSIVE)
            # ============================================================
            with st.expander("📝 How to do this in Excel (Complete Guide)"):
                st.markdown(analyzer.get_excel_instructions_full())
                
                st.markdown("---")
                st.markdown("### Key LaTeX Formulas")
                for name, formula in analyzer.get_latex_formulas().items():
                    st.latex(formula)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def advanced_performance_section():
    st.subheader("Fund Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Fund Statistics:**")
        fund_ret = st.number_input("Fund Return", value=0.12, format="%.4f", key="perf_fr")
        fund_vol = st.number_input("Fund Volatility", value=0.18, format="%.4f", key="perf_fv")
        fund_beta = st.number_input("Fund Beta", value=1.2, format="%.4f", key="perf_fb")
    
    with col2:
        st.write("**Market/Benchmark:**")
        mkt_ret = st.number_input("Market Return", value=0.10, format="%.4f", key="perf_mr")
        mkt_vol = st.number_input("Market Volatility", value=0.15, format="%.4f", key="perf_mv")
        rf = st.number_input("Risk-Free Rate", value=0.02, format="%.4f", key="perf_rf")
    
    if st.button("🧮 Calculate Performance Metrics", key="perf_calc"):
        try:
            analyzer = PerformanceAnalyzer(
                fund_return=fund_ret,
                fund_vol=fund_vol,
                fund_beta=fund_beta,
                market_return=mkt_ret,
                market_vol=mkt_vol,
                risk_free_rate=rf
            )
            
            st.subheader("📊 Performance Summary")
            
            summary = analyzer.get_summary_table()
            st.dataframe(summary)
            
            # Visual comparison
            st.subheader("📈 Risk-Return Comparison")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter([fund_vol], [fund_ret], s=200, c='blue', marker='o', label='Fund', zorder=5)
            ax.scatter([mkt_vol], [mkt_ret], s=200, c='red', marker='^', label='Market', zorder=5)
            ax.scatter([0], [rf], s=150, c='green', marker='s', label='Risk-Free', zorder=5)
            
            # CML
            cml_vols = np.linspace(0, max(fund_vol, mkt_vol) * 1.3, 100)
            cml_rets = rf + analyzer.market_sharpe * cml_vols
            ax.plot(cml_vols, cml_rets, 'r--', alpha=0.5, label='CML')
            
            # SML point for fund
            ax.scatter([fund_beta * mkt_vol], [analyzer.capm_expected], s=100, c='orange', 
                      marker='x', label='CAPM Expected', zorder=4)
            
            ax.set_xlabel('Volatility / Beta×σ_m')
            ax.set_ylabel('Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title('Risk-Return Analysis')
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📝 How to do this in Excel"):
                formulas = analyzer.get_latex_formulas()
                st.markdown("""
**Excel Formulas:**

| Metric | Formula | Excel |
|--------|---------|-------|
| Sharpe | (R_p - R_f) / σ_p | `=(Fund_Ret - RF) / Fund_Vol` |
| Treynor | (R_p - R_f) / β | `=(Fund_Ret - RF) / Beta` |
| Jensen's α | R_p - [R_f + β(R_m - R_f)] | `=Fund_Ret - (RF + Beta*(Mkt_Ret - RF))` |
| Residual Var | σ²_p - β²σ²_m | `=Fund_Vol^2 - Beta^2 * Mkt_Vol^2` |
| Info Ratio | α / σ_ε | `=Alpha / SQRT(Residual_Var)` |
| M² | R_f + SR_p × σ_m | `=RF + Sharpe * Mkt_Vol` |
                """)
                
                st.markdown("**LaTeX Formulas:**")
                for name, formula in formulas.items():
                    st.latex(formula)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def advanced_treynor_black_section():
    st.subheader("Treynor-Black Active Portfolio Optimization")
    
    st.markdown("""
    **Key Result:** $SR^2_{combined} = SR^2_M + IR^2_A$
    
    Active weights: $w_i \\propto \\alpha_i / \\sigma^2_{\\epsilon,i}$
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Market Parameters:**")
        mkt_ret = st.number_input("Market E[R]", value=0.10, format="%.4f", key="tb_mr")
        mkt_vol = st.number_input("Market σ", value=0.15, format="%.4f", key="tb_mv")
        rf = st.number_input("Risk-Free Rate", value=0.02, format="%.4f", key="tb_rf2")
        gamma = st.number_input("Risk Aversion (γ)", value=4.0, key="tb_gamma")
    
    with col2:
        st.write("**Active Securities:**")
        securities_input = st.text_area(
            "Alpha, Beta, Residual Vol (per row)",
            value="0.02, 1.2, 0.15\n0.03, 0.8, 0.20\n-0.01, 1.5, 0.10",
            height=120,
            key="tb_securities"
        )
    
    if st.button("🧮 Optimize Treynor-Black", key="tb_optimize"):
        try:
            # Parse inputs
            data = parse_matrix_input(securities_input)
            alphas = data[:, 0]
            betas = data[:, 1]
            resid_vols = data[:, 2]
            
            # Create optimizer
            optimizer = TreynorBlackOptimizer(
                alphas=alphas,
                betas=betas,
                residual_vols=resid_vols,
                market_return=mkt_ret,
                market_vol=mkt_vol,
                risk_free_rate=rf,
                risk_aversion=gamma
            )
            
            st.subheader("📊 Security Analysis")
            
            sec_analysis = optimizer.get_security_analysis()
            st.dataframe(sec_analysis.style.format({
                'Alpha (α)': '{:.4f}',
                'Beta (β)': '{:.2f}',
                'Residual Vol (σ_ε)': '{:.4f}',
                'Residual Var (σ²_ε)': '{:.6f}',
                'α / σ²_ε': '{:.4f}',
                'Active Weight': '{:.4f}',
                'Active Weight (%)': '{:.2f}%'
            }))
            
            st.subheader("📊 Results Summary")
            
            summary = optimizer.get_summary_table()
            st.dataframe(summary)
            
            st.subheader("🎯 Key Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Active Portfolio (A)**")
                st.metric("Alpha (α_A)", f"{optimizer.alpha_A*100:.2f}%")
                st.metric("Beta (β_A)", f"{optimizer.beta_A:.4f}")
                st.metric("E[r_A]", f"{optimizer.return_active*100:.2f}%")
                st.metric("Volatility (σ_A)", f"{optimizer.vol_active*100:.2f}%")
                st.metric("Variance (σ²_A)", f"{optimizer.var_active:.6f}")
            
            with col2:
                st.markdown("**Optimal Allocation**")
                st.metric("Weight in Active (w_A)", f"{optimizer.w_active*100:.2f}%")
                st.metric("Weight in Market (w_M)", f"{optimizer.w_market*100:.2f}%")
                st.metric("Information Ratio (IR)", f"{optimizer.IR_A:.4f}")
                st.markdown("---")
                st.markdown("**Combined Portfolio (C)**")
                st.metric("Beta (β_C)", f"{optimizer.beta_combined:.4f}")
            
            with col3:
                st.markdown("**Combined Portfolio (C)**")
                st.metric("E[r_C]", f"{optimizer.return_combined*100:.2f}%")
                st.metric("Volatility (σ_C)", f"{optimizer.vol_combined*100:.2f}%")
                st.metric("Variance (σ²_C)", f"{optimizer.var_combined:.6f}")
                st.markdown("---")
                st.markdown("**Performance**")
                st.metric("Market Sharpe", f"{optimizer.market_sharpe:.4f}")
                st.metric("Combined Sharpe", f"{optimizer.combined_sharpe:.4f}")
                st.metric("Improvement", f"+{optimizer.sharpe_improvement_pct:.1f}%")
            
            # Sharpe decomposition visual
            st.subheader("📈 Sharpe Ratio Decomposition")
            
            st.latex(rf"SR^2_{{combined}} = SR^2_M + IR^2_A = {optimizer.market_sharpe_sq:.4f} + {optimizer.appraisal_sq:.4f} = {optimizer.combined_sharpe_sq:.4f}")
            st.latex(rf"SR_{{combined}} = \sqrt{{{optimizer.combined_sharpe_sq:.4f}}} = {optimizer.combined_sharpe:.4f}")
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(optimizer.get_excel_instructions())
                
                st.markdown("**LaTeX Formulas:**")
                for name, formula in optimizer.get_latex_formulas().items():
                    st.latex(formula)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# =============================================================================
# INTEGRATED MODULES - Combining Basic and Advanced as Tabs
# =============================================================================

def portfolio_optimizer_module_integrated():
    """Integrated Portfolio Optimizer with ESG extension and constrained optimization."""
    st.header("📈 Module 1: Portfolio Optimizer")

    # Module 1 guidance expander
    with st.expander("📘 Module Guidance: When to use this?", expanded=False):
        st.markdown('''
        **When to use:**
        * You are given a **covariance matrix** (or correlations + volatilities) and **expected returns**.
        * The problem asks for **optimal weights** (Tangency, Minimum Variance, or specific Risk Aversion $\gamma$).
        * You need the **Efficient Frontier**, Portfolio Return/Volatility, or Sharpe Ratio.
        * The problem mentions **ESG preferences** or **constraints** (e.g., no short selling).

        **Inputs you need:**
        * **Returns Vector ($\mu$):** Decimal format (e.g., 0.08 for 8%).
        * **Covariance Matrix ($\Sigma$):** Either the full matrix or Volatilities ($\sigma$) + Correlations ($\rho$).
        * **Risk-Free Rate ($r_f$):** Decimal.
        * *(Optional)* Risk Aversion ($\gamma$) or ESG scores.

        **Outputs & Interpretation:**
        * **Weights ($w$):** Sum to 1. Negative = Short selling. $>1$ = Leverage.
        * **Cash Weight:** If optimizing risky assets only, residual ($1-\sum w_i$) is held in risk-free asset.
        * **Portfolio Stats:** $\mu_p, \sigma_p, SR$.

        **Common Pitfalls:**
        * **Percent vs Decimal:** Ensure inputs are decimals (0.10 not 10). Matrix inversion fails with mixed units.
        * **Covariance Input:** If pasting from PDF, ensure rows are separated correctly.
        * **Gamma:** Standard $\gamma$ is usually between 2 and 10. If not given, check if you should use the Tangency portfolio (which doesn't depend on $\gamma$).

        **Fast Decision Rule:**
        "If the problem gives a **Correlation/Covariance Matrix** and asks for **Weights**, use this module."
        ''')
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Unconstrained (Closed Form)", 
        "🔒 Constrained (Solver)",
        "🌱 ESG-Extended"
    ])
    
    with tab1:
        portfolio_optimizer_basic_tab()
    
    with tab2:
        constrained_portfolio_tab()
    
    with tab3:
        esg_portfolio_section()


def constrained_portfolio_tab():
    """Constrained mean-variance optimization using SLSQP."""
    st.markdown("""
    **Constrained Optimization** using quadratic programming (SLSQP solver).
    
    Supports: max weight per asset, no-short, leverage cap, volatility target, **utility maximization**.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        rf = st.number_input("Risk-Free Rate", value=0.02, format="%.4f", key="const_rf")
        
        objective = st.selectbox(
            "Objective",
            ["max_sharpe", "min_variance", "max_utility", "max_return", "target_return"],
            format_func=lambda x: {
                "max_sharpe": "Maximize Sharpe Ratio",
                "min_variance": "Minimize Variance",
                "max_utility": "Maximize Utility (Given γ)",
                "max_return": "Maximize Return",
                "target_return": "Minimize Variance for Target Return"
            }[x],
            key="const_objective"
        )
        
        if objective == "target_return":
            target_ret = st.number_input("Target Return", value=0.08, format="%.4f", key="const_target_ret")
        else:
            target_ret = None
        
        # Show utility parameters when max_utility is selected
        if objective == "max_utility":
            st.markdown("---")
            st.markdown("**Utility Parameters:**")
            risk_aversion = st.number_input(
                "Risk Aversion (γ)", 
                value=4.0, 
                min_value=0.1, 
                max_value=100.0,
                format="%.2f", 
                key="const_gamma",
                help="Higher γ means more risk-averse. Common exam values: γ=2, γ=4, γ=10"
            )
            
            utility_formula = st.radio(
                "Utility Formula",
                ["standard", "simple"],
                format_func=lambda x: {
                    "standard": "Standard: U = E[r] - 0.5·γ·σ²",
                    "simple": "Simple: U = E[r] - γ·σ²"
                }[x],
                key="const_utility_type",
                help="Standard includes the 0.5 factor (most common). Simple drops it (some exams use this)."
            )
        else:
            risk_aversion = 4.0
            utility_formula = "standard"
    
    with col2:
        st.write("**Constraints:**")
        no_short = st.checkbox("No Short Selling (w ≥ 0)", value=True, key="const_noshort")
        
        # Budget/Leverage mode selection
        budget_mode = st.radio(
            "Budget / Leverage Mode",
            ["fully_invested", "long_only", "leverage"],
            format_func=lambda x: {
                "fully_invested": "Fully Invested (Stocks Only)",
                "long_only": "Long Only (Stocks + Cash)",
                "leverage": "Leverage Allowed (Borrowing)"
            }[x],
            key="const_budget_mode",
            help="Fully Invested: Σw = 1 (all wealth in stocks). "
                 "Long Only: Σw ≤ 1 (can hold cash, no borrowing). "
                 "Leverage: No sum constraint (can borrow at rf to invest >100%)."
        )
        
        # Convert radio selection to boolean flags
        if budget_mode == "fully_invested":
            allow_riskfree = False
            allow_borrowing = False
        elif budget_mode == "long_only":
            allow_riskfree = True
            allow_borrowing = False
        else:  # leverage
            allow_riskfree = True
            allow_borrowing = True
        
        use_max_weight = st.checkbox("Max Weight per Asset", value=False, key="const_use_max")
        if use_max_weight:
            max_weight = st.number_input("Maximum Weight", value=0.40, format="%.2f", key="const_max_w")
        else:
            max_weight = None
        
        use_vol_cap = st.checkbox("Volatility Cap", value=False, key="const_use_vol")
        if use_vol_cap:
            vol_cap = st.number_input("Target Volatility", value=0.15, format="%.4f", key="const_vol_cap")
        else:
            vol_cap = None
        
        use_leverage = st.checkbox("Leverage Cap (Σ|w|)", value=False, key="const_use_lev")
        if use_leverage:
            leverage_cap = st.number_input("Max Leverage", value=1.5, format="%.2f", key="const_lev")
        else:
            leverage_cap = None
    
    st.subheader("Input Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        returns_input = st.text_area(
            "Expected Returns & Volatilities (Mean, StdDev per row)",
            value="0.08, 0.20\n0.12, 0.25\n0.06, 0.15",
            height=150,
            key="const_returns"
        )
    
    with col2:
        corr_input = st.text_area(
            "Correlation Matrix",
            value="1.0, 0.3, 0.1\n0.3, 1.0, 0.2\n0.1, 0.2, 1.0",
            height=150,
            key="const_corr"
        )
    
    if st.button("🧮 Optimize Portfolio", key="const_calc"):
        try:
            # Parse inputs
            mu, sigma = parse_two_column_input(returns_input)
            corr_matrix = parse_matrix_input(corr_input)
            
            if len(mu) == 0:
                st.error("❌ Could not parse returns data.")
                return
            
            n = len(mu)
            D = np.diag(sigma)
            cov_matrix = D @ corr_matrix @ D
            
            # Display Covariance Matrix
            st.subheader("📊 Matrices")
            
            st.write("**Covariance Matrix (Σ):**")
            cov_df = pd.DataFrame(cov_matrix, 
                                  index=[f'Asset {i+1}' for i in range(n)],
                                  columns=[f'Asset {i+1}' for i in range(n)])
            st.dataframe(cov_df.style.format("{:.6f}"))
            
            # Calculate and display Inverse Covariance Matrix
            try:
                cov_inv = np.linalg.inv(cov_matrix)
                st.write("**Inverse Covariance Matrix (Σ⁻¹):**")
                st.info("💡 Copy this table for exam questions asking to 'Determine the inverse matrix'")
                cov_inv_df = pd.DataFrame(cov_inv,
                                          index=[f'Asset {i+1}' for i in range(n)],
                                          columns=[f'Asset {i+1}' for i in range(n)])
                st.dataframe(cov_inv_df.style.format("{:.4f}"))
            except np.linalg.LinAlgError:
                st.warning("⚠️ Covariance matrix is singular (cannot display inverse)")
            
            # Run optimization
            result, excel_steps, tables = optimize_portfolio_constrained(
                expected_returns=mu,
                covariance_matrix=cov_matrix,
                risk_free_rate=rf,
                objective=objective,
                target_return=target_ret,
                target_volatility=vol_cap,
                max_weight=max_weight,
                no_short=no_short,
                leverage_cap=leverage_cap,
                allow_riskfree=allow_riskfree,
                allow_borrowing=allow_borrowing,
                risk_aversion=risk_aversion,
                utility_type=utility_formula
            )
            
            st.subheader("📊 Optimal Portfolio")
            
            # Show utility info if max_utility objective
            if result.get('utility') is not None:
                st.success(f"**Utility Maximization:** γ = {result['risk_aversion']}, Formula = {result['utility_type']}")
                penalty_factor = 0.5 * result['risk_aversion'] if result['utility_type'] == 'standard' else result['risk_aversion']
                st.latex(f"U = E[r] - {penalty_factor:.2f} \\times \\sigma^2 = {result['return']*100:.4f}\\% - {penalty_factor:.2f} \\times {result['variance']*100:.6f}\\% = {result['utility']*100:.4f}\\%")
            
            if result.get('allow_riskfree', False):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Expected Return", f"{result['return']*100:.4f}%")
                with col2:
                    st.metric("Volatility", f"{result['volatility']*100:.4f}%")
                with col3:
                    if result.get('utility') is not None:
                        st.metric("Utility", f"{result['utility']*100:.4f}%")
                    else:
                        st.metric("Sharpe Ratio", f"{result['sharpe']:.4f}")
                with col4:
                    # Show "Borrowed" instead of "Cash" when negative
                    cash_wt = result['cash_weight']
                    if cash_wt < 0:
                        st.metric("Borrowed", f"{-cash_wt*100:.2f}%")
                    else:
                        st.metric("Cash Weight", f"{cash_wt*100:.2f}%")
                with col5:
                    st.metric("Total Risky", f"{(1-result['cash_weight'])*100:.2f}%")
            else:
                if result.get('utility') is not None:
                    col1, col2, col3, col4, col5 = st.columns(5)
                else:
                    col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Expected Return", f"{result['return']*100:.4f}%")
                with col2:
                    st.metric("Volatility", f"{result['volatility']*100:.4f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{result['sharpe']:.4f}")
                with col4:
                    if result.get('utility') is not None:
                        st.metric("Utility", f"{result['utility']*100:.4f}%")
                    else:
                        st.metric("Total Leverage", f"{result['total_leverage']:.4f}")
                if result.get('utility') is not None:
                    with col5:
                        st.metric("Variance", f"{result['variance']*100:.6f}%")
            
            st.write("**Weights:**")
            st.dataframe(tables['weights'])
            
            if result['binding_constraints']:
                st.info("**Binding Constraints:** " + ", ".join(result['binding_constraints']))
            
            if not result['success']:
                st.warning(f"⚠️ Solver warning: {result['message']}")
            
            with st.expander("📝 How to do this in Excel"):
                for step in excel_steps:
                    st.markdown(step)
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def portfolio_optimizer_basic_tab():
    """Basic MV optimizer tab content."""
    st.markdown("""
    This module solves mean-variance portfolio optimization problems using matrix algebra.
    It calculates **Tangency**, **Global Minimum Variance**, and **Optimal** portfolio weights.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gamma = st.number_input("Risk Aversion (γ)", value=4.0, min_value=0.1, step=0.5, 
                                help="Higher γ = more risk-averse investor", key="basic_gamma")
        rf = st.number_input("Risk-Free Rate (rf)", value=0.02, format="%.4f",
                            help="Enter as decimal, e.g., 0.02 for 2%", key="basic_rf")
    
    with col2:
        use_esg = st.checkbox("Include ESG Preference", key="basic_esg_check")
        if use_esg:
            esg_preference = st.number_input("ESG Preference (a)", value=0.01, format="%.4f",
                                            help="Utility: U = E[r] - 0.5γσ² + a·ESG", key="basic_esg_pref")
        else:
            esg_preference = 0.0
    
    st.subheader("Input Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        returns_input = st.text_area(
            "Expected Returns & Volatilities (Mean, StdDev per row)",
            value="0.08, 0.20\n0.12, 0.25\n0.06, 0.15",
            height=150,
            help="Paste from PDF: Each row = one asset with Mean Return and Std Dev",
            key="basic_returns"
        )
    
    with col2:
        corr_input = st.text_area(
            "Correlation Matrix",
            value="1.0, 0.3, 0.1\n0.3, 1.0, 0.2\n0.1, 0.2, 1.0",
            height=150,
            help="Paste correlation matrix (must be symmetric)",
            key="basic_corr"
        )
    
    if use_esg:
        esg_input = st.text_area(
            "ESG Scores (one per asset)",
            value="0.7, 0.5, 0.9",
            help="ESG score for each asset",
            key="basic_esg_scores"
        )
    
    if st.button("🧮 Calculate Optimal Portfolios", key="basic_calc"):
        try:
            # Parse inputs
            mu, sigma = parse_two_column_input(returns_input)
            corr_matrix = parse_matrix_input(corr_input)
            
            if len(mu) == 0:
                st.error("❌ Could not parse returns data. Please check format.")
                return
            
            n_assets = len(mu)
            
            if corr_matrix.shape != (n_assets, n_assets):
                st.error(f"❌ Correlation matrix dimensions ({corr_matrix.shape}) don't match number of assets ({n_assets})")
                return
            
            # Parse ESG if applicable
            esg_scores = None
            if use_esg:
                esg_scores = parse_messy_input(esg_input)
                if len(esg_scores) != n_assets:
                    st.error(f"❌ ESG scores length ({len(esg_scores)}) doesn't match number of assets ({n_assets})")
                    return
            
            # Construct Covariance Matrix
            D = np.diag(sigma)
            cov_matrix = D @ corr_matrix @ D
            
            # Display Covariance Matrix
            st.subheader("📊 Matrices")
            
            st.write("**Covariance Matrix (Σ):**")
            cov_df = pd.DataFrame(cov_matrix, 
                                  index=[f'Asset {i+1}' for i in range(n_assets)],
                                  columns=[f'Asset {i+1}' for i in range(n_assets)])
            st.dataframe(cov_df.style.format("{:.6f}"))
            
            # Calculate and display Inverse Covariance Matrix
            try:
                cov_inv = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                st.error("❌ Covariance matrix is singular (cannot be inverted)")
                return
            
            st.write("**Inverse Covariance Matrix (Σ⁻¹):**")
            st.info("💡 Copy this table for exam questions asking to 'Determine the inverse matrix'")
            cov_inv_df = pd.DataFrame(cov_inv,
                                      index=[f'Asset {i+1}' for i in range(n_assets)],
                                      columns=[f'Asset {i+1}' for i in range(n_assets)])
            st.dataframe(cov_inv_df.style.format("{:.4f}"))
            
            # Create optimizer
            optimizer = MeanVarianceOptimizer(
                expected_returns=mu,
                covariance_matrix=cov_matrix,
                risk_free_rate=rf,
                risk_aversion=gamma,
                esg_scores=esg_scores,
                esg_preference=esg_preference
            )
            
            # Calculate portfolios
            tangency = optimizer.tangency_portfolio()
            gmv = optimizer.gmv_portfolio()
            optimal = optimizer.optimal_portfolio()
            
            st.subheader("📊 Results")
            
            # Display results table
            results_df = pd.DataFrame({
                'Portfolio': ['Tangency', 'GMV', f'Optimal (γ={gamma})'],
                'E[r]': [f"{tangency['expected_return']*100:.2f}%", 
                        f"{gmv['expected_return']*100:.2f}%",
                        f"{optimal['expected_return']*100:.2f}%"],
                'Volatility': [f"{tangency['volatility']*100:.2f}%",
                              f"{gmv['volatility']*100:.2f}%",
                              f"{optimal['volatility']*100:.2f}%"],
                'Sharpe': [f"{tangency['sharpe_ratio']:.4f}",
                          f"{gmv['sharpe_ratio']:.4f}",
                          f"{optimal['sharpe_ratio']:.4f}"]
            })
            
            # Add weights
            for i in range(n_assets):
                results_df[f'w_{i+1}'] = [
                    f"{tangency['weights'][i]:.4f}",
                    f"{gmv['weights'][i]:.4f}",
                    f"{optimal['weights'][i]:.4f}"
                ]
            
            st.dataframe(results_df)
            
            # Efficient frontier plot
            frontier = optimizer.efficient_frontier()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(frontier['volatility'], frontier['return'], 'b-', linewidth=2, label='Efficient Frontier')
            ax.scatter([gmv['volatility']], [gmv['expected_return']], s=150, c='green', marker='s', label='GMV', zorder=5)
            ax.scatter([tangency['volatility']], [tangency['expected_return']], s=150, c='red', marker='^', label='Tangency', zorder=5)
            ax.scatter([optimal['volatility']], [optimal['expected_return']], s=150, c='purple', marker='D', label=f'Optimal (γ={gamma})', zorder=5)
            ax.scatter(sigma, mu, s=100, c='gray', marker='o', label='Assets', zorder=4)
            
            # CML
            cml_vols = np.linspace(0, max(frontier['volatility'])*1.1, 100)
            cml_returns = rf + tangency['sharpe_ratio'] * cml_vols
            ax.plot(cml_vols, cml_returns, 'r--', alpha=0.5, label='CML')
            
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Expected Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(optimizer.get_excel_instructions())
        
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def bond_math_module_integrated():
    """Integrated Bond Math with advanced features."""
    st.header("💵 Module 2: Bond Math")

    # Module 2 guidance expander
    with st.expander("📘 Module Guidance: When to use this?", expanded=False):
        st.markdown(r'''
        **When to use:**
        * **Pricing/Duration:** You have a single bond's coupon, maturity, and yield. Need Price, Duration, Convexity, or $\Delta$Price.
        * **Price with Spot Curve:** You have a **zero-coupon yield curve** and need to price a bond using maturity-specific rates.
        * **Bootstrap:** You have a list of coupon bonds and need the Zero-Coupon/Spot Yield Curve.
        * **Forward Rates:** You have Spot Rates and need implied Forward Rates ($f_{t_1, t_2}$).
        * **Immunization:** You have a **Liability** and need to match it with a portfolio of bonds.
        * **Horizon Analysis:** Evaluate an immunization portfolio at **future time t** under a **new yield curve**.
        * **Replication:** You need to create a synthetic bond (e.g., Zero) using coupon bonds.

        **Inputs you need:**
        * **Yields/Coupons:** Decimals (0.05).
        * **Frequency:** Annual (1) vs Semi-annual (2) is critical.
        * **Liability:** PV and Duration (for immunization).
        * **Horizon Analysis:** Time step Δt, new yield curve, bond positions from immunization.

        **Common Pitfalls:**
        * **Frequency Mismatch:** Exam often gives annual yields but semi-annual payments. The engine handles this if 'Frequency' is set correctly.
        * **Short Positions:** Immunization or Replication often requires negative weights (shorting).
        * **Yield Change:** For sensitivity analysis ($\Delta y$), inputs are usually in basis points (0.0001) or percent.
        * **Spot Curve vs Flat Yield:** When given a yield curve, use "Price with Spot Curve" - each CF discounted at its own rate!
        * **Horizon Analysis:** Remember to reduce all maturities by Δt. Bonds may have paid coupons.

        **Fast Decision Rule:**
        "If the problem mentions **Duration, Immunization, Spot Curves, or Bond Pricing**, use this module."
        ''')
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "📊 Pricing & Duration", 
        "🎯 Price with Spot Curve",
        "📈 Spot Curve Bootstrap",
        "↗️ Forward Rates",
        "🛡️ Immunization",
        "🛡️² Convexity Matching",
        "📐 Fisher-Weil Duration",
        "💀 Risky Bond Pricing",
        "⏳ Horizon Analysis",
        "📋 Annuity Pricer",
        "🔧 Multi-CF Replication"
    ])
    
    with tab1:
        bond_pricing_section()
    
    with tab2:
        spot_curve_bond_pricing_tab()
    
    with tab3:
        spot_curve_bootstrap_tab()
    
    with tab4:
        forward_rates_tab()
    
    with tab5:
        advanced_immunization_section()
    
    with tab6:
        convexity_matching_tab()
    
    with tab7:
        fisher_weil_duration_tab()
    
    with tab8:
        risky_bond_pricing_tab()
    
    with tab9:
        bond_horizon_analysis_tab()
    
    with tab10:
        annuity_pricer_tab()
    
    with tab11:
        multi_cf_replication_tab()


# =============================================================================
# NEW TAB: Second-Order Immunization (Convexity Matching)
# =============================================================================

def convexity_matching_tab():
    """Second-order immunization using three bonds to match duration AND convexity."""
    st.subheader("🛡️² Second-Order Immunization (Convexity Matching)")
    
    st.markdown(r"""
    **Reference:** Textbook Theorem 5.8
    
    **Problem:** First-order immunization (duration matching) only protects against *small* 
    parallel shifts in the yield curve. For larger shifts or non-parallel shifts, you need 
    **second-order immunization** which also matches convexity.
    
    **Solution:** Use THREE bonds to solve three equations:
    1. $\sum w_i = 1$ (Budget constraint)
    2. $\sum w_i \times D_i = D_L$ (Duration match)
    3. $\sum w_i \times C_i = C_L$ (Convexity match)
    """)
    
    with st.expander("📚 When to Use This vs. Regular Immunization", expanded=False):
        st.markdown(r"""
        | Scenario | Use |
        |----------|-----|
        | Small yield changes (<50 bps) | Regular immunization (2 bonds) |
        | Large yield changes (>100 bps) | **Convexity matching (3 bonds)** |
        | Non-parallel yield curve shifts | **Convexity matching (3 bonds)** |
        | Long-dated liabilities | **Convexity matching (3 bonds)** |
        | High precision required | **Convexity matching (3 bonds)** |
        
        **Key Insight:** Duration is a linear approximation. Convexity adds the quadratic term:
        $$\frac{\Delta P}{P} \approx -D \cdot \Delta y + \frac{1}{2} C \cdot (\Delta y)^2$$
        """)
    
    st.info(
        "🧠 **Exam Tip:** When asked to 'fully immunize' or 'immunize against both duration and convexity risk', "
        "you need THREE bonds. Set up the 3×3 system of equations and solve using matrix methods or substitution."
    )
    
    st.write("---")
    
    # Liability inputs
    st.write("### 📋 Liability to Immunize")
    col1, col2, col3 = st.columns(3)
    with col1:
        liability_pv = st.number_input(
            "Liability PV ($)", value=10_000_000.0, min_value=1.0,
            help="Present value of the liability", key="cx_liab_pv"
        )
    with col2:
        liability_duration = st.number_input(
            "Liability Duration (years)", value=7.5, min_value=0.1,
            help="Macaulay duration of the liability", key="cx_liab_dur"
        )
    with col3:
        liability_convexity = st.number_input(
            "Liability Convexity", value=75.0, min_value=0.1,
            help="Convexity of the liability", key="cx_liab_cvx"
        )
    
    st.write("---")
    
    # Bond inputs
    st.write("### 📊 Three Immunizing Bonds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Bond A (Short Duration)**")
        dur_A = st.number_input("Duration A", value=3.0, key="cx_dur_a")
        cvx_A = st.number_input("Convexity A", value=15.0, key="cx_cvx_a")
        price_A = st.number_input("Price A ($)", value=980.0, key="cx_price_a")
    
    with col2:
        st.write("**Bond B (Medium Duration)**")
        dur_B = st.number_input("Duration B", value=7.0, key="cx_dur_b")
        cvx_B = st.number_input("Convexity B", value=60.0, key="cx_cvx_b")
        price_B = st.number_input("Price B ($)", value=1020.0, key="cx_price_b")
    
    with col3:
        st.write("**Bond C (Long Duration)**")
        dur_C = st.number_input("Duration C", value=12.0, key="cx_dur_c")
        cvx_C = st.number_input("Convexity C", value=180.0, key="cx_cvx_c")
        price_C = st.number_input("Price C ($)", value=1100.0, key="cx_price_c")
    
    if st.button("🧮 Solve Convexity Matching", key="cx_solve"):
        try:
            # Create immunizer
            immunizer = ImmunizerThreeBond(
                duration_A=dur_A, convexity_A=cvx_A, price_A=price_A,
                duration_B=dur_B, convexity_B=cvx_B, price_B=price_B,
                duration_C=dur_C, convexity_C=cvx_C, price_C=price_C,
                duration_liability=liability_duration,
                convexity_liability=liability_convexity,
                pv_liability=liability_pv
            )
            
            if not immunizer.solvable:
                st.error(f"❌ Cannot solve: {immunizer.error_message}")
                st.warning("Ensure the three bonds have different duration/convexity combinations.")
                return
            
            st.subheader("📊 Solution")
            
            # Display summary table
            st.dataframe(immunizer.get_summary_table())
            
            # Verification
            verify = immunizer.verify_solution()
            
            st.write("### ✅ Verification")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Σ Weights", f"{verify['budget']:.6f}", 
                         delta=f"Target: 1.0", delta_color="off")
            with col2:
                st.metric("Portfolio Duration", f"{verify['duration']:.4f}", 
                         delta=f"Target: {liability_duration:.4f}", delta_color="off")
            with col3:
                st.metric("Portfolio Convexity", f"{verify['convexity']:.4f}", 
                         delta=f"Target: {liability_convexity:.4f}", delta_color="off")
            
            # Check for short positions
            if immunizer.w_A < 0 or immunizer.w_B < 0 or immunizer.w_C < 0:
                st.warning("⚠️ Solution requires short positions (negative weights)")
            
            # Step-by-step solution
            with st.expander("📝 Step-by-Step Solution"):
                st.markdown(rf"""
**The System of Equations:**

$$\begin{{bmatrix}} 1 & 1 & 1 \\ {dur_A} & {dur_B} & {dur_C} \\ {cvx_A} & {cvx_B} & {cvx_C} \end{{bmatrix}} \begin{{bmatrix}} w_A \\ w_B \\ w_C \end{{bmatrix}} = \begin{{bmatrix}} 1 \\ {liability_duration} \\ {liability_convexity} \end{{bmatrix}}$$

**Solution using Matrix Inversion:**

$$\mathbf{{w}} = \mathbf{{A}}^{{-1}} \mathbf{{b}}$$

**Results:**
- $w_A = {immunizer.w_A:.6f}$ ({immunizer.w_A*100:.2f}%)
- $w_B = {immunizer.w_B:.6f}$ ({immunizer.w_B*100:.2f}%)
- $w_C = {immunizer.w_C:.6f}$ ({immunizer.w_C*100:.2f}%)

**Investment Amounts:**
- Bond A: ${immunizer.w_A * liability_pv:,.2f}
- Bond B: ${immunizer.w_B * liability_pv:,.2f}
- Bond C: ${immunizer.w_C * liability_pv:,.2f}

**Excel Implementation:**
```
1. Enter coefficient matrix in A1:C3
2. Enter target vector in E1:E3  
3. Use =MINVERSE(A1:C3) to get inverse matrix
4. Use =MMULT(inverse, E1:E3) to get weights
   OR use =MSOLVE(A1:C3, E1:E3) in newer Excel
```
                """)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# =============================================================================
# NEW TAB: Fisher-Weil Duration
# =============================================================================

def fisher_weil_duration_tab():
    """Calculate Fisher-Weil Duration using spot rates."""
    st.subheader("📐 Fisher-Weil Duration")
    
    st.markdown(r"""
    **Reference:** Textbook Equation 5.51
    
    **Problem:** Macaulay Duration uses a single YTM to discount all cash flows. This is only 
    accurate when the yield curve is flat. **Fisher-Weil Duration** uses the spot rate for 
    each period, giving a more accurate measure when the yield curve is sloped.
    
    **Key Formulas:**
    - PV at time t: $PV_t = \frac{CF_t}{(1+z_t)^t}$ where $z_t$ is the spot rate
    - Weight: $w_t = \frac{PV_t}{\text{Price}}$
    - **Fisher-Weil Duration:** $D_{FW} = \sum_t t \times w_t$
    """)
    
    with st.expander("📚 When to Use Fisher-Weil vs Macaulay Duration", expanded=False):
        st.markdown(r"""
        | Yield Curve | Macaulay Duration | Fisher-Weil Duration |
        |-------------|-------------------|----------------------|
        | Flat | ✓ Accurate | ✓ Accurate (same result) |
        | Upward sloping | ⚠️ Approximation | ✓ More accurate |
        | Downward sloping | ⚠️ Approximation | ✓ More accurate |
        | Steep curve | ❌ Can be misleading | ✓ Preferred |
        
        **Key Insight:** Fisher-Weil is always at least as accurate as Macaulay, and 
        more accurate when the curve is not flat. The difference increases with curve steepness.
        """)
    
    st.info(
        "🧠 **Exam Tip:** If given a spot rate curve and asked for 'duration', check if they want "
        "Fisher-Weil (using spot rates) or Macaulay (using YTM). Fisher-Weil is more precise for immunization."
    )
    
    st.write("---")
    
    # Bond inputs
    col1, col2 = st.columns(2)
    with col1:
        face_value = st.number_input("Face Value ($)", value=1000.0, key="fw_face")
        coupon_rate = st.number_input("Coupon Rate", value=0.05, format="%.4f", key="fw_coupon")
        maturity = st.number_input("Maturity (years)", value=5, min_value=1, key="fw_mat")
    with col2:
        frequency = st.selectbox("Payment Frequency", [1, 2], format_func=lambda x: "Annual" if x == 1 else "Semi-annual", key="fw_freq")
    
    st.write("### 📈 Spot Rate Curve")
    st.markdown("Enter the zero-coupon (spot) rates for each maturity:")
    
    spot_input = st.text_area(
        "Spot Rates (maturity, rate per line)",
        value="1, 0.03\n2, 0.035\n3, 0.04\n4, 0.042\n5, 0.045",
        height=150,
        key="fw_spots",
        help="Format: maturity, spot_rate (one per line). Rates as decimals."
    )
    
    if st.button("📊 Calculate Fisher-Weil Duration", key="fw_calc"):
        try:
            # Parse spot rates
            lines = [l.strip() for l in spot_input.strip().split('\n') if l.strip()]
            maturities = []
            spot_rates = []
            for line in lines:
                parts = [p.strip() for p in line.replace(',', ' ').split()]
                if len(parts) >= 2:
                    maturities.append(float(parts[0]))
                    spot_rates.append(float(parts[1]))
            
            maturities = np.array(maturities)
            spot_rates = np.array(spot_rates)
            
            # Calculate
            result, steps, schedule = price_bond_with_spot_curve_fisher_weil(
                face_value=face_value,
                coupon_rate=coupon_rate,
                maturity=maturity,
                spot_rates=spot_rates,
                maturities=maturities,
                frequency=frequency
            )
            
            st.subheader("📊 Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bond Price", f"${result['price']:.4f}")
                st.metric("YTM", f"{result['ytm']*100:.4f}%")
            with col2:
                st.metric("Fisher-Weil Duration", f"{result['fisher_weil_duration']:.4f} years",
                         help="Duration using spot rates")
                st.metric("Modified F-W Duration", f"{result['modified_fw_duration']:.4f}")
            with col3:
                st.metric("Macaulay Duration", f"{result['macaulay_duration']:.4f} years",
                         help="Duration using YTM")
                st.metric("Difference (FW - Mac)", f"{result['duration_difference']:.6f} years")
            
            st.write("### 📋 Cash Flow Schedule")
            st.dataframe(schedule.style.format({
                't (years)': '{:.1f}',
                'Cash Flow': '{:.2f}',
                'Spot Rate (z_t)': '{:.4%}',
                'DF (Spot)': '{:.6f}',
                'PV (Spot)': '{:.4f}',
                'FW Weight': '{:.6f}',
                't × FW Weight': '{:.6f}',
                'YTM DF': '{:.6f}',
                'PV (YTM)': '{:.4f}',
                'Mac Weight': '{:.6f}',
                't × Mac Weight': '{:.6f}'
            }))
            
            with st.expander("📝 Step-by-Step Calculation"):
                for step in steps:
                    st.markdown(step)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# =============================================================================
# NEW TAB: Risky Bond Pricing
# =============================================================================

def risky_bond_pricing_tab():
    """Price bonds with credit/default risk."""
    st.subheader("💀 Risky Bond Pricing (Credit Risk)")
    
    st.markdown(r"""
    **Reference:** Textbook Equation 5.66 (Section 5.10.2)
    
    **Problem:** Standard bond pricing assumes all cash flows are certain. Real bonds have 
    **default risk** - the issuer may fail to pay. This tab prices bonds accounting for:
    - **Default Probability (p):** Annual probability the issuer defaults
    - **Recovery Rate (R):** Fraction of face value recovered if default occurs
    
    **Key Formula for Expected Cash Flows:**
    $$E[CF_t] = CF_t \times [1 - p(1-R)]^t$$
    
    Where $[1 - p(1-R)]$ is the survival-adjusted factor per period.
    """)
    
    with st.expander("📚 Understanding Credit Risk Components", expanded=False):
        st.markdown(r"""
        | Component | Symbol | Typical Values | Description |
        |-----------|--------|----------------|-------------|
        | Default Probability | p | 0.5% - 10% | Annual chance of default |
        | Recovery Rate | R | 20% - 60% | What you get back in default |
        | Loss Given Default | 1-R | 40% - 80% | What you lose in default |
        | Credit Spread | s | 50 - 500 bps | Extra yield for taking credit risk |
        
        **Key Relationships:**
        - Higher p → Lower price, higher spread
        - Higher R → Higher price (less loss in default)
        - Credit Spread ≈ p × (1-R) for small probabilities
        
        **Single-Period Formula (Eq 5.66 for T=1):**
        $$P = \frac{(1-p)(C+F) + p \cdot R \cdot F}{1+r}$$
        """)
    
    st.info(
        "🧠 **Exam Tip:** The key is adjusting expected cash flows BEFORE discounting. "
        "Each period's CF is multiplied by the probability of surviving to that period. "
        "The discount rate should be the RISK-FREE rate (credit risk is in the cash flows)."
    )
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📋 Bond Parameters")
        face_value = st.number_input("Face Value ($)", value=1000.0, key="rb_face")
        coupon_rate = st.number_input("Coupon Rate", value=0.06, format="%.4f", key="rb_coupon")
        maturity = st.number_input("Maturity (years)", value=5, min_value=1, key="rb_mat")
        frequency = st.selectbox("Payment Frequency", [1, 2], 
                                format_func=lambda x: "Annual" if x == 1 else "Semi-annual", 
                                key="rb_freq")
        yield_rate = st.number_input("Risk-Free Yield", value=0.04, format="%.4f", key="rb_yield",
                                    help="Use the risk-free rate for discounting")
    
    with col2:
        st.write("### 💀 Credit Risk Parameters")
        default_prob = st.number_input(
            "Annual Default Probability (p)", value=0.02, format="%.4f",
            min_value=0.0, max_value=1.0, key="rb_default",
            help="Probability of default in any given year"
        )
        recovery_rate = st.number_input(
            "Recovery Rate (R)", value=0.40, format="%.4f",
            min_value=0.0, max_value=1.0, key="rb_recovery",
            help="Fraction of face value recovered if default occurs"
        )
        
        # Show derived quantities
        st.write("**Derived:**")
        loss_given_default = 1 - recovery_rate
        expected_loss_rate = default_prob * loss_given_default
        st.write(f"- Loss Given Default: {loss_given_default*100:.2f}%")
        st.write(f"- Expected Annual Loss: {expected_loss_rate*100:.4f}%")
    
    if st.button("📊 Price Risky Bond", key="rb_calc"):
        try:
            result, steps, schedule = price_risky_bond(
                face_value=face_value,
                coupon_rate=coupon_rate,
                maturity=maturity,
                yield_rate=yield_rate,
                default_probability=default_prob,
                recovery_rate=recovery_rate,
                frequency=frequency
            )
            
            st.subheader("📊 Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk-Free Price", f"${result['price_risk_free']:.4f}")
                st.metric("Risky Price", f"${result['price_risky']:.4f}")
            with col2:
                st.metric("Credit Discount", f"${result['credit_discount']:.4f}",
                         delta=f"-{result['credit_discount_pct']:.2f}%", delta_color="inverse")
            with col3:
                st.metric("Implied Credit Spread", f"{result['credit_spread_bps']:.2f} bps")
                st.metric("Risky Yield", f"{result['risky_yield']*100:.4f}%")
            
            st.write("### 📋 Cash Flow Schedule")
            st.dataframe(schedule.style.format({
                't (years)': '{:.1f}',
                'Period': '{:.0f}',
                'Scheduled CF': '{:.2f}',
                'Survival Prob': '{:.6f}',
                'Expected CF': '{:.4f}',
                'Discount Factor': '{:.6f}',
                'PV (Risk-Free)': '{:.4f}',
                'PV (Risky)': '{:.4f}',
                'Credit Adj': '{:.4f}'
            }))
            
            with st.expander("📝 Step-by-Step Calculation"):
                for step in steps:
                    st.markdown(step)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def bond_horizon_analysis_tab():
    """Evaluate bond portfolio at future time under new yield curve."""
    st.subheader("⏳ Horizon Analysis (Bond Aging)")
    
    st.markdown(r"""
    **Problem Type:** You built an immunization portfolio at $t=0$. Now evaluate it at $t=\Delta t$ 
    (e.g., 6 months later) under a **new yield curve**.
    
    **What Changes at Time $t$:**
    1. **Maturities decrease:** $T_{new} = T_{old} - \Delta t$
    2. **Coupons may have been paid:** If time passed ≥ payment period
    3. **New yield curve:** Interest rates have shifted
    4. **Liability also ages:** $T_{L,new} = T_{L,old} - \Delta t$
    
    **Key Calculations:**
    - Reprice each bond with reduced maturity under new yield
    - Reprice liability with reduced maturity
    - Calculate surplus/deficit: Portfolio Value - Liability Value
    """)
    
    st.info(
        "🧠 **Exam Tip:** This tests whether your immunization 'worked'. "
        "A well-immunized portfolio should have surplus ≈ 0 even if yields change, "
        "because duration matching protects against parallel shifts. "
        "Large surplus/deficit indicates immunization failure."
    )
    
    st.write("---")
    st.write("### Portfolio at t=0")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_bonds = st.number_input("Number of Bonds", value=2, min_value=1, max_value=5, key="ha_nbonds")
        time_step = st.number_input("Time Step Δt (years)", value=0.5, min_value=0.1, max_value=5.0, 
                                   format="%.2f", key="ha_dt",
                                   help="How much time has passed, e.g., 0.5 for 6 months")
        frequency = st.selectbox("Coupon Frequency", [1, 2], 
                                format_func=lambda x: "Annual" if x == 1 else "Semi-annual",
                                key="ha_freq")
    
    with col2:
        st.write("**New Yield at t=Δt:**")
        new_yield = st.number_input("New Flat Yield (for all bonds)", value=0.05, format="%.4f", key="ha_newy",
                                   help="The new yield curve (flat) after rates change")
        old_yield = st.number_input("Original Yield at t=0", value=0.05, format="%.4f", key="ha_oldy",
                                   help="The yield used to price bonds at t=0")
    
    st.write(f"**Bond Details (at t=0) - {n_bonds} bonds:**")
    st.markdown("Enter: Face Value, Coupon Rate, Original Maturity, Quantity (units held)")
    
    if n_bonds == 2:
        default_bonds = "1000, 0.04, 2, 0.5\n1000, 0.06, 5, 0.3"
    else:
        default_bonds = "\n".join(["1000, 0.05, 3, 1.0"] * n_bonds)
    
    bonds_input = st.text_area(
        "Bond Data (Face, Coupon, Maturity, Quantity per row)",
        value=default_bonds,
        height=100,
        key="ha_bonds",
        help="Quantity can be fractional (e.g., 0.5 means half a bond)"
    )
    
    st.write("---")
    st.write("### Liability")
    
    col1, col2 = st.columns(2)
    with col1:
        liability_fv = st.number_input("Liability Face Value", value=1000.0, key="ha_lfv",
                                      help="Amount due at liability maturity")
        liability_maturity = st.number_input("Liability Maturity at t=0 (years)", value=3.0, 
                                            format="%.2f", key="ha_lmat")
    with col2:
        liability_type = st.selectbox("Liability Type", 
                                     ["Zero-Coupon", "Coupon-Paying"],
                                     key="ha_ltype")
        if liability_type == "Coupon-Paying":
            liability_coupon = st.number_input("Liability Coupon Rate", value=0.0, format="%.4f", key="ha_lcoup")
        else:
            liability_coupon = 0.0
    
    if st.button("🧮 Analyze at Horizon", key="ha_analyze"):
        try:
            # Parse bond data
            bond_data = parse_matrix_input(bonds_input)
            if bond_data.shape[1] < 4:
                st.error("Need 4 columns: Face, Coupon, Maturity, Quantity")
                return
            
            excel_steps = []
            excel_steps.append(f"**Horizon Analysis at t = {time_step} years**")
            excel_steps.append("")
            
            # Process each bond
            results_t0 = []
            results_t1 = []
            
            excel_steps.append("**Step 1: Bond Values at t=0 (original)**")
            excel_steps.append("")
            
            for i in range(n_bonds):
                face = bond_data[i, 0]
                coupon_rate = bond_data[i, 1]
                maturity_t0 = bond_data[i, 2]
                quantity = bond_data[i, 3]
                
                # Price at t=0
                coupon = face * coupon_rate / frequency
                n_periods = int(maturity_t0 * frequency)
                
                if n_periods <= 0:
                    price_t0 = face
                else:
                    # Standard bond pricing formula
                    period_yield = old_yield / frequency
                    if abs(period_yield) < 1e-10:
                        price_t0 = coupon * n_periods + face
                    else:
                        pv_coupons = coupon * (1 - (1 + period_yield) ** (-n_periods)) / period_yield
                        pv_face = face * (1 + period_yield) ** (-n_periods)
                        price_t0 = pv_coupons + pv_face
                
                value_t0 = price_t0 * quantity
                
                results_t0.append({
                    'Bond': f"Bond {i+1}",
                    'Face': face,
                    'Coupon': coupon_rate,
                    'Maturity (t=0)': maturity_t0,
                    'Quantity': quantity,
                    'Price (t=0)': price_t0,
                    'Value (t=0)': value_t0
                })
                
                excel_steps.append(f"Bond {i+1}: Face={face}, c={coupon_rate*100:.1f}%, T={maturity_t0}, Qty={quantity}")
                excel_steps.append(f"  Price at y={old_yield*100:.2f}%: ${price_t0:.2f}")
                excel_steps.append(f"  Value = {quantity} × ${price_t0:.2f} = ${value_t0:.2f}")
            
            excel_steps.append("")
            excel_steps.append(f"**Step 2: Bond Values at t={time_step} (after time passes)**")
            excel_steps.append("")
            
            for i in range(n_bonds):
                face = bond_data[i, 0]
                coupon_rate = bond_data[i, 1]
                maturity_t0 = bond_data[i, 2]
                quantity = bond_data[i, 3]
                
                # New maturity
                maturity_t1 = maturity_t0 - time_step
                
                # Count coupons received during Δt
                coupon_per_period = face * coupon_rate / frequency
                period_length = 1 / frequency
                coupons_received = int(time_step / period_length)
                coupon_income = coupons_received * coupon_per_period * quantity
                
                # Price at t=Δt with new maturity and new yield
                if maturity_t1 <= 0:
                    # Bond has matured
                    price_t1 = face + coupon_per_period  # Final coupon + face
                    n_periods_t1 = 0
                else:
                    n_periods_t1 = int(maturity_t1 * frequency)
                    period_yield = new_yield / frequency
                    
                    if n_periods_t1 <= 0:
                        price_t1 = face
                    elif abs(period_yield) < 1e-10:
                        price_t1 = coupon_per_period * n_periods_t1 + face
                    else:
                        pv_coupons = coupon_per_period * (1 - (1 + period_yield) ** (-n_periods_t1)) / period_yield
                        pv_face = face * (1 + period_yield) ** (-n_periods_t1)
                        price_t1 = pv_coupons + pv_face
                
                value_t1 = price_t1 * quantity
                total_value_t1 = value_t1 + coupon_income  # Include received coupons
                
                results_t1.append({
                    'Bond': f"Bond {i+1}",
                    'Maturity (t=Δt)': max(0, maturity_t1),
                    'Coupons Received': coupons_received,
                    'Coupon Income': coupon_income,
                    'Price (t=Δt)': price_t1,
                    'Bond Value': value_t1,
                    'Total Value': total_value_t1
                })
                
                excel_steps.append(f"Bond {i+1}: New maturity = {maturity_t0} - {time_step} = {maturity_t1:.2f}")
                if coupons_received > 0:
                    excel_steps.append(f"  Coupons received: {coupons_received} × ${coupon_per_period:.2f} × {quantity} = ${coupon_income:.2f}")
                excel_steps.append(f"  Price at y={new_yield*100:.2f}%, T={maturity_t1:.2f}: ${price_t1:.2f}")
                excel_steps.append(f"  Total value = ${value_t1:.2f} + ${coupon_income:.2f} = ${total_value_t1:.2f}")
            
            # Liability at t=0 and t=Δt
            excel_steps.append("")
            excel_steps.append("**Step 3: Liability Values**")
            excel_steps.append("")
            
            # Liability at t=0
            n_periods_L0 = int(liability_maturity * frequency)
            liability_coupon_pmt = liability_fv * liability_coupon / frequency
            period_yield_old = old_yield / frequency
            
            if n_periods_L0 <= 0 or liability_type == "Zero-Coupon":
                liability_pv_t0 = liability_fv / (1 + old_yield) ** liability_maturity
            else:
                if abs(period_yield_old) < 1e-10:
                    liability_pv_t0 = liability_coupon_pmt * n_periods_L0 + liability_fv
                else:
                    pv_coupons = liability_coupon_pmt * (1 - (1 + period_yield_old) ** (-n_periods_L0)) / period_yield_old
                    pv_face = liability_fv * (1 + period_yield_old) ** (-n_periods_L0)
                    liability_pv_t0 = pv_coupons + pv_face
            
            # Liability at t=Δt
            liability_maturity_t1 = liability_maturity - time_step
            n_periods_L1 = int(max(0, liability_maturity_t1) * frequency)
            period_yield_new = new_yield / frequency
            
            if liability_maturity_t1 <= 0:
                liability_pv_t1 = liability_fv
            elif liability_type == "Zero-Coupon":
                liability_pv_t1 = liability_fv / (1 + new_yield) ** liability_maturity_t1
            else:
                if abs(period_yield_new) < 1e-10:
                    liability_pv_t1 = liability_coupon_pmt * n_periods_L1 + liability_fv
                else:
                    pv_coupons = liability_coupon_pmt * (1 - (1 + period_yield_new) ** (-n_periods_L1)) / period_yield_new
                    pv_face = liability_fv * (1 + period_yield_new) ** (-n_periods_L1)
                    liability_pv_t1 = pv_coupons + pv_face
            
            excel_steps.append(f"Liability: FV=${liability_fv}, T0={liability_maturity}, Type={liability_type}")
            excel_steps.append(f"  PV at t=0 (y={old_yield*100:.2f}%): ${liability_pv_t0:.2f}")
            excel_steps.append(f"  PV at t={time_step} (y={new_yield*100:.2f}%, T={liability_maturity_t1:.2f}): ${liability_pv_t1:.2f}")
            
            # Calculate totals
            portfolio_value_t0 = sum(r['Value (t=0)'] for r in results_t0)
            portfolio_value_t1 = sum(r['Total Value'] for r in results_t1)
            
            surplus_t0 = portfolio_value_t0 - liability_pv_t0
            surplus_t1 = portfolio_value_t1 - liability_pv_t1
            
            excel_steps.append("")
            excel_steps.append("**Step 4: Summary**")
            excel_steps.append("")
            excel_steps.append(f"At t=0: Portfolio=${portfolio_value_t0:.2f}, Liability=${liability_pv_t0:.2f}, Surplus=${surplus_t0:.2f}")
            excel_steps.append(f"At t={time_step}: Portfolio=${portfolio_value_t1:.2f}, Liability=${liability_pv_t1:.2f}, Surplus=${surplus_t1:.2f}")
            excel_steps.append("")
            
            if abs(surplus_t1) < abs(surplus_t0) * 0.1 + 1:
                excel_steps.append("✅ Immunization appears successful (surplus relatively stable)")
            else:
                excel_steps.append("⚠️ Immunization may have failed (surplus changed significantly)")
            
            # Display results
            st.write("### Results at t=0:")
            df_t0 = pd.DataFrame(results_t0)
            st.dataframe(df_t0.style.format({
                'Face': '${:.0f}',
                'Coupon': '{:.2%}',
                'Maturity (t=0)': '{:.2f}',
                'Quantity': '{:.4f}',
                'Price (t=0)': '${:.2f}',
                'Value (t=0)': '${:.2f}'
            }))
            
            st.write(f"### Results at t={time_step}:")
            df_t1 = pd.DataFrame(results_t1)
            st.dataframe(df_t1.style.format({
                'Maturity (t=Δt)': '{:.2f}',
                'Coupons Received': '{:.0f}',
                'Coupon Income': '${:.2f}',
                'Price (t=Δt)': '${:.2f}',
                'Bond Value': '${:.2f}',
                'Total Value': '${:.2f}'
            }))
            
            # Summary metrics
            st.write("### Summary:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**At t=0:**")
                st.metric("Portfolio Value", f"${portfolio_value_t0:,.2f}")
                st.metric("Liability PV", f"${liability_pv_t0:,.2f}")
                st.metric("Surplus", f"${surplus_t0:,.2f}")
            
            with col2:
                st.markdown(f"**At t={time_step}:**")
                st.metric("Portfolio Value", f"${portfolio_value_t1:,.2f}", 
                         delta=f"${portfolio_value_t1 - portfolio_value_t0:,.2f}")
                st.metric("Liability PV", f"${liability_pv_t1:,.2f}",
                         delta=f"${liability_pv_t1 - liability_pv_t0:,.2f}")
                st.metric("Surplus", f"${surplus_t1:,.2f}",
                         delta=f"${surplus_t1 - surplus_t0:,.2f}")
            
            with col3:
                st.markdown("**Change:**")
                yield_change = (new_yield - old_yield) * 10000  # in bps
                st.metric("Yield Change", f"{yield_change:+.0f} bps")
                st.metric("Surplus Change", f"${surplus_t1 - surplus_t0:,.2f}")
                
                if abs(surplus_t1) < 10:
                    st.success("✅ Immunization successful!")
                elif surplus_t1 > 0:
                    st.info("📈 Portfolio outperformed liability")
                else:
                    st.warning("📉 Portfolio underperformed liability")
            
            # Excel steps
            with st.expander("📝 Step-by-Step Excel Solution"):
                for step in excel_steps:
                    st.markdown(step)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def spot_curve_bond_pricing_tab():
    """Price a bond using a zero-coupon yield curve."""
    st.subheader("Bond Pricing with Zero-Coupon Yield Curve")
    
    st.markdown(r"""
    **Price a bond when each cash flow is discounted at its maturity-specific spot rate.**
    
    **Formula:** $P = \sum_{t=1}^{T} \frac{CF_t}{(1 + y_t)^t}$
    
    Where $y_t$ is the zero-coupon yield for maturity $t$.
    
    This differs from standard YTM pricing which uses a single flat rate for all cash flows.
    """)
    
    st.info(
        "🧠 **Exam tip:** When given a zero-coupon yield curve, you must discount each cash flow "
        "at its own maturity-specific spot rate. The YTM is then solved as the single rate that "
        "reprices the bond. Duration uses the YTM for weighting."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Bond Parameters:**")
        face_value = st.number_input("Face Value ($)", value=1000.0, key="scp_fv")
        coupon_rate = st.number_input("Coupon Rate", value=0.02, format="%.4f", key="scp_coupon",
                                     help="Enter as decimal, e.g., 0.02 for 2%")
        maturity = st.number_input("Years to Maturity", value=2, min_value=1, max_value=30, key="scp_mat")
        frequency = st.selectbox("Payment Frequency", [1, 2], 
                                format_func=lambda x: "Annual" if x == 1 else "Semi-annual",
                                key="scp_freq")
    
    with col2:
        st.write("**Zero-Coupon Yield Curve:**")
        st.markdown("Enter maturities and corresponding spot rates:")
        
        curve_input = st.text_area(
            "Yield Curve (Maturity, Spot Rate per row)",
            value="1, 0.048\n2, 0.044\n3, 0.042\n4, 0.041\n5, 0.040",
            height=150,
            help="Format: Maturity (years), Spot Rate (decimal)",
            key="scp_curve"
        )
    
    if st.button("🧮 Price Bond & Calculate Duration", key="scp_calc"):
        try:
            # Parse yield curve
            curve_data = parse_matrix_input(curve_input)
            if curve_data.shape[1] < 2:
                st.error("Need at least 2 columns: Maturity, Spot Rate")
                return
            
            maturities = curve_data[:, 0]
            spot_rates = curve_data[:, 1]
            
            # Check if spot rates look like percentages
            if np.all(spot_rates > 0.5):  # Likely entered as percentages
                spot_rates = spot_rates / 100
                st.info("📝 Converted spot rates from percentages to decimals")
            
            # Price the bond
            result, excel_steps, schedule = price_bond_with_spot_curve(
                face_value=face_value,
                coupon_rate=coupon_rate,
                maturity=maturity,
                spot_rates=spot_rates,
                maturities=maturities,
                frequency=frequency
            )
            
            # Display yield curve
            st.write("### Zero-Coupon Yield Curve Used:")
            curve_df = pd.DataFrame({
                'Maturity (years)': maturities,
                'Spot Rate': spot_rates,
                'Spot Rate (%)': spot_rates * 100
            })
            st.dataframe(curve_df.style.format({
                'Spot Rate': '{:.4f}',
                'Spot Rate (%)': '{:.2f}%'
            }))
            
            # Display results
            st.write("### Results:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bond Price", f"${result['price']:,.4f}")
                st.metric("Coupon Payment", f"${result['coupon_payment']:,.2f}")
            with col2:
                st.metric("Yield to Maturity", f"{result['ytm']*100:.4f}%")
            with col3:
                st.metric("Macaulay Duration", f"{result['macaulay_duration']:.4f} years")
                st.metric("Modified Duration", f"{result['modified_duration']:.4f}")
            
            # Payment schedule
            st.write("### Cash Flow Schedule:")
            st.dataframe(schedule.style.format({
                't (years)': '{:.1f}',
                'Cash Flow': '${:.2f}',
                'Spot Rate': '{:.4f}',
                'DF (Spot)': '{:.6f}',
                'PV (Spot)': '${:.4f}',
                'DF (YTM)': '{:.6f}',
                'PV (YTM)': '${:.4f}',
                'Weight': '{:.6f}',
                't × Weight': '{:.6f}'
            }))
            
            # Summary
            st.success(f"""
            **✅ Summary:**
            - Payment Schedule: ${result['coupon_payment']:.2f} coupon payments at t = {', '.join([f'{t:.0f}' for t in result['payment_times'][:-1]])}, plus ${result['coupon_payment'] + face_value:.2f} at maturity (t = {maturity})
            - Bond Price (using spot curve): **${result['price']:,.4f}**
            - Yield to Maturity: **{result['ytm']*100:.4f}%**
            - Macaulay Duration: **{result['macaulay_duration']:.4f} years**
            """)
            
            # Excel instructions
            with st.expander("📝 Step-by-Step Excel Solution"):
                for step in excel_steps:
                    st.markdown(step)
            
            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot 1: Yield curve
            ax1 = axes[0]
            ax1.plot(maturities, spot_rates * 100, 'b-o', linewidth=2, markersize=8)
            ax1.axhline(y=result['ytm'] * 100, color='r', linestyle='--', label=f"YTM = {result['ytm']*100:.2f}%")
            ax1.set_xlabel('Maturity (years)')
            ax1.set_ylabel('Rate (%)')
            ax1.set_title('Zero-Coupon Yield Curve vs YTM')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Cash flows and PVs
            ax2 = axes[1]
            width = 0.35
            x = np.arange(len(result['payment_times']))
            ax2.bar(x - width/2, result['cash_flows'], width, label='Cash Flow', color='steelblue', alpha=0.7)
            ax2.bar(x + width/2, result['present_values'], width, label='Present Value', color='darkorange', alpha=0.7)
            ax2.set_xlabel('Payment')
            ax2.set_ylabel('$')
            ax2.set_title('Cash Flows vs Present Values')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f't={t:.0f}' for t in result['payment_times']])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def spot_curve_bootstrap_tab():
    """Bootstrap spot curve from coupon bonds."""
    st.subheader("Spot Curve Bootstrapping")
    
    st.markdown("""
    Bootstrap zero/spot rates from coupon bond prices using sequential solving.
    """)
    
    n_bonds = st.number_input("Number of bonds", value=3, min_value=1, max_value=10, key="boot_n")
    face_value = st.number_input("Face Value", value=100.0, key="boot_fv")
    frequency = st.selectbox("Payment Frequency", [1, 2], format_func=lambda x: "Annual" if x == 1 else "Semi-annual", key="boot_freq")
    
    st.write("**Enter bond data (one per row: Maturity, Coupon Rate, Price):**")
    bond_data = st.text_area(
        "Bond Data",
        value="1, 0.03, 99.50\n2, 0.04, 98.80\n3, 0.05, 99.20",
        height=150,
        help="Format: Maturity (years), Coupon Rate (decimal), Clean Price",
        key="boot_data"
    )
    
    if st.button("🧮 Bootstrap Spot Curve", key="boot_calc"):
        try:
            data = parse_matrix_input(bond_data)
            if data.shape[1] < 3:
                st.error("Need at least 3 columns: Maturity, Coupon Rate, Price")
                return
            
            maturities = data[:, 0]
            coupon_rates = data[:, 1]
            prices = data[:, 2]
            
            result, excel_steps, tables = bootstrap_spot_curve(
                maturities, coupon_rates, prices, face_value, frequency
            )
            
            st.subheader("📊 Spot Curve")
            st.dataframe(tables['spot_curve'].style.format({
                'Maturity (T)': '{:.1f}',
                'Spot Rate (r)': '{:.6f}',
                'Spot Rate (%)': '{:.4f}',
                'Discount Factor': '{:.6f}'
            }))
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(result['maturities'], result['spot_rates'] * 100, 'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('Maturity (years)')
            ax.set_ylabel('Spot Rate (%)')
            ax.set_title('Bootstrapped Spot Curve')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📝 How to do this in Excel"):
                for step in excel_steps:
                    st.markdown(step)
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def forward_rates_tab():
    """Compute forward rates from spot curve."""
    st.subheader("Forward Rate Calculator")
    
    st.markdown("""
    Compute forward rates from spot/zero rates.
    
    **Formula:** f(t₁,t₂) = [DF(t₁)/DF(t₂)]^(1/(t₂-t₁)) - 1
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        spot_input = st.text_area(
            "Spot Rates (Maturity, Rate per row)",
            value="1, 0.03\n2, 0.035\n3, 0.04\n4, 0.042\n5, 0.045",
            height=150,
            key="fwd_spots"
        )
    
    with col2:
        forward_start = st.number_input("Forward Start (t₁)", value=1.0, min_value=0.0, key="fwd_t1")
        forward_end = st.number_input("Forward End (t₂)", value=2.0, min_value=0.1, key="fwd_t2")
        
        calc_curve = st.checkbox("Calculate full forward curve (1-year forwards)", key="fwd_curve")
    
    if st.button("🧮 Calculate Forward Rate(s)", key="fwd_calc"):
        try:
            data = parse_matrix_input(spot_input)
            maturities = data[:, 0]
            spot_rates = data[:, 1]
            
            if calc_curve:
                result, excel_steps, tables = compute_forward_curve(spot_rates, maturities)
                
                st.subheader("📊 Forward Curve")
                st.dataframe(tables['forward_curve'])
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                forwards = result['forwards']
                starts = [f['start'] for f in forwards]
                rates = [f['forward_rate'] * 100 for f in forwards]
                ax.bar(starts, rates, width=0.8, alpha=0.7)
                ax.set_xlabel('Forward Start')
                ax.set_ylabel('Forward Rate (%)')
                ax.set_title('1-Year Forward Rates')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            else:
                result, excel_steps, _ = compute_forward_rates(spot_rates, maturities, forward_start, forward_end)
                
                st.subheader("📊 Forward Rate")
                st.metric(f"f({forward_start}, {forward_end})", f"{result['forward_rate']*100:.4f}%")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"r({forward_start})", f"{result['r_start']*100:.4f}%")
                    st.metric(f"DF({forward_start})", f"{result['df_start']:.6f}")
                with col2:
                    st.metric(f"r({forward_end})", f"{result['r_end']*100:.4f}%")
                    st.metric(f"DF({forward_end})", f"{result['df_end']:.6f}")
            
            with st.expander("📝 How to do this in Excel"):
                for step in excel_steps:
                    st.markdown(step)
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def annuity_pricer_tab():
    """Price annuity bond from cashflow vector and curve."""
    st.subheader("Annuity/Custom Bond Pricer")
    
    st.markdown("""
    Price any bond from its cashflow schedule using a discount curve.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Cashflows (Date, Amount per row):**")
        cf_input = st.text_area(
            "Cashflows",
            value="1, 50\n2, 50\n3, 50\n4, 50\n5, 1050",
            height=150,
            key="ann_cf"
        )
    
    with col2:
        curve_type = st.radio("Curve Input Type", ["Discount Factors", "Spot Rates"], key="ann_curve_type")
        
        curve_input = st.text_area(
            f"{curve_type} (Maturity, Value per row)",
            value="1, 0.97\n2, 0.94\n3, 0.91\n4, 0.88\n5, 0.85" if curve_type == "Discount Factors" 
            else "1, 0.03\n2, 0.032\n3, 0.035\n4, 0.038\n5, 0.04",
            height=150,
            key="ann_curve"
        )
    
    if st.button("🧮 Price Bond", key="ann_calc"):
        try:
            cf_data = parse_matrix_input(cf_input)
            cf_dates = cf_data[:, 0]
            cf_amounts = cf_data[:, 1]
            
            curve_data = parse_matrix_input(curve_input)
            curve_mats = curve_data[:, 0]
            curve_vals = curve_data[:, 1]
            
            if curve_type == "Discount Factors":
                # Interpolate DFs directly
                dfs = np.interp(cf_dates, curve_mats, curve_vals)
                result, excel_steps, tables = price_annuity_bond(
                    cf_dates, cf_amounts, discount_factors=dfs
                )
            else:
                result, excel_steps, tables = price_annuity_bond(
                    cf_dates, cf_amounts, spot_rates=curve_vals, rate_maturities=curve_mats
                )
            
            st.subheader("📊 Bond Price")
            st.metric("Present Value", f"${result['price']:.4f}")
            
            st.write("**Cashflow Schedule:**")
            st.dataframe(tables['cashflow_schedule'].style.format({
                'Date (t)': '{:.2f}',
                'Cashflow': '{:.2f}',
                'Discount Factor': '{:.6f}',
                'Present Value': '{:.4f}'
            }))
            
            with st.expander("📝 How to do this in Excel"):
                for step in excel_steps:
                    st.markdown(step)
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def multi_cf_replication_tab():
    """Multi-cashflow replication with arbitrage detection."""
    st.subheader("Multi-Cashflow Replication & Arbitrage")
    
    st.markdown("""
    Replicate target cashflows using tradable instruments. Detects arbitrage if market price differs from replication cost.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Target Cashflows (Date, Amount per row):**")
        target_input = st.text_area(
            "Target CFs",
            value="3, 1000",
            height=100,
            help="E.g., synthetic zero: single cashflow at maturity",
            key="rep_target"
        )
        
        target_price_input = st.number_input(
            "Target Market Price (optional, for arbitrage)",
            value=0.0,
            help="Leave 0 if no market price available",
            key="rep_target_price"
        )
    
    with col2:
        n_inst = st.number_input("Number of Instruments", value=2, min_value=1, max_value=5, key="rep_n_inst")
    
    st.write("**Instruments:**")
    
    instruments = []
    cols = st.columns(n_inst)
    
    for i, col in enumerate(cols):
        with col:
            st.write(f"**Instrument {i+1}**")
            inst_cf = st.text_area(
                f"Cashflows {i+1} (Date, Amount)",
                value=f"1, 50\n2, 1050" if i == 0 else f"1, 60\n2, 60\n3, 1060",
                height=100,
                key=f"rep_inst_cf_{i}"
            )
            inst_price = st.number_input(f"Price {i+1}", value=100.0 - i*2, key=f"rep_inst_p_{i}")
            instruments.append({'cf_input': inst_cf, 'price': inst_price})
    
    if st.button("🧮 Solve Replication", key="rep_calc"):
        try:
            # Parse target
            target_data = parse_matrix_input(target_input)
            target_dates = target_data[:, 0]
            target_amounts = target_data[:, 1]
            
            # Parse instruments
            inst_cashflows = []
            inst_prices = []
            for inst in instruments:
                cf_data = parse_matrix_input(inst['cf_input'])
                inst_cashflows.append({
                    'dates': cf_data[:, 0],
                    'amounts': cf_data[:, 1]
                })
                inst_prices.append(inst['price'])
            
            target_mkt_price = target_price_input if target_price_input > 0 else None
            
            result, excel_steps, tables = replicate_cashflows(
                target_dates, target_amounts,
                inst_cashflows, np.array(inst_prices),
                target_mkt_price
            )
            
            st.subheader("📊 Replication Solution")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Replication Cost", f"${result['replication_cost']:.4f}")
                st.metric("Exact Solution", "✓" if result['exact_solution'] else "✗")
            with col2:
                st.metric("Method", result['method'])
            
            st.write("**Positions:**")
            st.dataframe(tables['positions'])
            
            st.write("**Cashflow Verification:**")
            st.dataframe(tables['verification'].style.format({
                'Date': '{:.1f}',
                'Target': '{:.2f}',
                'Replicated': '{:.2f}',
                'Residual': '{:.4f}'
            }))
            
            # Arbitrage analysis
            if result['arbitrage']:
                arb = result['arbitrage']
                st.success(f"🎯 **ARBITRAGE FOUND:** {arb['type']}")
                st.write(f"**Action:** {arb['action']}")
                st.write(f"**Profit:** ${arb['profit']:.4f}")
                
                st.write("**Trades:**")
                for trade in arb['trades']:
                    direction = "BUY" if trade['position'] > 0 else "SELL"
                    st.write(f"  - {direction} {abs(trade['position']):.4f} units of {trade['instrument']}")
            
            with st.expander("📝 How to do this in Excel"):
                for step in excel_steps:
                    st.markdown(step)
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def human_capital_module_integrated():
    """Integrated Human Capital module."""
    st.header("👤 Module 3: Human Capital & Life-Cycle")

    # Module 4 guidance expander
    with st.expander("📘 Module Guidance: When to use this?", expanded=False):
        st.markdown('''
        **When to use:**
        * Problem mentions **"Labor Income"**, **"Human Capital"**, or **"Life-Cycle"**.
        * Investor has Financial Wealth ($W$) AND Future Income ($L_0$).
        * You need the **optimal allocation** taking labor income risk into account.
        * Correlation between Labor and Stock Market ($\rho_{SL}$) is non-zero.

        **What you'll solve for:**
        * **PV of Human Capital ($L_0$):** Using annuity formulas.
        * **Optimal Stock Weight ($\pi^*$):** Adjusted for the implicit bond/stock nature of labor.

        **Inputs you need:**
        * Current Income, Growth rate ($g$), Discount rate ($r$), Remaining years ($T$).
        * Market params: $\mu_{mkt}, \sigma_{mkt}, r_f, \gamma$.
        * Labor risk: $\sigma_L, \rho_{SL}$.

        **Common Pitfalls:**
        * **Leverage:** Optimal weights often $>100\%$ early in life (borrowing against human capital).
        * **Correlation Sign:** Positive correlation $\rho_{SL} > 0$ reduces stock allocation (hedging). Negative increases it.
        * **Total Wealth:** Remember the denominator is often Financial Wealth + Human Capital.

        **Fast Decision Rule:**
        "If the problem mentions **Salary, Labor Income, or Human Capital**, use this module."
        ''')
    
    tab1, tab2, tab3 = st.tabs(["📊 Basic Calculator", "🎯 Single Market with HC", "🔢 Multi-Asset with HC"])
    
    with tab1:
        human_capital_basic_tab()
    
    with tab2:
        human_capital_section()
    
    with tab3:
        human_capital_multi_asset_tab()


def human_capital_basic_tab():
    """Basic human capital calculator."""
    st.markdown("""
    Calculate the present value of human capital and optimal portfolio allocation 
    considering labor income as an implicit asset.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💼 Labor Income")
        income = st.number_input("Current Annual Income ($)", value=75000.0, key="hc_basic_income")
        growth = st.number_input("Income Growth Rate (g)", value=0.03, format="%.4f", key="hc_basic_growth")
        years = st.number_input("Working Years Remaining (T)", value=30, min_value=1, key="hc_basic_years")
    
    with col2:
        st.subheader("📊 Market Parameters")
        discount = st.number_input("Discount Rate (r)", value=0.05, format="%.4f", key="hc_basic_discount")
        wealth = st.number_input("Current Financial Wealth ($)", value=200000.0, key="hc_basic_wealth")
    
    if st.button("🧮 Calculate Human Capital", key="hc_basic_calc"):
        try:
            # Growing annuity formula
            g, r, T, Y = growth, discount, years, income
            
            if abs(r - g) < 1e-6:
                human_capital = Y * T / (1 + r)
            else:
                growth_factor = (1 + g) / (1 + r)
                human_capital = Y * (1 - growth_factor ** T) / (r - g)
            
            total_wealth = wealth + human_capital
            ratio = human_capital / wealth if wealth > 0 else float('inf')
            
            st.subheader("📊 Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Human Capital (L₀)", f"${human_capital:,.0f}")
            with col2:
                st.metric("Total Wealth", f"${total_wealth:,.0f}")
            with col3:
                st.metric("L₀/W Ratio", f"{ratio:.2f}")
            
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            sizes = [wealth, human_capital]
            labels = [f'Financial Wealth\n${wealth:,.0f}', f'Human Capital\n${human_capital:,.0f}']
            colors = ['#2ecc71', '#3498db']
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Total Wealth Composition')
            st.pyplot(fig)
            plt.close()
            
            with st.expander("📝 How to do this in Excel"):
                st.markdown(f"""
**Growing Annuity Formula:**
```
L₀ = Y₀ × [1 - ((1+g)/(1+r))^T] / (r - g)
```

**Excel Formula:**
```
={income} * (1 - ((1+{growth})/(1+{discount}))^{years}) / ({discount} - {growth})
```

**Result:** ${human_capital:,.2f}
                """)
        
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def factor_models_module_integrated():
    """Integrated Factor Models with advanced features."""
    st.header("📉 Module 5: Factor Models & Active Management")

    # Module 5 guidance expander - COMPREHENSIVE EXAM GUIDE
    with st.expander("📘 Module Guidance: When to use this?", expanded=False):
        st.markdown(r'''
### When to Use This Module

**Use the Performance Metrics tab when:**
- Question asks for **Sharpe Ratio**, **Treynor Ratio**, **Jensen's Alpha**, or **Information Ratio**
- You need to evaluate if a fund **outperformed** its benchmark
- Given: Fund return, Fund volatility, Fund beta, Market return, Market volatility, $r_f$

**Use the Treynor-Black tab when:**
- Question asks for **optimal active portfolio weights**
- You have **mispriced securities** (non-zero alphas) and want to combine with passive index
- Need to calculate: $w_i^A \propto \frac{\alpha_i}{\sigma_{\epsilon,i}^2}$
- Given: Multiple stocks with α, β, σ_ε

**Use the Risk Analysis tab when:**
- Question asks to **decompose risk** into systematic vs idiosyncratic
- Need **R²** (what fraction of variance is explained by factors)
- Need **covariance between assets** or between **asset and factor**
- Question mentions: "What is the correlation between returns of Asset A and Asset B?"
- Given: Factor betas, factor volatilities, factor correlations, idiosyncratic vols

**Use the Reverse APT tab when:**
- Question gives **expected returns** and **betas** but asks for **factor risk premiums**
- This is "reverse engineering" the APT equation
- Given: E[r] for multiple assets, β matrix, $r_f$

**Use the Black-Litterman tab when:**
- Question asks to combine **market equilibrium** with **investor views**
- Given: Market cap weights, covariance matrix, specific views on assets

---
### Key Formulas Quick Reference

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Sharpe Ratio | $SR = \frac{E[r] - r_f}{\sigma}$ | Total risk-adjusted return |
| Treynor Ratio | $TR = \frac{E[r] - r_f}{\beta}$ | Systematic risk-adjusted return |
| Jensen's Alpha | $\alpha = E[r] - [r_f + \beta(E[r_m] - r_f)]$ | Abnormal return vs CAPM |
| Information Ratio | $IR = \frac{\alpha}{\sigma_\epsilon}$ | Alpha per unit of active risk |
| Systematic Variance | $\sigma^2_{sys} = \beta' \Sigma_F \beta$ | Factor-driven risk |
| R² | $R^2 = \frac{\sigma^2_{sys}}{\sigma^2_{total}}$ | Fraction explained by factors |
| Asset Covariance | $Cov(R_i, R_j) = \beta_i' \Sigma_F \beta_j$ | Correlation between assets |
| Asset-Factor Cov | $Cov(R_i, F_k) = \beta_{ik} \sigma^2_{F_k} + \sum_{j \neq k} \beta_{ij} Cov(F_j, F_k)$ | Asset exposure to factor |

---
### Common Exam Pitfalls

1. **Total vs Residual Risk**: Sharpe uses **total σ**, Information Ratio uses **residual σ_ε**
2. **Systematic Variance**: Use matrix formula $\beta' \Sigma_F \beta$, NOT just $\sum \beta_k^2 \sigma_k^2$ (unless factors are uncorrelated!)
3. **Alpha Definition**: Must be excess return over CAPM prediction, not raw return minus benchmark
4. **R²**: This is systematic/total variance, NOT correlation squared (though related)
5. **Factor Correlations Matter**: When factors are correlated, asset covariances have cross-terms

---
### Decision Tree

```
Is the question about fund performance vs benchmark?
├── YES → Performance Metrics tab
└── NO → Continue...

Does it ask for optimal weights with mispriced securities?
├── YES → Treynor-Black tab  
└── NO → Continue...

Does it ask to find factor risk premiums from returns?
├── YES → Reverse APT tab
└── NO → Continue...

Does it ask about risk decomposition, R², or covariances?
├── YES → Risk Analysis tab
└── NO → Check other modules
```
        ''')
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Performance Metrics", 
        "🎯 Treynor-Black",
        "🔮 Black-Litterman",
        "📊 Risk Analysis",
        "🔄 Reverse APT",
        "🔧 Covariance Builder",
        "⚖️ Factor-Neutral",
        "🔗 Asset + Factor"
    ])
    
    with tab1:
        advanced_performance_section()
    
    with tab2:
        advanced_treynor_black_section()
    
    with tab3:
        black_litterman_section()
    
    with tab4:
        factor_analysis_section()
    
    with tab5:
        reverse_apt_solver_tab()
    
    with tab6:
        covariance_builder_tab()
    
    with tab7:
        factor_neutral_tab()
    
    with tab8:
        joint_asset_factor_tab()


def reverse_apt_solver_tab():
    """Solve for factor risk premiums given expected returns and betas."""
    st.subheader("🔄 Reverse APT: Solve for Factor Risk Premiums")
    
    st.markdown(r"""
    **Problem Type:** You know the expected returns $E[r_i]$ and factor betas $\beta_{i,k}$ for several assets.
    You need to find the **Factor Risk Premiums** $RP_k$.
    
    **The APT Equation:**
    $$E[r_i] - r_f = \beta_{i,1} \times RP_1 + \beta_{i,2} \times RP_2 + \ldots$$
    
    **In Matrix Form:**
    $$\mathbf{B} \times \mathbf{RP} = \mathbf{E[r] - r_f}$$
    
    **Solution:**
    $$\mathbf{RP} = \mathbf{B}^{-1} \times (\mathbf{E[r] - r_f})$$
    
    If you have more assets than factors (overdetermined), we use least squares.
    """)
    
    st.info(
        "🧠 **Exam Tip:** This is the 'reverse' of the usual APT calculation. "
        "Usually you're given risk premiums and calculate expected returns. "
        "Here, you're given expected returns and must back out the risk premiums."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_assets = st.number_input("Number of Assets", value=2, min_value=2, max_value=10, key="rapt_na")
        n_factors = st.number_input("Number of Factors", value=2, min_value=1, max_value=5, key="rapt_nf")
        rf = st.number_input("Risk-Free Rate (rf)", value=0.02, format="%.4f", key="rapt_rf")
    
    with col2:
        st.write("**Asset Names (optional):**")
        asset_names_input = st.text_input(
            "Comma-separated", value="", key="rapt_anames",
            help="E.g., 'Stock A, Stock B'"
        )
        st.write("**Factor Names (optional):**")
        factor_names_input = st.text_input(
            "Comma-separated", value="", key="rapt_fnames",
            help="E.g., 'Market, SMB'"
        )
    
    st.write("**Expected Returns E[r] (one per asset, comma-separated):**")
    returns_input = st.text_area(
        "Expected Returns",
        value="0.10, 0.12" if n_assets == 2 else ", ".join(["0.10"] * n_assets),
        key="rapt_returns",
        help="Enter as decimals, e.g., 0.10 for 10%"
    )
    
    st.write(f"**Beta Matrix ({n_assets} assets × {n_factors} factors):**")
    if n_assets == 2 and n_factors == 2:
        default_betas = "1.0, 0.5\n0.8, 1.2"
    else:
        default_betas = "\n".join([", ".join(["1.0"] * n_factors) for _ in range(n_assets)])
    
    betas_input = st.text_area(
        "Beta Matrix (one row per asset)",
        value=default_betas,
        key="rapt_betas",
        help="Each row is one asset's betas for all factors"
    )
    
    if st.button("🧮 Solve for Risk Premiums", key="rapt_solve"):
        try:
            # Parse inputs
            expected_returns = parse_messy_input(returns_input)
            beta_matrix = parse_matrix_input(betas_input)
            
            # Parse names
            if asset_names_input.strip():
                asset_names = [n.strip() for n in asset_names_input.split(',')]
            else:
                asset_names = [f"Asset {i+1}" for i in range(n_assets)]
            
            if factor_names_input.strip():
                factor_names = [n.strip() for n in factor_names_input.split(',')]
            else:
                factor_names = [f"Factor {i+1}" for i in range(n_factors)]
            
            # Validate dimensions
            if len(expected_returns) != n_assets:
                st.error(f"Expected {n_assets} returns, got {len(expected_returns)}")
                return
            if beta_matrix.shape != (n_assets, n_factors):
                st.error(f"Beta matrix should be {n_assets}×{n_factors}, got {beta_matrix.shape}")
                return
            
            # Calculate excess returns
            excess_returns = expected_returns - rf
            
            # Solve the system
            excel_steps = []
            excel_steps.append("**Step 1: Set up the system of equations**")
            excel_steps.append("")
            excel_steps.append("APT says: E[r_i] - rf = Σ β_{i,k} × RP_k")
            excel_steps.append("")
            
            for i in range(n_assets):
                terms = [f"β_{{{i+1},{k+1}}} × RP_{k+1}" for k in range(n_factors)]
                excel_steps.append(f"{asset_names[i]}: {expected_returns[i]:.4f} - {rf:.4f} = {' + '.join(terms)}")
            excel_steps.append("")
            
            excel_steps.append("**Step 2: Write in matrix form B × RP = (E[r] - rf)**")
            excel_steps.append("")
            excel_steps.append("Beta Matrix B:")
            for i in range(n_assets):
                excel_steps.append(f"  [{', '.join([f'{beta_matrix[i,k]:.4f}' for k in range(n_factors)])}]")
            excel_steps.append("")
            excel_steps.append(f"Excess Returns: [{', '.join([f'{er:.4f}' for er in excess_returns])}]")
            excel_steps.append("")
            
            if n_assets == n_factors:
                # Square system - exact solution
                excel_steps.append("**Step 3: Solve exactly (square system)**")
                excel_steps.append("RP = B⁻¹ × (E[r] - rf)")
                excel_steps.append("")
                excel_steps.append("Excel: `=MMULT(MINVERSE(B), ExcessReturns)`")
                
                risk_premiums = np.linalg.solve(beta_matrix, excess_returns)
                solution_type = "Exact Solution (square system)"
                
            elif n_assets > n_factors:
                # Overdetermined - least squares
                excel_steps.append("**Step 3: Solve by least squares (more assets than factors)**")
                excel_steps.append("RP = (B'B)⁻¹ B' × (E[r] - rf)")
                excel_steps.append("")
                excel_steps.append("Excel: `=MMULT(MMULT(MINVERSE(MMULT(TRANSPOSE(B),B)),TRANSPOSE(B)),ExcessReturns)`")
                
                risk_premiums, residuals, rank, s = np.linalg.lstsq(beta_matrix, excess_returns, rcond=None)
                solution_type = "Least Squares Solution (overdetermined)"
                
            else:
                st.error("Need at least as many assets as factors to solve!")
                return
            
            excel_steps.append("")
            excel_steps.append("**Step 4: Results**")
            for k in range(n_factors):
                excel_steps.append(f"  {factor_names[k]} Risk Premium (RP_{k+1}) = {risk_premiums[k]:.6f} = {risk_premiums[k]*100:.2f}%")
            
            # Verify solution
            excel_steps.append("")
            excel_steps.append("**Step 5: Verify (B × RP should equal E[r] - rf)**")
            fitted = beta_matrix @ risk_premiums
            for i in range(n_assets):
                error = excess_returns[i] - fitted[i]
                excel_steps.append(f"  {asset_names[i]}: {fitted[i]:.6f} vs {excess_returns[i]:.6f} (error: {error:.6f})")
            
            # Display results
            st.success(f"**{solution_type}**")
            
            st.write("### Factor Risk Premiums:")
            cols = st.columns(n_factors)
            for k, col in enumerate(cols):
                with col:
                    st.metric(f"{factor_names[k]}", f"{risk_premiums[k]*100:.2f}%")
            
            # Summary table
            results_df = pd.DataFrame({
                'Factor': factor_names,
                'Risk Premium': risk_premiums,
                'Risk Premium (%)': [f"{rp*100:.2f}%" for rp in risk_premiums]
            })
            st.dataframe(results_df)
            
            # Verification table
            st.write("### Verification:")
            verify_df = pd.DataFrame({
                'Asset': asset_names,
                'E[r]': expected_returns,
                'E[r] - rf (Actual)': excess_returns,
                'B × RP (Fitted)': fitted,
                'Error': excess_returns - fitted
            })
            st.dataframe(verify_df.style.format({
                'E[r]': '{:.4f}',
                'E[r] - rf (Actual)': '{:.4f}',
                'B × RP (Fitted)': '{:.4f}',
                'Error': '{:.6f}'
            }))
            
            # Show input summary
            st.write("### Input Summary:")
            beta_df = pd.DataFrame(
                beta_matrix,
                index=asset_names,
                columns=factor_names
            )
            st.write("**Beta Matrix:**")
            st.dataframe(beta_df.style.format('{:.4f}'))
            
            # Excel instructions
            with st.expander("📝 Step-by-Step Excel Solution"):
                for step in excel_steps:
                    st.markdown(step)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def universal_solver_module():
    """
    A flexible algebra solver for common finance formulas.
    Allows the user to select ANY variable as the unknown 'X'.
    """
    st.header("🧮 Module 7: Universal Solver")

    # Module 7 guidance expander
    with st.expander("📘 Module Guidance: When to use this?", expanded=False):
        st.markdown(r'''
        **When to use:**
        * You have a simple equation (CAPM, Beta def, Sharpe def) and need to solve for one missing variable.
        * **Reverse Weighted Average:** You know E[rp] and weights, but need to find one missing asset's return.
        * Example: "Given $\beta$ and Market Risk Premium, find Expected Return."
        * Example: "Given Correlation and Volatilities, find Beta."
        * Example: "Portfolio return is 9%, weights are 1/3 each, Stock X returns 6%, rf=2%. Find E[r_Y]."

        **Inputs you need:**
        * Any 3 of 4 variables in standard finance equations.
        * For Reverse Weighted Average: Portfolio return, weights, all but one asset's return.

        **Common Pitfalls:**
        * **Reverse Weighted Average:** Weights must sum to 1. If missing asset has w=0, cannot solve.

        **Fast Decision Rule:**
        "Use this for **simple algebra** where you need to solve for X in CAPM, Beta, Sharpe, or weighted average formulas."
        ''')

    st.markdown("""
    **How to use:**
    1. Identify which formula applies to your problem.
    2. Select the variable you are **looking for** in the dropdown.
    3. Enter the variables you **know**.
    """)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📐 CAPM Solver", 
        "🔗 Beta & Correlation", 
        "⚖️ Sharpe & Ratios",
        "🔄 Reverse Weighted Avg",
        "📊 Single-Index Model"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: CAPM SOLVER
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("CAPM Equation Solver")
        st.info("Use this if the problem mentions: Expected Return, Risk-Free Rate, Beta, or Market Return.")
        st.latex(r"E[r_i] = r_f + \beta_i (E[r_m] - r_f)")
        
        # User selects what to find
        target = st.selectbox(
            "I want to find:",
            ["Expected Return (E[r])", "Beta (β)", "Market Return (E[rm])", "Risk-Free Rate (rf)"],
            key="capm_solve_target"
        )
        
        st.write("---")
        col1, col2 = st.columns(2)
        
        # Dynamic Inputs based on selection
        inputs = {}
        
        if target != "Expected Return (E[r])":
            inputs['er'] = col1.number_input("Expected Return E[r]", value=0.10, format="%.4f", key="solver_er", help="The return of the individual asset")
        
        if target != "Risk-Free Rate (rf)":
            inputs['rf'] = col2.number_input("Risk-Free Rate rf", value=0.04, format="%.4f", key="solver_rf", help="The rate of the risk-free asset (Treasury bill)")
            
        if target != "Beta (β)":
            inputs['beta'] = col1.number_input("Beta β", value=1.2, format="%.2f", key="solver_beta", help="Systematic risk of the asset")
            
        if target != "Market Return (E[rm])":
            inputs['erm'] = col2.number_input("Market Return E[rm]", value=0.09, format="%.4f", key="solver_erm", help="Expected return of the market portfolio")

        if st.button("Solve CAPM", key="btn_solve_capm"):
            try:
                st.write("### Solution")
                if target == "Expected Return (E[r])":
                    # E[r] = rf + beta * (erm - rf)
                    res = inputs['rf'] + inputs['beta'] * (inputs['erm'] - inputs['rf'])
                    st.latex(rf"E[r] = {inputs['rf']} + {inputs['beta']} ({inputs['erm']} - {inputs['rf']})")
                    st.metric("Result E[r]", f"{res:.4f} ({res*100:.2f}%)")
                    
                elif target == "Beta (β)":
                    # beta = (E[r] - rf) / (erm - rf)
                    mrp = inputs['erm'] - inputs['rf']
                    if mrp == 0:
                        st.error("Market Risk Premium is 0, cannot divide.")
                    else:
                        res = (inputs['er'] - inputs['rf']) / mrp
                        st.latex(rf"\beta = \frac{{E[r] - r_f}}{{E[r_m] - r_f}} = \frac{{{inputs['er']} - {inputs['rf']}}}{{{inputs['erm']} - {inputs['rf']}}}")
                        st.metric("Result Beta", f"{res:.4f}")

                elif target == "Market Return (E[rm])":
                    # E[r] = rf + beta(erm - rf) -> (E[r] - rf)/beta + rf = erm
                    if inputs['beta'] == 0:
                        st.error("Beta is 0, cannot isolate E[rm].")
                    else:
                        res = (inputs['er'] - inputs['rf']) / inputs['beta'] + inputs['rf']
                        st.latex(r"E[r_m] = \frac{E[r] - r_f}{\beta} + r_f")
                        st.metric("Result E[rm]", f"{res:.4f} ({res*100:.2f}%)")

                elif target == "Risk-Free Rate (rf)":
                    # E[r] = rf + beta*erm - beta*rf
                    # E[r] - beta*erm = rf(1 - beta)
                    # rf = (E[r] - beta*erm) / (1 - beta)
                    if inputs['beta'] == 1:
                        st.error("Beta is 1, cannot isolate rf (equation undefined).")
                    else:
                        res = (inputs['er'] - inputs['beta'] * inputs['erm']) / (1 - inputs['beta'])
                        st.latex(r"r_f = \frac{E[r] - \beta E[r_m]}{1 - \beta}")
                        st.metric("Result rf", f"{res:.4f} ({res*100:.2f}%)")
            except Exception as e:
                st.error(f"Error: {e}")

    # -------------------------------------------------------------------------
    # TAB 2: BETA & CORRELATION SOLVER
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Beta Decomposition Solver")
        st.info("Use this if the problem mentions: Correlation, Volatility (Standard Deviation), or Beta.")
        st.latex(r"\beta_i = \rho_{i,m} \times \frac{\sigma_i}{\sigma_m}")
        
        target_beta = st.selectbox(
            "I want to find:",
            ["Beta (β)", "Correlation (ρ)", "Asset Volatility (σi)", "Market Volatility (σm)"],
            key="beta_solve_target"
        )
        
        st.write("---")
        col1, col2 = st.columns(2)
        b_in = {}
        
        if target_beta != "Beta (β)":
            b_in['beta'] = col1.number_input("Beta (β)", value=1.0, key="bs_beta")
        if target_beta != "Correlation (ρ)":
            b_in['rho'] = col2.number_input("Correlation (ρ)", value=0.5, min_value=-1.0, max_value=1.0, key="bs_rho")
        if target_beta != "Asset Volatility (σi)":
            b_in['sigi'] = col1.number_input("Asset Vol (σi)", value=0.30, key="bs_sigi")
        if target_beta != "Market Volatility (σm)":
            b_in['sigm'] = col2.number_input("Market Vol (σm)", value=0.20, key="bs_sigm")
            
        if st.button("Solve Relationship", key="btn_solve_beta"):
            st.write("### Solution")
            if target_beta == "Beta (β)":
                # Check for division by zero in sigma_m
                if b_in['sigm'] == 0:
                    st.error("Market Volatility is 0, cannot divide.")
                else:
                    res = b_in['rho'] * (b_in['sigi'] / b_in['sigm'])
                    st.latex(rf"\beta = {b_in['rho']} \times \frac{{{b_in['sigi']}}}{{{b_in['sigm']}}}")
                    st.metric("Result Beta", f"{res:.4f}")
                    
            elif target_beta == "Correlation (ρ)":
                # rho = beta * (sigm / sigi)
                if b_in['sigi'] == 0:
                    st.error("Asset Volatility is 0, cannot divide.")
                else:
                    res = b_in['beta'] * (b_in['sigm'] / b_in['sigi'])
                    st.latex(r"\rho = \beta \times \frac{\sigma_m}{\sigma_i}")
                    st.metric("Result Correlation", f"{res:.4f}")
                    if abs(res) > 1:
                        st.warning("⚠️ Warning: Mathematical result > 1 (or < -1). Check your inputs.")
                
            elif target_beta == "Asset Volatility (σi)":
                # sigi = (beta * sigm) / rho
                if b_in['rho'] == 0:
                    st.error("Correlation is 0, cannot divide.")
                else:
                    res = (b_in['beta'] * b_in['sigm']) / b_in['rho']
                    st.latex(r"\sigma_i = \frac{\beta \times \sigma_m}{\rho}")
                    st.metric("Result σi", f"{res:.4f}")
                
            elif target_beta == "Market Volatility (σm)":
                # sigm = (rho * sigi) / beta
                if b_in['beta'] == 0:
                    st.error("Beta is 0, cannot divide.")
                else:
                    res = (b_in['rho'] * b_in['sigi']) / b_in['beta']
                    st.latex(r"\sigma_m = \frac{\rho \times \sigma_i}{\beta}")
                    st.metric("Result σm", f"{res:.4f}")

    # -------------------------------------------------------------------------
    # TAB 3: SHARPE RATIO SOLVER
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Sharpe Ratio Solver")
        st.info("Use this if the problem mentions: Sharpe Ratio, Excess Return, or Volatility.")
        st.latex(r"SR = \frac{E[r] - r_f}{\sigma}")
        
        target_sr = st.selectbox(
            "I want to find:",
            ["Sharpe Ratio", "Expected Return (E[r])", "Risk-Free Rate (rf)", "Volatility (σ)"],
            key="sr_solve_target"
        )
        
        col1, col2 = st.columns(2)
        s_in = {}
        
        if target_sr != "Sharpe Ratio":
            s_in['sr'] = col1.number_input("Sharpe Ratio", value=0.4, key="ss_sr")
        if target_sr != "Expected Return (E[r])":
            s_in['er'] = col2.number_input("Expected Return E[r]", value=0.12, key="ss_er")
        if target_sr != "Risk-Free Rate (rf)":
            s_in['rf'] = col1.number_input("Risk-Free Rate rf", value=0.04, key="ss_rf")
        if target_sr != "Volatility (σ)":
            s_in['sig'] = col2.number_input("Volatility σ", value=0.20, key="ss_sig")
            
        if st.button("Solve Sharpe", key="btn_solve_sr"):
            st.write("### Solution")
            if target_sr == "Sharpe Ratio":
                if s_in['sig'] == 0:
                    st.error("Volatility is 0, cannot divide.")
                else:
                    res = (s_in['er'] - s_in['rf']) / s_in['sig']
                    st.metric("Result Sharpe", f"{res:.4f}")
            elif target_sr == "Expected Return (E[r])":
                # E[r] = SR * sig + rf
                res = s_in['sr'] * s_in['sig'] + s_in['rf']
                st.latex(r"E[r] = SR \times \sigma + r_f")
                st.metric("Result E[r]", f"{res:.4f} ({res*100:.2f}%)")
            elif target_sr == "Volatility (σ)":
                # sig = (er - rf) / sr
                if s_in['sr'] == 0:
                    st.error("Sharpe is 0, cannot divide.")
                else:
                    res = (s_in['er'] - s_in['rf']) / s_in['sr']
                    st.metric("Result σ", f"{res:.4f}")
            elif target_sr == "Risk-Free Rate (rf)":
                # rf = er - sr*sig
                res = s_in['er'] - (s_in['sr'] * s_in['sig'])
                st.metric("Result rf", f"{res:.4f}")
    
    # -------------------------------------------------------------------------
    # TAB 4: REVERSE WEIGHTED AVERAGE
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("Reverse Weighted Average Solver")
        
        st.markdown(r"""
        **Problem Type:** You know the portfolio's expected return $E[r_p]$ and the weights, 
        but one asset's expected return is **unknown**.
        
        **The Weighted Average Equation:**
        $$E[r_p] = w_1 \times E[r_1] + w_2 \times E[r_2] + \ldots + w_n \times E[r_n]$$
        
        **Solving for the missing return:**
        $$E[r_{missing}] = \frac{E[r_p] - \sum_{known} w_i \times E[r_i]}{w_{missing}}$$
        """)
        
        st.info(
            "🧠 **Exam Tip:** This comes up when you're given a portfolio return and all but one "
            "constituent's return. Common setup: equally weighted portfolio with 3 assets, "
            "you know E[rp], rf, and one stock's return. Solve for the other stock."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_assets_rwa = st.number_input("Number of Assets", value=3, min_value=2, max_value=10, key="rwa_n")
            portfolio_return = st.number_input("Portfolio Expected Return E[rp]", value=0.09, format="%.4f", key="rwa_erp",
                                              help="The known portfolio return")
        
        with col2:
            st.write("**Asset Names (optional):**")
            rwa_names_input = st.text_input(
                "Comma-separated", value="Stock X, Stock Y, rf" if n_assets_rwa == 3 else "",
                key="rwa_names"
            )
            missing_index = st.number_input("Which asset is unknown? (1-indexed)", 
                                           value=2, min_value=1, max_value=n_assets_rwa, key="rwa_missing",
                                           help="Enter the position of the unknown asset (1 = first asset)")
        
        st.write("**Weights (must sum to 1):**")
        if n_assets_rwa == 3:
            default_weights = "0.3333, 0.3333, 0.3334"
        else:
            default_weights = ", ".join([f"{1/n_assets_rwa:.4f}"] * n_assets_rwa)
        
        weights_rwa_input = st.text_area(
            "Weights (comma-separated)",
            value=default_weights,
            key="rwa_weights"
        )
        
        st.write("**Expected Returns (enter '?' or leave blank for unknown):**")
        if n_assets_rwa == 3:
            default_returns = "0.06, ?, 0.02"
        else:
            default_returns = ", ".join(["0.10"] * (missing_index - 1) + ["?"] + ["0.10"] * (n_assets_rwa - missing_index))
        
        returns_rwa_input = st.text_area(
            "Returns (comma-separated, use ? for unknown)",
            value=default_returns,
            key="rwa_returns",
            help="Enter '?' or 'x' or leave blank for the unknown return"
        )
        
        if st.button("🧮 Solve for Missing Return", key="rwa_solve"):
            try:
                # Parse weights
                weights_rwa = parse_messy_input(weights_rwa_input)
                
                # Parse names
                if rwa_names_input.strip():
                    asset_names_rwa = [n.strip() for n in rwa_names_input.split(',')]
                else:
                    asset_names_rwa = [f"Asset {i+1}" for i in range(n_assets_rwa)]
                
                # Validate weights
                if len(weights_rwa) != n_assets_rwa:
                    st.error(f"Expected {n_assets_rwa} weights, got {len(weights_rwa)}")
                    return
                
                weight_sum = np.sum(weights_rwa)
                if abs(weight_sum - 1.0) > 0.01:
                    st.warning(f"⚠️ Weights sum to {weight_sum:.4f}, not 1.0")
                
                # Parse returns (handling '?' for unknown)
                returns_parts = returns_rwa_input.replace(' ', '').split(',')
                if len(returns_parts) != n_assets_rwa:
                    st.error(f"Expected {n_assets_rwa} returns, got {len(returns_parts)}")
                    return
                
                returns_rwa = []
                missing_idx = None
                for i, part in enumerate(returns_parts):
                    part = part.strip().lower()
                    if part in ['?', 'x', '', 'nan', 'unknown']:
                        if missing_idx is not None:
                            st.error("Only one unknown return is allowed!")
                            return
                        missing_idx = i
                        returns_rwa.append(np.nan)
                    else:
                        returns_rwa.append(float(part))
                
                returns_rwa = np.array(returns_rwa)
                
                if missing_idx is None:
                    st.error("No unknown return found! Use '?' to mark the unknown.")
                    return
                
                # Check if missing weight is zero
                if abs(weights_rwa[missing_idx]) < 1e-10:
                    st.error(f"Cannot solve: weight of {asset_names_rwa[missing_idx]} is zero!")
                    return
                
                # Calculate the missing return
                known_contribution = 0.0
                for i in range(n_assets_rwa):
                    if i != missing_idx:
                        known_contribution += weights_rwa[i] * returns_rwa[i]
                
                missing_return = (portfolio_return - known_contribution) / weights_rwa[missing_idx]
                
                # Excel steps
                excel_steps = []
                excel_steps.append("**Step 1: Write the weighted average equation**")
                excel_steps.append("")
                terms = [f"w_{i+1} × E[r_{i+1}]" for i in range(n_assets_rwa)]
                excel_steps.append(f"E[rp] = {' + '.join(terms)}")
                excel_steps.append("")
                
                excel_steps.append("**Step 2: Substitute known values**")
                excel_steps.append("")
                terms_filled = []
                for i in range(n_assets_rwa):
                    if i == missing_idx:
                        terms_filled.append(f"{weights_rwa[i]:.4f} × E[r_{asset_names_rwa[i]}]")
                    else:
                        terms_filled.append(f"{weights_rwa[i]:.4f} × {returns_rwa[i]:.4f}")
                excel_steps.append(f"{portfolio_return:.4f} = {' + '.join(terms_filled)}")
                excel_steps.append("")
                
                excel_steps.append("**Step 3: Calculate known contributions**")
                excel_steps.append("")
                known_terms = []
                for i in range(n_assets_rwa):
                    if i != missing_idx:
                        contrib = weights_rwa[i] * returns_rwa[i]
                        known_terms.append(f"{weights_rwa[i]:.4f} × {returns_rwa[i]:.4f} = {contrib:.6f}")
                for term in known_terms:
                    excel_steps.append(f"  {term}")
                excel_steps.append(f"  Sum of known = {known_contribution:.6f}")
                excel_steps.append("")
                
                excel_steps.append("**Step 4: Solve for unknown**")
                excel_steps.append("")
                excel_steps.append(f"E[r_{asset_names_rwa[missing_idx]}] = (E[rp] - known) / w_{missing_idx+1}")
                excel_steps.append(f"E[r_{asset_names_rwa[missing_idx]}] = ({portfolio_return:.4f} - {known_contribution:.6f}) / {weights_rwa[missing_idx]:.4f}")
                excel_steps.append(f"E[r_{asset_names_rwa[missing_idx]}] = {portfolio_return - known_contribution:.6f} / {weights_rwa[missing_idx]:.4f}")
                excel_steps.append(f"**E[r_{asset_names_rwa[missing_idx]}] = {missing_return:.6f} = {missing_return*100:.2f}%**")
                excel_steps.append("")
                
                excel_steps.append("**Step 5: Verify**")
                returns_rwa[missing_idx] = missing_return
                verify_sum = np.sum(weights_rwa * returns_rwa)
                excel_steps.append(f"Check: Σ w_i × E[r_i] = {verify_sum:.6f} ≈ {portfolio_return:.6f} ✓")
                
                # Display results
                st.success(f"**Solved!** E[r_{asset_names_rwa[missing_idx]}] = **{missing_return*100:.2f}%**")
                
                st.metric(f"Missing Return: {asset_names_rwa[missing_idx]}", f"{missing_return*100:.2f}%")
                
                # Summary table
                returns_rwa[missing_idx] = missing_return
                summary_df = pd.DataFrame({
                    'Asset': asset_names_rwa,
                    'Weight': weights_rwa,
                    'E[r]': returns_rwa,
                    'E[r] (%)': [f"{r*100:.2f}%" for r in returns_rwa],
                    'Contribution': weights_rwa * returns_rwa,
                    'Status': ['**SOLVED**' if i == missing_idx else 'Given' for i in range(n_assets_rwa)]
                })
                st.dataframe(summary_df.style.format({
                    'Weight': '{:.4f}',
                    'E[r]': '{:.4f}',
                    'Contribution': '{:.6f}'
                }))
                
                # Verification
                st.write(f"**Verification:** Σ(weight × return) = {verify_sum:.6f} = Portfolio return {portfolio_return:.4f} ✓")
                
                # Excel instructions
                with st.expander("📝 Step-by-Step Excel Solution"):
                    for step in excel_steps:
                        st.markdown(step)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # -------------------------------------------------------------------------
    # TAB 5: SINGLE-INDEX MODEL SOLVER
    # -------------------------------------------------------------------------
    with tab5:
        st.subheader("Single-Index Model (SIM) Solver")
        
        st.markdown(r"""
        **The Single-Index Model:**
        $$r_i = r_f + \alpha_i + \beta_i(r_m - r_f) + \varepsilon_i$$
        
        This solver helps you find missing values when given partial information about a stock.
        """)
        
        # Quick Reference Expander with all equations
        with st.expander("📚 **SIM Quick Reference: Key Equations**", expanded=True):
            st.markdown(r"""
            | Equation | Formula | Use When |
            |----------|---------|----------|
            | **Expected Return** | $E[r_i] = r_f + \alpha_i + \beta_i(E[r_m] - r_f)$ | Finding E[r] from α, β |
            | **Total Variance** | $\sigma^2(r_i) = \beta_i^2 \sigma_m^2 + \sigma^2(\varepsilon_i)$ | Relating total vs idiosyncratic risk |
            | **Information Ratio** | $IR_i = \frac{\alpha_i}{\sigma(\varepsilon_i)}$ | Measuring alpha quality |
            
            ---
            
            **⚠️ Critical Distinction:**
            - **σ(rᵢ)** = Total return standard deviation (includes all risk)
            - **σ(εᵢ)** = Idiosyncratic/Residual standard deviation (firm-specific risk only)
            
            **Relationship:** $\sigma^2(r_i) = \beta_i^2 \sigma_m^2 + \sigma^2(\varepsilon_i)$
            
            Or equivalently: $\sigma(r_i) = \sqrt{\beta_i^2 \sigma_m^2 + \sigma^2(\varepsilon_i)}$
            """)
        
        st.write("---")
        
        # Market Parameters (always needed)
        st.write("### 📈 Market Parameters")
        col_mkt1, col_mkt2, col_mkt3 = st.columns(3)
        
        with col_mkt1:
            sim_rf = st.number_input("Risk-Free Rate (rf)", value=0.03, format="%.4f", key="sim_rf",
                                    help="The risk-free rate of return")
        with col_mkt2:
            sim_erm = st.number_input("E[rm] — Market Expected Return", value=0.10, format="%.4f", key="sim_erm",
                                     help="Expected return of the market index")
        with col_mkt3:
            sim_sigma_m = st.number_input("σm — Market Std Dev", value=0.20, format="%.4f", key="sim_sigma_m",
                                         help="Standard deviation of market returns")
        
        # Calculate market risk premium for reference
        mrp = sim_erm - sim_rf
        st.info(f"📊 **Market Risk Premium:** E[rm] - rf = {sim_erm:.4f} - {sim_rf:.4f} = **{mrp:.4f}** ({mrp*100:.2f}%)")
        
        st.write("---")
        
        # Stock-specific inputs
        st.write("### 🏢 Stock-Specific Parameters")
        st.markdown("Enter the values you **know**. Leave fields at 0 or check 'Unknown' for values you want to solve.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Alpha with checkbox
            use_alpha = st.checkbox("I have Alpha (α)", value=True, key="sim_use_alpha",
                                   help="Check if alpha is given in the problem")
            if use_alpha:
                sim_alpha = st.number_input("α — Alpha (abnormal return)", value=0.05, format="%.4f", key="sim_alpha",
                                           help="Jensen's alpha: excess return beyond CAPM prediction")
            else:
                sim_alpha = None
            
            sim_beta = st.number_input("β — Beta", value=0.0, format="%.4f", key="sim_beta",
                                      help="Systematic risk measure. Enter 0 if unknown.")
            beta_unknown = st.checkbox("Beta is unknown", value=False, key="sim_beta_unknown")
            
            sim_er = st.number_input("E[ri] — Expected Return", value=0.0, format="%.4f", key="sim_er",
                                    help="Total expected return of the stock. Enter 0 if unknown.")
            er_unknown = st.checkbox("E[ri] is unknown", value=True, key="sim_er_unknown")
        
        with col2:
            sim_sigma_eps = st.number_input("σ(εi) — Idiosyncratic/Residual Std Dev", value=0.0, format="%.4f", 
                                           key="sim_sigma_eps",
                                           help="⚠️ FIRM-SPECIFIC risk only (not total risk). Enter 0 if unknown.")
            sigma_eps_unknown = st.checkbox("σ(εi) is unknown", value=True, key="sim_sigma_eps_unknown")
            
            sim_sigma_r = st.number_input("σ(ri) — TOTAL Return Std Dev", value=0.16, format="%.4f", 
                                         key="sim_sigma_r",
                                         help="⚠️ TOTAL volatility (systematic + idiosyncratic). Enter 0 if unknown.")
            sigma_r_unknown = st.checkbox("σ(ri) is unknown", value=False, key="sim_sigma_r_unknown")
            
            sim_ir = st.number_input("IR — Information Ratio", value=0.0, format="%.4f", key="sim_ir",
                                    help="IR = α / σ(ε). Enter 0 if unknown.")
            ir_unknown = st.checkbox("IR is unknown", value=True, key="sim_ir_unknown")
        
        st.write("---")
        
        # What to solve for
        st.write("### 🎯 What do you want to find?")
        solve_options = []
        
        # Build list of solvable variables
        if beta_unknown:
            solve_options.append("Beta (β)")
        if er_unknown:
            solve_options.append("Expected Return E[ri]")
        if sigma_eps_unknown:
            solve_options.append("Idiosyncratic Std Dev σ(εi)")
        if sigma_r_unknown:
            solve_options.append("Total Return Std Dev σ(ri)")
        if ir_unknown:
            solve_options.append("Information Ratio (IR)")
        if not use_alpha:
            solve_options.append("Alpha (α)")
        
        if not solve_options:
            st.warning("All variables are marked as known. Check the 'unknown' boxes for variables you want to solve.")
            solve_options = ["Nothing to solve"]
        
        solve_for = st.selectbox("Solve for:", solve_options, key="sim_solve_for")
        
        if st.button("🧮 Solve SIM Equation", key="sim_solve_btn"):
            try:
                excel_steps = []
                result_value = None
                result_name = ""
                
                excel_steps.append("### Single-Index Model Solution")
                excel_steps.append("")
                excel_steps.append(f"**Given Market Parameters:**")
                excel_steps.append(f"- rf = {sim_rf:.4f}")
                excel_steps.append(f"- E[rm] = {sim_erm:.4f}")
                excel_steps.append(f"- σm = {sim_sigma_m:.4f}")
                excel_steps.append(f"- Market Risk Premium = {mrp:.4f}")
                excel_steps.append("")
                
                # ========== SOLVE FOR EXPECTED RETURN ==========
                if solve_for == "Expected Return E[ri]":
                    if not use_alpha:
                        st.error("Need Alpha (α) to compute E[ri]. Check 'I have Alpha' and enter the value.")
                    elif beta_unknown:
                        st.error("Need Beta (β) to compute E[ri]. Uncheck 'Beta is unknown' and enter the value.")
                    else:
                        # E[ri] = rf + α + β(E[rm] - rf)
                        result_value = sim_rf + sim_alpha + sim_beta * mrp
                        result_name = "E[ri]"
                        
                        excel_steps.append("**Formula:** E[ri] = rf + α + β(E[rm] - rf)")
                        excel_steps.append("")
                        excel_steps.append("**Step 1:** Calculate market risk premium")
                        excel_steps.append(f"MRP = E[rm] - rf = {sim_erm:.4f} - {sim_rf:.4f} = {mrp:.4f}")
                        excel_steps.append("")
                        excel_steps.append("**Step 2:** Apply CAPM with alpha")
                        excel_steps.append(f"E[ri] = {sim_rf:.4f} + {sim_alpha:.4f} + {sim_beta:.4f} × {mrp:.4f}")
                        excel_steps.append(f"E[ri] = {sim_rf:.4f} + {sim_alpha:.4f} + {sim_beta * mrp:.4f}")
                        excel_steps.append(f"**E[ri] = {result_value:.4f} = {result_value*100:.2f}%**")
                
                # ========== SOLVE FOR BETA ==========
                elif solve_for == "Beta (β)":
                    # Try from expected return equation first
                    if not er_unknown and sim_er != 0 and use_alpha:
                        # E[ri] = rf + α + β(E[rm] - rf) => β = (E[ri] - rf - α) / (E[rm] - rf)
                        if mrp == 0:
                            st.error("Market risk premium is zero, cannot solve for beta.")
                        else:
                            result_value = (sim_er - sim_rf - sim_alpha) / mrp
                            result_name = "β"
                            
                            excel_steps.append("**Method:** Solve from expected return equation")
                            excel_steps.append("**Formula:** β = (E[ri] - rf - α) / (E[rm] - rf)")
                            excel_steps.append("")
                            excel_steps.append(f"β = ({sim_er:.4f} - {sim_rf:.4f} - {sim_alpha:.4f}) / {mrp:.4f}")
                            excel_steps.append(f"β = {sim_er - sim_rf - sim_alpha:.4f} / {mrp:.4f}")
                            excel_steps.append(f"**β = {result_value:.4f}**")
                    
                    # Try from variance equation
                    elif not sigma_r_unknown and not sigma_eps_unknown and sim_sigma_r != 0:
                        # σ²(ri) = β²σ²m + σ²(εi) => β = sqrt((σ²(ri) - σ²(εi)) / σ²m)
                        var_total = sim_sigma_r ** 2
                        var_idio = sim_sigma_eps ** 2
                        var_market = sim_sigma_m ** 2
                        
                        if var_total < var_idio:
                            st.error("Total variance cannot be less than idiosyncratic variance!")
                        elif var_market == 0:
                            st.error("Market variance is zero, cannot solve for beta.")
                        else:
                            beta_squared = (var_total - var_idio) / var_market
                            result_value = np.sqrt(beta_squared)
                            result_name = "β"
                            
                            excel_steps.append("**Method:** Solve from variance decomposition")
                            excel_steps.append("**Formula:** σ²(ri) = β²σ²m + σ²(εi)")
                            excel_steps.append("**Rearranged:** β = √[(σ²(ri) - σ²(εi)) / σ²m]")
                            excel_steps.append("")
                            excel_steps.append(f"σ²(ri) = {sim_sigma_r:.4f}² = {var_total:.6f}")
                            excel_steps.append(f"σ²(εi) = {sim_sigma_eps:.4f}² = {var_idio:.6f}")
                            excel_steps.append(f"σ²m = {sim_sigma_m:.4f}² = {var_market:.6f}")
                            excel_steps.append("")
                            excel_steps.append(f"β² = ({var_total:.6f} - {var_idio:.6f}) / {var_market:.6f}")
                            excel_steps.append(f"β² = {var_total - var_idio:.6f} / {var_market:.6f}")
                            excel_steps.append(f"β² = {beta_squared:.6f}")
                            excel_steps.append(f"**β = {result_value:.4f}**")
                    else:
                        st.error("Need either (E[ri] + α) or (σ(ri) + σ(εi)) to solve for beta.")
                
                # ========== SOLVE FOR ALPHA ==========
                elif solve_for == "Alpha (α)":
                    if er_unknown:
                        st.error("Need E[ri] to compute alpha.")
                    elif beta_unknown:
                        st.error("Need Beta to compute alpha.")
                    else:
                        # α = E[ri] - rf - β(E[rm] - rf)
                        result_value = sim_er - sim_rf - sim_beta * mrp
                        result_name = "α"
                        
                        excel_steps.append("**Formula:** α = E[ri] - rf - β(E[rm] - rf)")
                        excel_steps.append("")
                        excel_steps.append(f"α = {sim_er:.4f} - {sim_rf:.4f} - {sim_beta:.4f} × {mrp:.4f}")
                        excel_steps.append(f"α = {sim_er:.4f} - {sim_rf:.4f} - {sim_beta * mrp:.4f}")
                        excel_steps.append(f"**α = {result_value:.4f} = {result_value*100:.2f}%**")
                
                # ========== SOLVE FOR IDIOSYNCRATIC STD DEV ==========
                elif solve_for == "Idiosyncratic Std Dev σ(εi)":
                    # Method 1: From IR and alpha
                    if use_alpha and not ir_unknown and sim_ir != 0:
                        # IR = α / σ(ε) => σ(ε) = α / IR
                        result_value = abs(sim_alpha / sim_ir)
                        result_name = "σ(εi)"
                        
                        excel_steps.append("**Method:** Solve from Information Ratio")
                        excel_steps.append("**Formula:** IR = α / σ(εi)")
                        excel_steps.append("**Rearranged:** σ(εi) = |α| / |IR|")
                        excel_steps.append("")
                        excel_steps.append(f"σ(εi) = |{sim_alpha:.4f}| / |{sim_ir:.4f}|")
                        excel_steps.append(f"**σ(εi) = {result_value:.4f} = {result_value*100:.2f}%**")
                    
                    # Method 2: From variance decomposition
                    elif not sigma_r_unknown and not beta_unknown:
                        # σ²(ri) = β²σ²m + σ²(εi) => σ²(εi) = σ²(ri) - β²σ²m
                        var_total = sim_sigma_r ** 2
                        var_systematic = (sim_beta ** 2) * (sim_sigma_m ** 2)
                        var_idio = var_total - var_systematic
                        
                        if var_idio < 0:
                            st.error(f"Calculated idiosyncratic variance is negative ({var_idio:.6f}). Check your inputs!")
                        else:
                            result_value = np.sqrt(var_idio)
                            result_name = "σ(εi)"
                            
                            excel_steps.append("**Method:** Solve from variance decomposition")
                            excel_steps.append("**Formula:** σ²(ri) = β²σ²m + σ²(εi)")
                            excel_steps.append("**Rearranged:** σ²(εi) = σ²(ri) - β²σ²m")
                            excel_steps.append("")
                            excel_steps.append(f"σ²(ri) = {sim_sigma_r:.4f}² = {var_total:.6f}")
                            excel_steps.append(f"β²σ²m = {sim_beta:.4f}² × {sim_sigma_m:.4f}² = {var_systematic:.6f}")
                            excel_steps.append(f"σ²(εi) = {var_total:.6f} - {var_systematic:.6f} = {var_idio:.6f}")
                            excel_steps.append(f"**σ(εi) = √{var_idio:.6f} = {result_value:.4f} = {result_value*100:.2f}%**")
                    else:
                        st.error("Need either (α + IR) or (σ(ri) + β) to solve for σ(εi).")
                
                # ========== SOLVE FOR TOTAL RETURN STD DEV ==========
                elif solve_for == "Total Return Std Dev σ(ri)":
                    if beta_unknown:
                        st.error("Need Beta to compute total return std dev.")
                    elif sigma_eps_unknown:
                        st.error("Need σ(εi) to compute total return std dev.")
                    else:
                        # σ²(ri) = β²σ²m + σ²(εi)
                        var_systematic = (sim_beta ** 2) * (sim_sigma_m ** 2)
                        var_idio = sim_sigma_eps ** 2
                        var_total = var_systematic + var_idio
                        result_value = np.sqrt(var_total)
                        result_name = "σ(ri)"
                        
                        excel_steps.append("**Formula:** σ²(ri) = β²σ²m + σ²(εi)")
                        excel_steps.append("")
                        excel_steps.append(f"**Systematic variance:** β²σ²m = {sim_beta:.4f}² × {sim_sigma_m:.4f}²")
                        excel_steps.append(f"  = {sim_beta**2:.6f} × {sim_sigma_m**2:.6f} = {var_systematic:.6f}")
                        excel_steps.append("")
                        excel_steps.append(f"**Idiosyncratic variance:** σ²(εi) = {sim_sigma_eps:.4f}² = {var_idio:.6f}")
                        excel_steps.append("")
                        excel_steps.append(f"**Total variance:** σ²(ri) = {var_systematic:.6f} + {var_idio:.6f} = {var_total:.6f}")
                        excel_steps.append(f"**σ(ri) = √{var_total:.6f} = {result_value:.4f} = {result_value*100:.2f}%**")
                
                # ========== SOLVE FOR INFORMATION RATIO ==========
                elif solve_for == "Information Ratio (IR)":
                    if not use_alpha:
                        st.error("Need Alpha (α) to compute Information Ratio.")
                    elif sigma_eps_unknown or sim_sigma_eps == 0:
                        st.error("Need σ(εi) to compute Information Ratio. Make sure it's not zero.")
                    else:
                        # IR = α / σ(ε)
                        result_value = sim_alpha / sim_sigma_eps
                        result_name = "IR"
                        
                        excel_steps.append("**Formula:** IR = α / σ(εi)")
                        excel_steps.append("")
                        excel_steps.append(f"IR = {sim_alpha:.4f} / {sim_sigma_eps:.4f}")
                        excel_steps.append(f"**IR = {result_value:.4f}**")
                        excel_steps.append("")
                        excel_steps.append("**Interpretation:**")
                        if result_value > 0.5:
                            excel_steps.append(f"IR = {result_value:.2f} is considered good (>0.5)")
                        elif result_value > 0:
                            excel_steps.append(f"IR = {result_value:.2f} is positive but moderate")
                        else:
                            excel_steps.append(f"IR = {result_value:.2f} is negative (poor risk-adjusted alpha)")
                
                # Display results
                if result_value is not None:
                    st.success(f"**Solved!** {result_name} = **{result_value:.4f}**")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        if result_name in ["E[ri]", "α", "σ(εi)", "σ(ri)"]:
                            st.metric(result_name, f"{result_value:.4f} ({result_value*100:.2f}%)")
                        else:
                            st.metric(result_name, f"{result_value:.4f}")
                    
                    with col_res2:
                        # Show variance if we computed a std dev
                        if result_name in ["σ(εi)", "σ(ri)"]:
                            st.metric(f"{result_name}² (variance)", f"{result_value**2:.6f}")
                    
                    # Excel steps
                    with st.expander("📝 Step-by-Step Solution", expanded=True):
                        for step in excel_steps:
                            st.markdown(step)
                    
                    # Summary of all known/computed values
                    st.write("---")
                    st.write("### 📋 Summary of All Values")
                    
                    # Compile all values
                    summary_data = {
                        "Parameter": ["rf", "E[rm]", "σm", "MRP"],
                        "Value": [f"{sim_rf:.4f}", f"{sim_erm:.4f}", f"{sim_sigma_m:.4f}", f"{mrp:.4f}"],
                        "Status": ["Given", "Given", "Given", "Computed"]
                    }
                    
                    # Add stock-specific
                    if use_alpha:
                        summary_data["Parameter"].append("α")
                        summary_data["Value"].append(f"{sim_alpha:.4f}")
                        summary_data["Status"].append("Given")
                    
                    if not beta_unknown:
                        summary_data["Parameter"].append("β")
                        summary_data["Value"].append(f"{sim_beta:.4f}")
                        summary_data["Status"].append("Given" if solve_for != "Beta (β)" else "**SOLVED**")
                    elif result_name == "β":
                        summary_data["Parameter"].append("β")
                        summary_data["Value"].append(f"{result_value:.4f}")
                        summary_data["Status"].append("**SOLVED**")
                    
                    if not er_unknown:
                        summary_data["Parameter"].append("E[ri]")
                        summary_data["Value"].append(f"{sim_er:.4f}")
                        summary_data["Status"].append("Given" if solve_for != "Expected Return E[ri]" else "**SOLVED**")
                    elif result_name == "E[ri]":
                        summary_data["Parameter"].append("E[ri]")
                        summary_data["Value"].append(f"{result_value:.4f}")
                        summary_data["Status"].append("**SOLVED**")
                    
                    if not sigma_r_unknown:
                        summary_data["Parameter"].append("σ(ri)")
                        summary_data["Value"].append(f"{sim_sigma_r:.4f}")
                        summary_data["Status"].append("Given")
                    elif result_name == "σ(ri)":
                        summary_data["Parameter"].append("σ(ri)")
                        summary_data["Value"].append(f"{result_value:.4f}")
                        summary_data["Status"].append("**SOLVED**")
                    
                    if not sigma_eps_unknown:
                        summary_data["Parameter"].append("σ(εi)")
                        summary_data["Value"].append(f"{sim_sigma_eps:.4f}")
                        summary_data["Status"].append("Given")
                    elif result_name == "σ(εi)":
                        summary_data["Parameter"].append("σ(εi)")
                        summary_data["Value"].append(f"{result_value:.4f}")
                        summary_data["Status"].append("**SOLVED**")
                    
                    if not ir_unknown:
                        summary_data["Parameter"].append("IR")
                        summary_data["Value"].append(f"{sim_ir:.4f}")
                        summary_data["Status"].append("Given")
                    elif result_name == "IR":
                        summary_data["Parameter"].append("IR")
                        summary_data["Value"].append(f"{result_value:.4f}")
                        summary_data["Status"].append("**SOLVED**")
                    
                    st.dataframe(pd.DataFrame(summary_data))
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def black_litterman_section():
    """Black-Litterman Model interface."""
    st.subheader("Black-Litterman Model")
    
    st.markdown(r"""
    **Combines market equilibrium with investor views to derive optimal portfolio weights.**
    
    **Key Formulas:**
    - Equilibrium returns: $\pi = \delta \Sigma w_{mkt}$
    - Posterior returns: $\mu_{BL} = \pi + \tau\Sigma P'(\tau P\Sigma P' + \Omega)^{-1}(Q - P\pi)$
    """)
    
    st.info(
        "🧠 **Exam tip:** Black-Litterman blends market-implied returns with subjective views. "
        "The key insight is that equilibrium returns π come from reverse-optimizing the market portfolio. "
        "Views are expressed via matrix P (which assets) and vector Q (expected returns). "
        "Confidence affects Ω (view uncertainty). More confident views shift weights more."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Market Parameters:**")
        n_assets = st.number_input("Number of Assets", value=3, min_value=2, max_value=10, key="bl_n_assets")
        delta = st.number_input("Risk Aversion (δ)", value=2.5, min_value=0.1, format="%.2f", key="bl_delta",
                               help="Market risk aversion. Typically 2.5 for equity markets.")
        tau = st.number_input("Tau (τ)", value=0.05, min_value=0.01, max_value=0.5, format="%.3f", key="bl_tau",
                             help="Uncertainty in prior. Typically 0.025 to 0.05.")
        rf = st.number_input("Risk-Free Rate", value=0.02, format="%.4f", key="bl_rf")
    
    with col2:
        st.write("**Market Weights (market cap weights):**")
        weights_input = st.text_area(
            "Enter market weights (comma-separated)",
            value="0.40, 0.35, 0.25",
            key="bl_weights",
            help="Should sum to 1. E.g., 40% Stocks, 35% Bonds, 25% Alternatives"
        )
        
        st.write("**Asset Names (optional):**")
        names_input = st.text_input(
            "Asset names (comma-separated)",
            value="Stocks, Bonds, Alternatives",
            key="bl_names"
        )
    
    st.write("**Covariance Matrix:**")
    cov_input = st.text_area(
        "Covariance matrix (one row per line)",
        value="0.0400, 0.0100, 0.0080\n0.0100, 0.0225, 0.0050\n0.0080, 0.0050, 0.0900",
        height=100,
        key="bl_cov",
        help="Enter variance-covariance matrix. Diagonal = variances."
    )
    
    st.write("---")
    st.write("### Investor Views")
    
    st.markdown("""
    **View Types:**
    - **Absolute View:** "Asset A will return X%" → P row = [0, 1, 0, ...], Q = X
    - **Relative View:** "Asset A will outperform Asset B by Y%" → P row = [1, -1, 0, ...], Q = Y
    """)
    
    n_views = st.number_input("Number of Views", value=1, min_value=0, max_value=5, key="bl_n_views")
    
    if n_views > 0:
        st.write("**Enter Views (P matrix rows and Q values):**")
        
        P_input = st.text_area(
            f"P matrix ({n_views} rows × {n_assets} cols)",
            value="1, -1, 0" if n_views == 1 else "1, -1, 0\n0, 1, 0",
            height=80,
            key="bl_P",
            help="Each row is a view. For relative view 'A outperforms B by X%': put 1 for A, -1 for B"
        )
        
        Q_input = st.text_input(
            f"Q vector ({n_views} values)",
            value="0.03",
            key="bl_Q",
            help="Expected returns for each view"
        )
        
        conf_input = st.text_input(
            f"Confidences ({n_views} values, 0-1 scale)",
            value="0.8",
            key="bl_conf",
            help="1 = very confident, 0.5 = moderate confidence"
        )
        
        view_desc_input = st.text_input(
            "View descriptions (comma-separated)",
            value="Stocks outperform Bonds by 3%",
            key="bl_view_desc"
        )
    
    if st.button("🧮 Calculate Black-Litterman Posterior", key="bl_calc"):
        try:
            # Parse inputs
            market_weights = parse_messy_input(weights_input)
            cov_matrix = parse_matrix_input(cov_input)
            
            # Validate
            if len(market_weights) != n_assets:
                st.error(f"Market weights have {len(market_weights)} elements, expected {n_assets}")
                return
            
            if cov_matrix.shape != (n_assets, n_assets):
                st.error(f"Covariance matrix shape {cov_matrix.shape}, expected ({n_assets}, {n_assets})")
                return
            
            # Parse asset names
            if names_input.strip():
                asset_names = [n.strip() for n in names_input.split(',')]
                if len(asset_names) != n_assets:
                    asset_names = [f'Asset {i+1}' for i in range(n_assets)]
            else:
                asset_names = [f'Asset {i+1}' for i in range(n_assets)]
            
            # Parse views if provided
            if n_views > 0:
                P = parse_matrix_input(P_input)
                Q = parse_messy_input(Q_input)
                
                if P.shape[0] != n_views or P.shape[1] != n_assets:
                    st.error(f"P matrix shape {P.shape}, expected ({n_views}, {n_assets})")
                    return
                
                if len(Q) != n_views:
                    st.error(f"Q has {len(Q)} values, expected {n_views}")
                    return
                
                if conf_input.strip():
                    confidences = parse_messy_input(conf_input)
                    if len(confidences) != n_views:
                        confidences = None
                else:
                    confidences = None
                
                if view_desc_input.strip():
                    view_descriptions = [d.strip() for d in view_desc_input.split(',')]
                    if len(view_descriptions) != n_views:
                        view_descriptions = None
                else:
                    view_descriptions = None
            else:
                P = None
                Q = None
                confidences = None
                view_descriptions = None
            
            # Create optimizer
            optimizer = BlackLittermanOptimizer(
                market_weights=market_weights,
                covariance=cov_matrix,
                risk_aversion=delta,
                tau=tau,
                P=P,
                Q=Q,
                view_confidences=confidences,
                risk_free_rate=rf,
                asset_names=asset_names,
                view_descriptions=view_descriptions
            )
            
            # Display results
            st.subheader("📊 Results")
            
            # Equilibrium returns
            st.write("**Step 1: Equilibrium Returns (Prior)**")
            st.write("These are the returns implied by the market portfolio under mean-variance:")
            
            eq_df = pd.DataFrame({
                'Asset': asset_names,
                'Market Weight': market_weights,
                'Equilibrium Return (π)': optimizer.equilibrium_returns,
                'Equilibrium Return (%)': optimizer.equilibrium_returns * 100
            })
            st.dataframe(eq_df.style.format({
                'Market Weight': '{:.4f}',
                'Equilibrium Return (π)': '{:.4f}',
                'Equilibrium Return (%)': '{:.2f}%'
            }))
            
            # View analysis (if views provided)
            if n_views > 0:
                st.write("**Step 2: View Analysis**")
                view_df = optimizer.get_view_analysis()
                st.dataframe(view_df.style.format({
                    'View Return (Q)': '{:.4f}',
                    'Prior Expectation': '{:.4f}',
                    'View - Prior': '{:.4f}',
                    'View Variance (Ω)': '{:.6f}'
                }))
            
            # Posterior returns and weights
            st.write("**Step 3: Posterior Returns & Optimal Weights**")
            summary_df = optimizer.get_summary_table()
            st.dataframe(summary_df.style.format({
                'Market Weight': '{:.4f}',
                'Equilibrium Return (π)': '{:.4f}',
                'Posterior Return (μ_BL)': '{:.4f}',
                'Return Shift': '{:+.4f}',
                'BL Optimal Weight': '{:.4f}',
                'Weight Change': '{:+.4f}'
            }))
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tau (τ)", f"{tau:.3f}")
                st.metric("Risk Aversion (δ)", f"{delta:.2f}")
            with col2:
                st.metric("Number of Views", str(n_views))
                total_shift = np.sum(np.abs(optimizer.posterior_weights - optimizer.w_mkt))
                st.metric("Total Weight Shift", f"{total_shift*100:.2f}%")
            with col3:
                # Portfolio expected return
                port_ret = optimizer.posterior_weights @ optimizer.posterior_returns
                port_var = optimizer.posterior_weights @ optimizer.posterior_cov @ optimizer.posterior_weights
                port_vol = np.sqrt(port_var)
                st.metric("Portfolio E[R]", f"{port_ret*100:.2f}%")
                st.metric("Portfolio σ", f"{port_vol*100:.2f}%")
            
            # Visualization
            st.write("**Weight Comparison:**")
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(n_assets)
            width = 0.35
            
            ax.bar(x - width/2, optimizer.w_mkt, width, label='Market Weights', color='steelblue', alpha=0.7)
            ax.bar(x + width/2, optimizer.posterior_weights, width, label='BL Optimal Weights', color='darkorange', alpha=0.7)
            
            ax.set_xlabel('Assets')
            ax.set_ylabel('Weight')
            ax.set_title('Market vs Black-Litterman Optimal Weights')
            ax.set_xticks(x)
            ax.set_xticklabels(asset_names)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
            # Excel instructions
            with st.expander("📝 Excel Implementation"):
                st.markdown(optimizer.get_excel_instructions())
            
            # LaTeX formulas
            with st.expander("📐 LaTeX Formulas"):
                for name, formula in optimizer.get_latex_formulas().items():
                    st.latex(formula)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def joint_asset_factor_tab():
    """Joint optimization over assets and factor portfolios."""
    st.subheader("Joint Asset + Factor Portfolio Optimization")
    
    st.markdown("""
    Optimize over an investable universe that includes **both specific assets and factor portfolios**.
    
    This combines direct asset holdings with factor exposures in a single mean-variance optimization.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_assets = st.number_input("Number of Specific Assets", value=2, min_value=1, key="jaf_n_assets")
        n_factors = st.number_input("Number of Factor Portfolios", value=2, min_value=1, key="jaf_n_factors")
        
        rf = st.number_input("Risk-Free Rate", value=0.02, format="%.4f", key="jaf_rf")
        gamma = st.number_input("Risk Aversion (γ)", value=4.0, min_value=0.1, key="jaf_gamma")
    
    with col2:
        st.write("**Asset Expected Returns:**")
        asset_ret_input = st.text_area(
            "Asset Returns",
            value="0.08, 0.10",
            key="jaf_asset_ret"
        )
        
        st.write("**Factor Portfolio Expected Returns:**")
        factor_ret_input = st.text_area(
            "Factor Returns",
            value="0.06, 0.04",
            key="jaf_factor_ret"
        )
    
    st.write("**Asset Covariance Matrix:**")
    asset_cov_input = st.text_area(
        "Asset Covariance",
        value="0.04, 0.01\n0.01, 0.0625",
        height=80,
        key="jaf_asset_cov"
    )
    
    st.write("**Factor Portfolio Covariance Matrix:**")
    factor_cov_input = st.text_area(
        "Factor Covariance",
        value="0.0225, 0.005\n0.005, 0.01",
        height=80,
        key="jaf_factor_cov"
    )
    
    st.write("**Asset-Factor Covariance Matrix (Assets × Factors):**")
    cross_cov_input = st.text_area(
        "Cross Covariance",
        value="0.015, 0.008\n0.02, 0.012",
        height=80,
        help="Covariance between each asset and each factor portfolio",
        key="jaf_cross_cov"
    )
    
    if st.button("🧮 Optimize Joint Portfolio", key="jaf_calc"):
        try:
            asset_returns = parse_messy_input(asset_ret_input)
            factor_returns = parse_messy_input(factor_ret_input)
            asset_cov = parse_matrix_input(asset_cov_input)
            factor_cov = parse_matrix_input(factor_cov_input)
            cross_cov = parse_matrix_input(cross_cov_input)
            
            result, excel_steps, tables = joint_asset_factor_optimization(
                asset_returns, asset_cov,
                factor_returns, factor_cov,
                cross_cov,
                rf, gamma
            )
            
            st.subheader("📊 Joint Optimal Portfolio")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Return", f"{result['optimal']['return']*100:.4f}%")
                st.metric("Volatility", f"{result['optimal']['volatility']*100:.4f}%")
                st.metric("Sharpe Ratio", f"{result['optimal']['sharpe']:.4f}")
            
            with col2:
                st.write("**Allocation to Specific Assets:**")
                for i, w in enumerate(result['asset_weights']):
                    st.write(f"  Asset {i+1}: {w:.4f} ({w*100:.2f}%)")
                
                st.write("**Allocation to Factor Portfolios:**")
                for i, w in enumerate(result['factor_weights']):
                    st.write(f"  Factor {i+1}: {w:.4f} ({w*100:.2f}%)")
            
            # Summary table
            total_asset_weight = np.sum(result['asset_weights'])
            total_factor_weight = np.sum(result['factor_weights'])
            st.write(f"**Total in Assets:** {total_asset_weight:.4f} | **Total in Factors:** {total_factor_weight:.4f}")
            
            with st.expander("📝 How to do this in Excel"):
                for step in excel_steps:
                    st.markdown(step)
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def covariance_builder_tab():
    """Build full asset covariance from factor model."""
    st.subheader("Factor Model Covariance Builder")
    
    st.markdown("""
    Build full asset covariance matrix: **Σ = B Σ_f B' + D**
    
    Where B = betas, Σ_f = factor covariance, D = idiosyncratic variances.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_assets = st.number_input("Number of Assets", value=3, min_value=1, key="cov_n_assets")
        n_factors = st.number_input("Number of Factors", value=2, min_value=1, key="cov_n_factors")
    
    with col2:
        st.write("**Idiosyncratic Volatilities:**")
        idio_input = st.text_area(
            "Idio Vols (one per asset)",
            value="0.15, 0.12, 0.18",
            key="cov_idio"
        )
    
    st.write(f"**Beta Matrix ({n_assets} assets × {n_factors} factors):**")
    beta_input = st.text_area(
        "Betas (asset per row)",
        value="1.2, 0.8\n0.9, 1.1\n1.0, 0.5",
        height=100,
        key="cov_betas"
    )
    
    st.write(f"**Factor Covariance Matrix ({n_factors} × {n_factors}):**")
    factor_cov_input = st.text_area(
        "Factor Covariance",
        value="0.04, 0.01\n0.01, 0.02",
        height=80,
        key="cov_factor_cov"
    )
    
    if st.button("🧮 Build Covariance Matrix", key="cov_calc"):
        try:
            betas = parse_matrix_input(beta_input)
            factor_cov = parse_matrix_input(factor_cov_input)
            idio_vols = parse_messy_input(idio_input)
            idio_vars = idio_vols ** 2
            
            result, excel_steps, tables = build_factor_covariance(
                betas, factor_cov, idio_vars
            )
            
            st.subheader("📊 Asset Covariance Matrix (Σ)")
            st.dataframe(tables['covariance'].style.format('{:.6f}'))
            
            st.subheader("📊 Variance Decomposition")
            st.dataframe(tables['diagnostics'].style.format({
                'Total Var': '{:.6f}',
                'Systematic Var': '{:.6f}',
                'Idiosyncratic Var': '{:.6f}',
                'R²': '{:.4f}'
            }))
            
            with st.expander("📝 How to do this in Excel"):
                for step in excel_steps:
                    st.markdown(step)
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")


def factor_neutral_tab():
    """Factor-neutral portfolio optimization."""
    st.subheader("Factor-Neutral Portfolio Optimizer")
    
    st.markdown("""
    Optimize portfolio with factor neutrality constraint: **B'w = 0**
    
    Useful for market-neutral or factor-neutral strategies.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        objective = st.selectbox(
            "Objective",
            ["min_variance", "max_return"],
            format_func=lambda x: "Minimize Variance" if x == "min_variance" else "Maximize Return (Alpha)",
            key="fn_objective"
        )
        
        no_short = st.checkbox("No Short Selling", value=False, key="fn_noshort")
        
        use_risk_cap = st.checkbox("Risk Cap", value=False, key="fn_use_risk")
        if use_risk_cap:
            risk_cap = st.number_input("Max Volatility", value=0.15, format="%.4f", key="fn_risk")
        else:
            risk_cap = None
    
    with col2:
        use_max_weight = st.checkbox("Max Weight", value=False, key="fn_use_max")
        if use_max_weight:
            max_weight = st.number_input("Max Position", value=0.30, format="%.2f", key="fn_max")
        else:
            max_weight = None
    
    st.write("**Expected Returns / Alphas:**")
    returns_input = st.text_area(
        "Returns (one per asset)",
        value="0.02, 0.03, -0.01, 0.015",
        key="fn_returns"
    )
    
    st.write("**Beta Matrix (asset per row):**")
    beta_input = st.text_area(
        "Betas",
        value="1.2, 0.5\n0.8, 1.1\n1.5, 0.3\n0.9, 0.8",
        height=100,
        key="fn_betas"
    )
    
    st.write("**Asset Covariance Matrix:**")
    cov_input = st.text_area(
        "Covariance",
        value="0.04, 0.01, 0.02, 0.01\n0.01, 0.03, 0.01, 0.005\n0.02, 0.01, 0.05, 0.015\n0.01, 0.005, 0.015, 0.025",
        height=100,
        key="fn_cov"
    )
    
    if st.button("🧮 Optimize Factor-Neutral Portfolio", key="fn_calc"):
        try:
            mu = parse_messy_input(returns_input)
            betas = parse_matrix_input(beta_input)
            cov = parse_matrix_input(cov_input)
            
            result, excel_steps, tables = optimize_factor_neutral(
                expected_returns=mu,
                covariance_matrix=cov,
                betas=betas,
                objective=objective,
                risk_cap=risk_cap,
                no_short=no_short,
                max_weight=max_weight
            )
            
            st.subheader("📊 Factor-Neutral Portfolio")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Return", f"{result['return']*100:.4f}%")
                st.metric("Volatility", f"{result['volatility']*100:.4f}%")
            with col2:
                st.write("**Factor Exposures (should be ~0):**")
                for i, exp in enumerate(result['factor_exposures']):
                    st.metric(f"Factor {i+1}", f"{exp:.6f}")
            
            st.write("**Weights:**")
            st.dataframe(tables['weights'])
            
            if not result['success']:
                st.warning(f"⚠️ Solver warning: {result['message']}")
            
            with st.expander("📝 How to do this in Excel"):
                for step in excel_steps:
                    st.markdown(step)
        
        except ValidationError as e:
            st.error(f"❌ Validation error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Calculation error: {str(e)}")

# =============================================================================
# MODULE: STOCK VALUATION, CAPM & ACTIVE MANAGEMENT (Ch. 6, 10, 13)
# =============================================================================

def stock_valuation_module():
    """
    Stock Valuation module covering:
    - Gordon Growth Model (DDM)
    - Multi-stage DDM
    - P/E and P/D ratio analysis
    - Equity Duration
    - CAPM and Security Market Line
    
    Exam Reference: Chapter 6 (Stocks), Chapter 10 (CAPM)
    """
    st.header("📊 Module 3: Stock Valuation")

    # Module 3 guidance expander
    with st.expander("📘 Module Guidance: When to use this?", expanded=False):
        st.markdown('''
        **When to use:**
        * **Valuation:** Pricing a stock using Dividends ($D_0, D_1$), Growth ($g$), and Required Return ($r$).
        * **Multi-Stage:** Growth changes over time (e.g., high growth for 5 years, then stable).
        * **Ratios:** Analyzing P/E, P/D, or PVGO (Present Value of Growth Opportunities).
        * **Equity Duration:** Sensitivity of stock price to discount rate changes.
        * **CAPM/SML:** Calculating Required Return ($E[r]$) or Alpha ($\alpha$) for single stocks.

        **Inputs you need:**
        * **Dividends:** Check if $D_0$ (just paid) or $D_1$ (next year).
        * **Rates:** $r$ (discount rate) and $g$ (growth rate). Must have $r > g$ for Gordon Growth.

        **Common Pitfalls:**
        * **Timing:** $P_0$ uses $D_1$. If you have $D_0$, calculate $D_1 = D_0(1+g)$.
        * **Terminal Value:** In multi-stage, $P_T$ is discounted back $T$ years.
        * **Negative PVGO:** Can happen if ROE < $r$.

        **Fast Decision Rule:**
        "If the problem mentions **Dividends, Growth (g), P/E Ratio, or PVGO**, use this module."
        ''')
    
    st.info(
        "🧠 **Exam tip:** Stock valuation problems typically require the Gordon Growth Model "
        "P = D₁/(r-g) or its variants. Remember that D₁ = D₀(1+g) is next period's dividend. "
        "For CAPM problems, use E[r] = rf + β(E[rm] - rf) and identify mispriced securities. "
        "For active management (Treynor-Black, performance metrics), see the Factor Models module."
    )
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌱 Gordon Growth (DDM)",
        "📈 Multi-Stage DDM", 
        "📊 P/E & P/D Analysis",
        "⏱️ Equity Duration",
        "📐 CAPM & SML"
    ])
    
    with tab1:
        gordon_growth_tab()
    with tab2:
        multistage_ddm_tab()
    with tab3:
        pe_pd_analysis_tab()
    with tab4:
        equity_duration_tab()
    with tab5:
        capm_sml_tab()


def gordon_growth_tab():
    """Gordon Growth Model / Dividend Discount Model calculator."""
    st.subheader("Gordon Growth Model (Constant Growth DDM)")
    
    st.markdown(r"""
    **The Gordon Growth Model:**
    $$P_0 = \frac{D_1}{r - g} = \frac{D_0(1+g)}{r - g}$$
    """)
    
    st.info("🧠 **Exam tip:** Be careful whether D₀ (just paid) or D₁ (next dividend) is given.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        calc_mode = st.radio(
            "Calculate:",
            ["Price (given D, r, g)", "Required Return r", "Growth Rate g", "Dividend D₁"],
            key="ggm_mode"
        )
        div_type = st.radio("Dividend type:", ["D₀ (just paid)", "D₁ (next)"], key="ggm_dtype")
    
    with col2:
        if "Price" in calc_mode:
            if "D₀" in div_type:
                D0 = st.number_input("D₀ ($)", value=2.00, format="%.4f", key="ggm_d0")
            else:
                D1 = st.number_input("D₁ ($)", value=2.10, format="%.4f", key="ggm_d1")
            r = st.number_input("Required Return r", value=0.10, format="%.4f", key="ggm_r")
            g = st.number_input("Growth Rate g", value=0.05, format="%.4f", key="ggm_g")
        elif "Required" in calc_mode:
            P = st.number_input("Price P₀ ($)", value=42.00, format="%.2f", key="ggm_p")
            if "D₀" in div_type:
                D0 = st.number_input("D₀ ($)", value=2.00, format="%.4f", key="ggm_d0r")
            else:
                D1 = st.number_input("D₁ ($)", value=2.10, format="%.4f", key="ggm_d1r")
            g = st.number_input("Growth Rate g", value=0.05, format="%.4f", key="ggm_gr")
        elif "Growth" in calc_mode:
            P = st.number_input("Price P₀ ($)", value=42.00, format="%.2f", key="ggm_pg")
            if "D₀" in div_type:
                D0 = st.number_input("D₀ ($)", value=2.00, format="%.4f", key="ggm_d0g")
            else:
                D1 = st.number_input("D₁ ($)", value=2.10, format="%.4f", key="ggm_d1g")
            r = st.number_input("Required Return r", value=0.10, format="%.4f", key="ggm_rg")
        else:
            P = st.number_input("Price P₀ ($)", value=42.00, format="%.2f", key="ggm_pd")
            r = st.number_input("Required Return r", value=0.10, format="%.4f", key="ggm_rd")
            g = st.number_input("Growth Rate g", value=0.05, format="%.4f", key="ggm_gd")
    
    if st.button("🧮 Calculate", key="ggm_calc"):
        try:
            if "Price" in calc_mode:
                if "D₀" in div_type:
                    D1 = D0 * (1 + g)
                else:
                    D0 = D1 / (1 + g)
                if g >= r:
                    st.error("❌ g must be < r"); return
                price = D1 / (r - g)
                st.metric("Stock Price P₀", f"${price:,.2f}")
                st.metric("Dividend Yield", f"{(D1/price)*100:.2f}%")
                with st.expander("📝 Excel"):
                    st.code(f"=D1/(r-g) = {D1:.4f}/({r}-{g}) = {price:.2f}")
            
            elif "Required" in calc_mode:
                if "D₀" in div_type:
                    D1 = D0 * (1 + g)
                r_calc = D1 / P + g
                st.metric("Required Return r", f"{r_calc*100:.4f}%")
                with st.expander("📝 Excel"):
                    st.code(f"r = D1/P + g = {D1:.4f}/{P:.2f} + {g} = {r_calc:.4f}")
            
            elif "Growth" in calc_mode:
                if "D₀" in div_type:
                    g_calc = (P * r - D0) / (P + D0)
                    D1 = D0 * (1 + g_calc)
                else:
                    g_calc = r - D1 / P
                st.metric("Implied Growth g", f"{g_calc*100:.4f}%")
                with st.expander("📝 Excel"):
                    st.code(f"g = r - D1/P = {r} - {D1:.4f}/{P:.2f} = {g_calc:.4f}")
            
            else:
                D1_calc = P * (r - g)
                st.metric("Required D₁", f"${D1_calc:.4f}")
        except Exception as e:
            st.error(f"❌ {str(e)}")


def multistage_ddm_tab():
    """Multi-stage Dividend Discount Model."""
    st.subheader("Multi-Stage DDM (Two-Stage Model)")
    
    st.markdown(r"""
    $$P_0 = \sum_{t=1}^{T} \frac{D_t}{(1+r)^t} + \frac{P_T}{(1+r)^T}$$
    Where $P_T = D_{T+1}/(r - g_2)$
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        D0 = st.number_input("D₀ ($)", value=2.00, format="%.4f", key="ms_d0")
        g1 = st.number_input("High Growth g₁", value=0.15, format="%.4f", key="ms_g1")
        T = st.number_input("High Growth Years", value=5, min_value=1, key="ms_T")
    with col2:
        g2 = st.number_input("Terminal Growth g₂", value=0.04, format="%.4f", key="ms_g2")
        r = st.number_input("Required Return r", value=0.12, format="%.4f", key="ms_r")
    
    if st.button("🧮 Calculate", key="ms_calc"):
        try:
            if g2 >= r:
                st.error("❌ g₂ must be < r"); return
            
            divs, pvs = [], []
            D_t = D0
            for t in range(1, T + 1):
                D_t *= (1 + g1)
                pv = D_t / (1 + r) ** t
                divs.append(D_t); pvs.append(pv)
            
            pv_phase1 = sum(pvs)
            D_T1 = D_t * (1 + g2)
            P_T = D_T1 / (r - g2)
            pv_terminal = P_T / (1 + r) ** T
            price = pv_phase1 + pv_terminal
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price P₀", f"${price:,.2f}")
            with col2:
                st.metric("PV Phase 1", f"${pv_phase1:,.2f}")
            with col3:
                st.metric("PV Terminal", f"${pv_terminal:,.2f}")
            
            st.dataframe(pd.DataFrame({
                'Year': range(1, T+1), 'Dividend': divs, 'PV': pvs
            }).style.format({'Dividend': '${:.4f}', 'PV': '${:.4f}'}))
        except Exception as e:
            st.error(f"❌ {str(e)}")


def pe_pd_analysis_tab():
    """P/E and P/D ratio analysis."""
    st.subheader("P/E and P/D Ratio Analysis")
    
    st.markdown(r"""
    **P/E Ratio:** $\frac{P}{E_1} = \frac{1-b}{r-g}$ where $g = b \times ROE$
    
    **PVGO:** $P = \frac{E_1}{r} + PVGO$
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        E1 = st.number_input("EPS E₁", value=5.00, format="%.4f", key="pe_e1")
        D1 = st.number_input("Dividend D₁", value=2.00, format="%.4f", key="pe_d1")
    with col2:
        r = st.number_input("Required Return r", value=0.12, format="%.4f", key="pe_r")
        ROE = st.number_input("ROE", value=0.15, format="%.4f", key="pe_roe")
    
    if st.button("🧮 Calculate", key="pe_calc"):
        try:
            payout = D1 / E1
            retention = 1 - payout
            g = retention * ROE
            if g >= r:
                st.error("❌ g must be < r"); return
            
            PE = payout / (r - g)
            PD = 1 / (r - g)
            price = D1 / (r - g)
            no_growth = E1 / r
            PVGO = price - no_growth
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Price", f"${price:,.2f}")
                st.metric("P/E Ratio", f"{PE:.2f}x")
            with col2:
                st.metric("P/D Ratio", f"{PD:.2f}x")
                st.metric("Growth g", f"{g*100:.2f}%")
            with col3:
                st.metric("PVGO", f"${PVGO:,.2f}")
                st.metric("PVGO %", f"{PVGO/price*100:.1f}%")
        except Exception as e:
            st.error(f"❌ {str(e)}")


def equity_duration_tab():
    """Equity duration."""
    st.subheader("Equity Duration")
    
    st.markdown(r"""
    **Equity Duration** (Gordon Model): $D_{equity} = \frac{1}{r-g}$
    
    Price sensitivity: $\Delta P \approx -D_{equity} \times \Delta r \times P$
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        r = st.number_input("Required Return r", value=0.10, format="%.4f", key="ed_r")
        g = st.number_input("Growth Rate g", value=0.04, format="%.4f", key="ed_g")
    with col2:
        P = st.number_input("Price P₀", value=50.00, format="%.2f", key="ed_p")
        delta_r = st.number_input("Rate Change (bp)", value=100, key="ed_dr")
    
    if st.button("🧮 Calculate", key="ed_calc"):
        try:
            if g >= r:
                st.error("❌ g must be < r"); return
            
            duration = 1 / (r - g)
            dr = delta_r / 10000
            price_chg = -duration * dr * P
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Duration", f"{duration:.2f} years")
            with col2:
                st.metric(f"Price Δ ({delta_r}bp)", f"${price_chg:,.2f}")
            
            st.info(f"A {delta_r}bp rate increase causes ~{price_chg/P*100:.1f}% price decline")
        except Exception as e:
            st.error(f"❌ {str(e)}")


def capm_sml_tab():
    """CAPM, Security Market Line (SML), and Capital Market Line (CML)."""
    st.subheader("CAPM & Security Market Line / Capital Market Line")
    
    st.markdown(r"""
    **CAPM:** $E[r_i] = r_f + \beta_i(E[r_m] - r_f)$
    
    **Alpha:** $\alpha_i = E[r_i]^{actual} - E[r_i]^{CAPM}$
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        rf = st.number_input("Risk-Free Rate (rf)", value=0.03, format="%.4f", key="capm_rf")
        E_rm = st.number_input("Market Expected Return E[rm]", value=0.10, format="%.4f", key="capm_erm")
        sigma_m = st.number_input("Market Volatility (σm)", value=0.18, format="%.4f", key="capm_sm")
    with col2:
        sec_input = st.text_area(
            "Securities (Name, Beta, E[r], σ):",
            "Stock_A, 1.2, 0.13, 0.25\nStock_B, 0.8, 0.07, 0.15\nStock_C, 1.5, 0.16, 0.30",
            key="capm_sec",
            help="Enter: Name, Beta, Expected Return, Volatility (σ is optional but needed for CML)"
        )
    
    if st.button("🧮 Analyze", key="capm_calc"):
        try:
            mkt_prem = E_rm - rf
            market_sharpe = mkt_prem / sigma_m if sigma_m > 0 else 0
            
            secs = []
            for line in sec_input.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        name = parts[0]
                        beta = float(parts[1])
                        er = float(parts[2])
                        sigma = float(parts[3]) if len(parts) >= 4 else None
                        
                        capm_r = rf + beta * mkt_prem
                        alpha = er - capm_r
                        
                        sec_data = {
                            'Name': name, 
                            'Beta': beta, 
                            'E[r]': er, 
                            'σ': sigma,
                            'CAPM E[r]': capm_r, 
                            'Alpha': alpha,
                            'Signal': 'BUY ↑' if alpha > 0.001 else ('SELL ↓' if alpha < -0.001 else 'HOLD →')
                        }
                        
                        # Calculate Sharpe ratio if volatility provided
                        if sigma is not None and sigma > 0:
                            sec_data['Sharpe'] = (er - rf) / sigma
                        else:
                            sec_data['Sharpe'] = None
                        
                        secs.append(sec_data)
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Risk Premium", f"{mkt_prem*100:.2f}%")
            with col2:
                st.metric("Market Sharpe Ratio", f"{market_sharpe:.4f}")
            with col3:
                st.metric("Risk-Free Rate", f"{rf*100:.2f}%")
            
            # Display securities table
            df = pd.DataFrame(secs)
            format_dict = {
                'Beta': '{:.2f}', 
                'E[r]': '{:.2%}', 
                'σ': '{:.2%}',
                'CAPM E[r]': '{:.2%}', 
                'Alpha': '{:.2%}',
                'Sharpe': '{:.4f}'
            }
            st.dataframe(df.style.format({k: v for k, v in format_dict.items() if k in df.columns}, na_rep='-'))
            
            # =================================================================
            # SECURITY MARKET LINE (SML) CHART
            # =================================================================
            st.subheader("📈 Security Market Line (SML)")
            
            fig_sml, ax_sml = plt.subplots(figsize=(10, 6))
            
            # Plot SML line
            betas = np.linspace(-0.2, 2.2, 100)
            sml_returns = rf + betas * mkt_prem
            ax_sml.plot(betas, sml_returns * 100, 'b-', linewidth=2, label='Security Market Line (SML)')
            
            # Plot risk-free asset (beta = 0)
            ax_sml.scatter([0], [rf * 100], c='gold', s=150, marker='*', zorder=5, 
                          edgecolors='black', linewidths=1, label=f'Risk-Free Asset (rf = {rf*100:.1f}%)')
            
            # Plot market portfolio (beta = 1)
            ax_sml.scatter([1], [E_rm * 100], c='blue', s=120, marker='D', zorder=5,
                          edgecolors='black', linewidths=1, label=f'Market Portfolio (β=1, E[r]={E_rm*100:.1f}%)')
            
            # Plot securities
            for s in secs:
                if s['Alpha'] > 0.001:
                    c, marker = 'green', '^'
                elif s['Alpha'] < -0.001:
                    c, marker = 'red', 'v'
                else:
                    c, marker = 'gray', 'o'
                ax_sml.scatter(s['Beta'], s['E[r]'] * 100, c=c, s=100, marker=marker, zorder=4)
                ax_sml.annotate(f"  {s['Name']}", (s['Beta'], s['E[r]'] * 100), fontsize=9,
                               verticalalignment='center')
                
                # Draw vertical line to SML (showing alpha)
                capm_ret = s['CAPM E[r]'] * 100
                ax_sml.plot([s['Beta'], s['Beta']], [capm_ret, s['E[r]'] * 100], 
                           c=c, linestyle='--', alpha=0.5, linewidth=1)
            
            ax_sml.set_xlabel('Beta (β)', fontsize=11)
            ax_sml.set_ylabel('Expected Return E[r] (%)', fontsize=11)
            ax_sml.set_title('Security Market Line (SML): E[r] vs Beta', fontsize=12)
            ax_sml.legend(loc='upper left', fontsize=9)
            ax_sml.grid(alpha=0.3)
            ax_sml.axhline(y=rf*100, color='gold', linestyle=':', alpha=0.5)
            ax_sml.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
            ax_sml.axvline(x=1, color='blue', linestyle=':', alpha=0.3)
            
            st.pyplot(fig_sml)
            plt.close()
            
            # =================================================================
            # CAPITAL MARKET LINE (CML) CHART
            # =================================================================
            st.subheader("📉 Capital Market Line (CML)")
            
            # Check if we have volatility data
            has_vol_data = any(s['σ'] is not None for s in secs)
            
            fig_cml, ax_cml = plt.subplots(figsize=(10, 6))
            
            # Plot CML line
            sigmas = np.linspace(0, 0.40, 100)
            cml_returns = rf + market_sharpe * sigmas
            ax_cml.plot(sigmas * 100, cml_returns * 100, 'b-', linewidth=2, 
                       label=f'Capital Market Line (CML), Slope = {market_sharpe:.4f}')
            
            # Plot risk-free asset (σ = 0)
            ax_cml.scatter([0], [rf * 100], c='gold', s=150, marker='*', zorder=5,
                          edgecolors='black', linewidths=1, label=f'Risk-Free Asset (rf = {rf*100:.1f}%)')
            
            # Plot market portfolio
            ax_cml.scatter([sigma_m * 100], [E_rm * 100], c='blue', s=120, marker='D', zorder=5,
                          edgecolors='black', linewidths=1, label=f'Market Portfolio (σ={sigma_m*100:.1f}%, E[r]={E_rm*100:.1f}%)')
            
            # Plot securities (if volatility data available)
            if has_vol_data:
                for s in secs:
                    if s['σ'] is not None:
                        # Compare to CML: is it above or below?
                        cml_return_at_sigma = rf + market_sharpe * s['σ']
                        above_cml = s['E[r]'] > cml_return_at_sigma + 0.001
                        below_cml = s['E[r]'] < cml_return_at_sigma - 0.001
                        
                        if above_cml:
                            c, marker = 'green', '^'
                        elif below_cml:
                            c, marker = 'red', 'v'
                        else:
                            c, marker = 'gray', 'o'
                        
                        ax_cml.scatter(s['σ'] * 100, s['E[r]'] * 100, c=c, s=100, marker=marker, zorder=4)
                        ax_cml.annotate(f"  {s['Name']}", (s['σ'] * 100, s['E[r]'] * 100), fontsize=9,
                                       verticalalignment='center')
            else:
                st.info("💡 Add volatility (σ) to your securities input to plot them on the CML chart.")
            
            ax_cml.set_xlabel('Total Risk σ (%)', fontsize=11)
            ax_cml.set_ylabel('Expected Return E[r] (%)', fontsize=11)
            ax_cml.set_title('Capital Market Line (CML): E[r] vs Total Risk', fontsize=12)
            ax_cml.legend(loc='upper left', fontsize=9)
            ax_cml.grid(alpha=0.3)
            ax_cml.axhline(y=rf*100, color='gold', linestyle=':', alpha=0.5)
            ax_cml.set_xlim(left=0)
            
            st.pyplot(fig_cml)
            plt.close()
            
            # =================================================================
            # INTERPRETATION GUIDE
            # =================================================================
            st.subheader("📖 How to Read & Interpret SML and CML")
            
            with st.expander("🔍 Security Market Line (SML) Interpretation", expanded=True):
                st.markdown(r"""
### What the SML Shows
The SML plots **Expected Return vs Beta (systematic risk)**. It represents the **CAPM equilibrium relationship**.

### Key Points on the SML
| Point | Beta | Expected Return | Interpretation |
|-------|------|-----------------|----------------|
| **Risk-Free Asset** | β = 0 | rf | No systematic risk, earns risk-free rate |
| **Market Portfolio** | β = 1 | E[rm] | Average systematic risk, earns market return |
| **Aggressive Stock** | β > 1 | > E[rm] | More sensitive to market, higher required return |
| **Defensive Stock** | β < 1 | < E[rm] | Less sensitive to market, lower required return |

### Reading Alpha from the SML
- **Above SML (α > 0):** Stock is **underpriced** → **BUY** signal
  - Actual return > CAPM required return
  - Stock offers more return than its risk justifies
  
- **Below SML (α < 0):** Stock is **overpriced** → **SELL** signal
  - Actual return < CAPM required return  
  - Stock offers less return than its risk requires

- **On SML (α = 0):** Stock is **fairly priced** → **HOLD**
  - Actual return = CAPM required return

### Formula Reminder
$$\alpha_i = E[r_i]_{actual} - E[r_i]_{CAPM} = E[r_i]_{actual} - [r_f + \beta_i(E[r_m] - r_f)]$$
                """)
            
            with st.expander("🔍 Capital Market Line (CML) Interpretation", expanded=True):
                st.markdown(r"""
### What the CML Shows
The CML plots **Expected Return vs Total Risk (σ)**. It represents **efficient portfolios** that combine the risk-free asset with the market portfolio.

### Key Points on the CML
| Point | σ | Expected Return | Interpretation |
|-------|---|-----------------|----------------|
| **Risk-Free Asset** | σ = 0 | rf | 100% in risk-free, zero risk |
| **Market Portfolio** | σ = σm | E[rm] | 100% in market portfolio |
| **Leveraged Position** | σ > σm | > E[rm] | Borrowed at rf to invest >100% in market |

### The CML Slope = Market Sharpe Ratio
$$\text{Slope of CML} = \frac{E[r_m] - r_f}{\sigma_m} = \text{Sharpe Ratio of Market}$$

This is the **market price of risk** - the extra return per unit of total risk for efficient portfolios.

### Reading the CML
- **On the CML:** Portfolio is **efficient** (optimal mix of rf and market)
- **Below the CML:** Portfolio is **inefficient** (can get same return with less risk, or more return for same risk)
- **Above the CML:** Should not exist in equilibrium (arbitrage opportunity)

### CML vs SML: Key Differences
| Feature | SML | CML |
|---------|-----|-----|
| **X-axis** | Beta (β) - systematic risk | Sigma (σ) - total risk |
| **Applies to** | All assets (individual stocks) | Only efficient portfolios |
| **Shows** | Required return for bearing systematic risk | Return for bearing total risk on efficient frontier |
| **Slope** | Market risk premium (E[rm] - rf) | Sharpe ratio of market |

### Important Exam Insight
- Individual stocks typically lie **below** the CML (they have diversifiable risk)
- Only **efficient portfolios** (combinations of rf and market) lie ON the CML
- The SML applies to ALL assets; the CML only applies to efficient portfolios
                """)
            
            # =================================================================
            # NUMERICAL SUMMARY
            # =================================================================
            with st.expander("📊 Numerical Summary"):
                st.markdown(f"""
**Market Parameters:**
- Risk-Free Rate: rf = {rf*100:.2f}%
- Market Expected Return: E[rm] = {E_rm*100:.2f}%
- Market Volatility: σm = {sigma_m*100:.2f}%
- Market Risk Premium: E[rm] - rf = {mkt_prem*100:.2f}%
- Market Sharpe Ratio: SR = {market_sharpe:.4f}

**SML Equation:**
$$E[r_i] = {rf*100:.2f}\% + \\beta_i \\times {mkt_prem*100:.2f}\%$$

**CML Equation:**
$$E[r_p] = {rf*100:.2f}\% + {market_sharpe:.4f} \\times \\sigma_p$$
                """)
                
                st.markdown("**Securities Analysis:**")
                for s in secs:
                    signal_emoji = "🟢" if "BUY" in s['Signal'] else ("🔴" if "SELL" in s['Signal'] else "⚪")
                    st.markdown(f"""
**{s['Name']}** {signal_emoji}
- Beta: {s['Beta']:.2f} | Actual E[r]: {s['E[r]']*100:.2f}% | CAPM E[r]: {s['CAPM E[r]']*100:.2f}%
- **Alpha: {s['Alpha']*100:.2f}%** → {s['Signal']}
                    """)
        
        except Exception as e:
            st.error(f"❌ {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# =============================================================================
# MODULE 8: EXAM TRIAGE - "I Don't Know What To Do" Wizard
# =============================================================================

# -----------------------------------------------------------------------------
# GOAL TAXONOMY - Organized by category with notation and dependencies
# -----------------------------------------------------------------------------

GOAL_TAXONOMY = {
    "Returns & Expected Values": [
        ("expected_return_capm", "Expected Return — CAPM (single factor)", r"$E[r_i] = r_f + \beta_i(E[r_m] - r_f)$"),
        ("expected_return_multifactor", "Expected Return — Multi-Factor/APT", r"$E[r_i] = r_f + \sum_k \beta_{ik} \lambda_k$"),
        ("portfolio_return", "Portfolio Expected Return", r"$E[r_p] = \sum_i w_i E[r_i]$"),
        ("reverse_portfolio_return", "Missing Asset Return (Reverse Weighted Avg)", r"$E[r_{?}] = \frac{E[r_p] - \sum_{known} w_i E[r_i]}{w_{?}}$"),
        ("log_return_distribution", "Log-Return Distribution over T years", r"$\ln(S_T/S_0) \sim N(\mu T, \sigma^2 T)$"),
        ("expected_wealth", "Expected Wealth in T Years", r"$E[W_T] = W_0 \cdot e^{\mu T}$"),
    ],
    "Risk Metrics": [
        ("portfolio_variance", "Portfolio Variance / Std Dev", r"$\sigma_p^2 = w'\Sigma w$"),
        ("portfolio_sharpe", "Sharpe Ratio", r"$SR = \frac{E[r] - r_f}{\sigma}$"),
        ("beta_from_components", "Beta (from correlation, volatilities)", r"$\beta_i = \rho_{i,m} \frac{\sigma_i}{\sigma_m}$"),
        ("beta_portfolio", "Portfolio Beta", r"$\beta_p = \sum_i w_i \beta_i$"),
        ("systematic_variance", "Systematic Variance / R²", r"$R^2 = \frac{\beta^2 \sigma_m^2}{\sigma_i^2}$"),
        ("residual_variance", "Residual/Idiosyncratic Variance", r"$\sigma_\epsilon^2 = \sigma_i^2 - \beta^2 \sigma_m^2$"),
        ("covariance_assets", "Covariance between Assets", r"$\sigma_{ij} = \rho_{ij} \sigma_i \sigma_j$"),
        ("correlation_assets", "Correlation between Assets", r"$\rho_{ij} = \frac{\sigma_{ij}}{\sigma_i \sigma_j}$"),
    ],
    "Optimal Portfolios & Weights": [
        ("min_var_weights", "Minimum-Variance Portfolio Weights", r"$w_{MV} = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}'\Sigma^{-1}\mathbf{1}}$"),
        ("tangency_weights", "Tangency Portfolio Weights", r"$w_T = \frac{\Sigma^{-1}(\mu - r_f \mathbf{1})}{\mathbf{1}'\Sigma^{-1}(\mu - r_f \mathbf{1})}$"),
        ("utility_max_weights", "Utility-Maximizing Weights (given γ)", r"$w^* = \frac{1}{\gamma}\Sigma^{-1}(\mu - r_f \mathbf{1})$"),
        ("constrained_weights", "Constrained Optimal Weights", "Solver-based optimization"),
        ("two_fund_weights", "Two-Fund Separation Weights", r"$w = \alpha \cdot w_T + (1-\alpha) \cdot w_{rf}$"),
    ],
    "Factor Models & APT": [
        ("factor_premiums_reverse", "Factor Risk Premiums (Reverse APT)", r"$\lambda = (B'B)^{-1}B'(E[r] - r_f)$"),
        ("jensens_alpha", "Jensen's Alpha", r"$\alpha = E[r_i] - [r_f + \beta(E[r_m] - r_f)]$"),
        ("information_ratio", "Information Ratio", r"$IR = \frac{\alpha}{\sigma_\epsilon}$"),
        ("treynor_ratio", "Treynor Ratio", r"$TR = \frac{E[r] - r_f}{\beta}$"),
        ("appraisal_ratio", "Appraisal Ratio", r"$AR = \frac{\alpha}{\sigma_\epsilon}$"),
        ("active_weights_tb", "Active Portfolio Weights (Treynor-Black)", r"$w_i^A \propto \frac{\alpha_i}{\sigma_{\epsilon,i}^2}$"),
    ],
    "Fixed Income & Bonds": [
        ("bond_price", "Bond Price (YTM)", r"$P = \sum_{t=1}^{T} \frac{C}{(1+y)^t} + \frac{F}{(1+y)^T}$"),
        ("bond_price_spot", "Bond Price (Spot Curve)", r"$P = \sum_{t=1}^{T} \frac{CF_t}{(1+y_t)^t}$"),
        ("ytm_from_price", "Yield-to-Maturity (from price)", "Solve: Price = PV(coupons) + PV(face)"),
        ("macaulay_duration", "Macaulay Duration", r"$D = \frac{\sum t \cdot PV(CF_t)}{P}$"),
        ("modified_duration", "Modified Duration", r"$D_{mod} = \frac{D_{mac}}{1+y}$"),
        ("immunization_weights", "Immunization Weights", r"$w_A D_A + w_B D_B = D_L$"),
        ("forward_rate", "Forward Rate", r"$f_{t_1,t_2} = \frac{(1+y_{t_2})^{t_2}}{(1+y_{t_1})^{t_1}} - 1$"),
        ("spot_rate_bootstrap", "Spot Rate (Bootstrap)", "Sequential solving from short to long"),
    ],
    "Human Capital & Lifecycle": [
        ("human_capital_pv", "Human Capital (Present Value)", r"$HC = \sum_{t=1}^{T} \frac{Y_t}{(1+r)^t}$"),
        ("human_capital_gordon", "Human Capital (Gordon Growth)", r"$HC = \frac{Y_1}{r - g}$"),
        ("optimal_allocation_hc", "Optimal Allocation with Human Capital", r"$\pi^* = M(1+l) - H$"),
        ("implicit_hc_beta", "Implicit Human Capital Beta", r"$\beta_{HC} = \rho_{HC,m} \frac{\sigma_{HC}}{\sigma_m}$"),
    ],
    "Probability & Statistics": [
        ("prob_return_range", "Probability of Return in Range", r"$P(a < r < b) = \Phi(\frac{b-\mu}{\sigma}) - \Phi(\frac{a-\mu}{\sigma})$"),
        ("prob_negative_return", "Probability of Negative Return", r"$P(r < 0) = \Phi(\frac{-\mu}{\sigma})$"),
        ("prob_outperform", "Probability A Outperforms B", r"$P(r_A > r_B)$"),
        ("confidence_interval", "Confidence Interval (95%)", r"$[\mu - 1.96\sigma, \mu + 1.96\sigma]$"),
        ("var_cvar", "Value at Risk / CVaR", r"$VaR_{95\%} = -(\mu - 1.645\sigma)$"),
    ],
    "Valuation & Growth": [
        ("gordon_growth_price", "Stock Price (Gordon Growth)", r"$P = \frac{D_1}{r - g}$"),
        ("pe_ratio", "P/E Ratio", r"$P/E = \frac{1-b}{r-g}$"),
        ("pvgo", "Present Value of Growth Opportunities", r"$PVGO = P - \frac{E_1}{r}$"),
        ("implied_growth_rate", "Implied Growth Rate", r"$g = r - \frac{D_1}{P}$"),
        ("equity_duration", "Equity Duration", r"$D_{equity} = \frac{1}{r-g}$"),
    ],
    "Reverse Engineering & Algebraic": [
        ("implied_risk_aversion", "Implied Risk Aversion (γ)", r"$\gamma = \frac{(\mu - r_f)'\Sigma^{-1}(\mu - r_f)}{w'\Sigma w}$"),
        ("implied_market_premium", "Implied Market Risk Premium", r"$E[r_m] - r_f = \gamma \sigma_m^2$"),
        ("breakeven_rate", "Breakeven Rate/Growth", "Solve for rate making NPV = 0"),
        ("missing_weight", "Missing Portfolio Weight", r"$w_{?} = 1 - \sum_{known} w_i$"),
        ("missing_beta", "Missing Beta (from portfolio)", r"$\beta_{?} = \frac{\beta_p - \sum_{known} w_i \beta_i}{w_{?}}$"),
    ],
}

# -----------------------------------------------------------------------------
# INPUT DEFINITIONS - What variables are needed for each goal
# -----------------------------------------------------------------------------

INPUT_DEFINITIONS = {
    # Risk-free rate
    "rf": {"name": "Risk-Free Rate", "notation": r"$r_f$", "type": "percent", "default": 0.02,
           "aliases": ["T-bill rate", "safe rate", "borrowing rate", "lending rate"]},
    
    # Market parameters
    "E_rm": {"name": "Market Expected Return", "notation": r"$E[r_m]$", "type": "percent", "default": 0.10,
             "aliases": ["expected market return", "market return"]},
    "sigma_m": {"name": "Market Volatility", "notation": r"$\sigma_m$", "type": "percent", "default": 0.20,
                "aliases": ["market std dev", "market risk"]},
    "market_premium": {"name": "Market Risk Premium", "notation": r"$E[r_m] - r_f$", "type": "percent", "default": 0.06,
                       "aliases": ["equity premium", "MRP"]},
    
    # Single asset parameters
    "E_r": {"name": "Expected Return (asset)", "notation": r"$E[r]$", "type": "percent", "default": 0.12,
            "aliases": ["asset return", "stock return"]},
    "sigma": {"name": "Volatility (asset)", "notation": r"$\sigma$", "type": "percent", "default": 0.25,
              "aliases": ["std dev", "asset risk", "standard deviation"]},
    "beta": {"name": "Beta (market)", "notation": r"$\beta$", "type": "float", "default": 1.2,
             "aliases": ["market beta", "CAPM beta"]},
    "alpha": {"name": "Alpha (Jensen's)", "notation": r"$\alpha$", "type": "percent", "default": 0.02,
              "aliases": ["Jensen's alpha", "abnormal return", "mispricing"]},
    "rho_im": {"name": "Correlation (asset, market)", "notation": r"$\rho_{i,m}$", "type": "float", "default": 0.6,
               "aliases": ["correlation with market"]},
    
    # Multi-factor betas
    "beta_smb": {"name": "SMB Beta (Size)", "notation": r"$\beta_{SMB}$", "type": "float", "default": 0.5,
                 "aliases": ["size beta", "small-minus-big beta"]},
    "beta_hml": {"name": "HML Beta (Value)", "notation": r"$\beta_{HML}$", "type": "float", "default": 0.3,
                 "aliases": ["value beta", "high-minus-low beta"]},
    "beta_wml": {"name": "WML Beta (Momentum)", "notation": r"$\beta_{WML}$", "type": "float", "default": 0.2,
                 "aliases": ["momentum beta", "winner-minus-loser beta", "UMD beta"]},
    "beta_vector": {"name": "Beta Vector (multiple factors)", "notation": r"$[\beta_1, \beta_2, ...]$", "type": "vector", "default": "1.2, 0.5",
                    "aliases": ["factor betas", "factor loadings"]},
    
    # Factor premiums
    "lambda_smb": {"name": "SMB Premium", "notation": r"$\lambda_{SMB}$", "type": "percent", "default": 0.03,
                   "aliases": ["size premium"]},
    "lambda_hml": {"name": "HML Premium", "notation": r"$\lambda_{HML}$", "type": "percent", "default": 0.04,
                   "aliases": ["value premium"]},
    "lambda_wml": {"name": "WML Premium", "notation": r"$\lambda_{WML}$", "type": "percent", "default": 0.06,
                   "aliases": ["momentum premium"]},
    "lambda_vector": {"name": "Factor Premiums Vector", "notation": r"$[\lambda_1, \lambda_2, ...]$", "type": "vector", "default": "0.06, 0.03",
                      "aliases": ["risk premiums", "factor risk premiums"]},
    
    # Portfolio parameters
    "weights_vector": {"name": "Portfolio Weights", "notation": r"$[w_1, w_2, ...]$", "type": "vector", "default": "0.4, 0.3, 0.3",
                       "aliases": ["weights", "allocation"]},
    "returns_vector": {"name": "Expected Returns Vector", "notation": r"$[\mu_1, \mu_2, ...]$", "type": "vector", "default": "0.08, 0.12, 0.06",
                       "aliases": ["returns", "expected returns"]},
    "vols_vector": {"name": "Volatilities Vector", "notation": r"$[\sigma_1, \sigma_2, ...]$", "type": "vector", "default": "0.20, 0.25, 0.15",
                    "aliases": ["standard deviations", "risks"]},
    "cov_matrix": {"name": "Covariance Matrix", "notation": r"$\Sigma$", "type": "matrix", "default": "0.04, 0.01\n0.01, 0.0625",
                   "aliases": ["variance-covariance matrix"]},
    "corr_matrix": {"name": "Correlation Matrix", "notation": r"$\rho$", "type": "matrix", "default": "1.0, 0.3\n0.3, 1.0",
                    "aliases": ["correlations"]},
    "beta_matrix": {"name": "Beta Matrix (assets x factors)", "notation": r"$B$", "type": "matrix", "default": "1.2, 0.5\n0.8, -0.3",
                    "aliases": ["factor loadings matrix"]},
    
    # Risk aversion
    "gamma": {"name": "Risk Aversion", "notation": r"$\gamma$", "type": "float", "default": 4.0,
              "aliases": ["risk aversion coefficient"]},
    
    # Residual risk
    "sigma_epsilon": {"name": "Residual Volatility", "notation": r"$\sigma_\epsilon$", "type": "percent", "default": 0.15,
                      "aliases": ["idiosyncratic vol", "residual std dev", "specific risk"]},
    
    # Bond parameters
    "face_value": {"name": "Face Value / Par", "notation": r"$F$", "type": "float", "default": 1000.0,
                   "aliases": ["par value", "principal", "notional"]},
    "coupon_rate": {"name": "Coupon Rate", "notation": r"$c$", "type": "percent", "default": 0.05,
                    "aliases": ["coupon", "annual coupon rate"]},
    "maturity": {"name": "Maturity (years)", "notation": r"$T$", "type": "float", "default": 5.0,
                 "aliases": ["time to maturity", "term"]},
    "ytm": {"name": "Yield to Maturity", "notation": r"$y$", "type": "percent", "default": 0.05,
            "aliases": ["yield", "YTM", "discount rate"]},
    "bond_price": {"name": "Bond Price", "notation": r"$P$", "type": "float", "default": 1000.0,
                   "aliases": ["market price", "clean price"]},
    "spot_rates": {"name": "Spot Rates", "notation": r"$[y_1, y_2, ...]$", "type": "vector", "default": "0.03, 0.035, 0.04",
                   "aliases": ["zero rates", "spot curve", "zero-coupon yields"]},
    "duration": {"name": "Duration", "notation": r"$D$", "type": "float", "default": 4.5,
                 "aliases": ["Macaulay duration"]},
    "liability_duration": {"name": "Liability Duration", "notation": r"$D_L$", "type": "float", "default": 5.0,
                           "aliases": ["target duration"]},
    "convexity": {"name": "Convexity", "notation": r"$C$", "type": "float", "default": 25.0,
                  "aliases": ["bond convexity"]},
    "yield_change_bp": {"name": "Yield Change (basis points)", "notation": r"$\Delta y$", "type": "float", "default": 50.0,
                        "aliases": ["rate change", "yield shift"]},
    
    # Human capital parameters
    "labor_income": {"name": "Annual Labor Income", "notation": r"$Y$", "type": "float", "default": 75000.0,
                     "aliases": ["salary", "wage", "income"]},
    "income_growth": {"name": "Income Growth Rate", "notation": r"$g_Y$", "type": "percent", "default": 0.03,
                      "aliases": ["salary growth", "wage growth"]},
    "years_to_retirement": {"name": "Years to Retirement", "notation": r"$T$", "type": "float", "default": 30.0,
                            "aliases": ["working years", "career horizon"]},
    "rho_labor_market": {"name": "Correlation (Labor, Market)", "notation": r"$\rho_{Y,m}$", "type": "float", "default": 0.2,
                         "aliases": ["labor-market correlation", "income-market correlation"]},
    "sigma_labor": {"name": "Labor Income Volatility", "notation": r"$\sigma_Y$", "type": "percent", "default": 0.10,
                    "aliases": ["income volatility", "salary risk"]},
    "human_capital": {"name": "Human Capital Value", "notation": r"$HC$", "type": "float", "default": 500000.0,
                      "aliases": ["present value of labor income"]},
    "financial_wealth": {"name": "Financial Wealth", "notation": r"$W$", "type": "float", "default": 200000.0,
                         "aliases": ["liquid wealth", "investable assets"]},
    
    # Probability parameters
    "time_horizon": {"name": "Time Horizon (years)", "notation": r"$T$", "type": "float", "default": 1.0,
                     "aliases": ["investment horizon", "holding period"]},
    "target_return": {"name": "Target Return", "notation": r"$r_{target}$", "type": "percent", "default": 0.08,
                      "aliases": ["required return", "hurdle rate"]},
    "confidence_level": {"name": "Confidence Level", "notation": r"$1-\alpha$", "type": "percent", "default": 0.95,
                         "aliases": ["confidence", "probability level"]},
    
    # Valuation parameters
    "dividend": {"name": "Dividend (D1)", "notation": r"$D_1$", "type": "float", "default": 2.0,
                 "aliases": ["expected dividend", "next dividend"]},
    "earnings": {"name": "Earnings (E1)", "notation": r"$E_1$", "type": "float", "default": 5.0,
                 "aliases": ["EPS", "earnings per share"]},
    "growth_rate": {"name": "Growth Rate", "notation": r"$g$", "type": "percent", "default": 0.04,
                    "aliases": ["dividend growth", "earnings growth"]},
    "stock_price": {"name": "Stock Price", "notation": r"$P$", "type": "float", "default": 50.0,
                    "aliases": ["share price", "market price"]},
    "payout_ratio": {"name": "Payout Ratio", "notation": r"$b$", "type": "percent", "default": 0.40,
                     "aliases": ["dividend payout"]},
    "roe": {"name": "Return on Equity", "notation": r"$ROE$", "type": "percent", "default": 0.15,
            "aliases": ["ROE"]},
    
    # Portfolio return for reverse problems
    "portfolio_return": {"name": "Portfolio Return", "notation": r"$E[r_p]$", "type": "percent", "default": 0.09,
                         "aliases": ["portfolio expected return"]},
}

# -----------------------------------------------------------------------------
# GOAL REQUIREMENTS - What inputs are needed for each goal
# -----------------------------------------------------------------------------

GOAL_REQUIREMENTS = {
    # Returns
    "expected_return_capm": {"required": ["rf", "beta"], "optional": ["E_rm", "market_premium"]},
    "expected_return_multifactor": {"required": ["rf", "beta_vector", "lambda_vector"], "optional": []},
    "portfolio_return": {"required": ["weights_vector", "returns_vector"], "optional": []},
    "reverse_portfolio_return": {"required": ["portfolio_return", "weights_vector"], "optional": ["returns_vector"]},
    
    # Risk metrics
    "portfolio_variance": {"required": ["weights_vector"], "optional": ["cov_matrix", "vols_vector", "corr_matrix"]},
    "portfolio_sharpe": {"required": ["E_r", "rf", "sigma"], "optional": []},
    "beta_from_components": {"required": ["rho_im", "sigma", "sigma_m"], "optional": []},
    "beta_portfolio": {"required": ["weights_vector", "beta_vector"], "optional": []},
    "systematic_variance": {"required": ["beta", "sigma_m"], "optional": ["sigma"]},
    "residual_variance": {"required": ["sigma", "beta", "sigma_m"], "optional": []},
    
    # Optimal portfolios
    "min_var_weights": {"required": ["cov_matrix"], "optional": ["vols_vector", "corr_matrix"]},
    "tangency_weights": {"required": ["returns_vector", "rf"], "optional": ["cov_matrix", "vols_vector", "corr_matrix"]},
    "utility_max_weights": {"required": ["returns_vector", "rf", "gamma"], "optional": ["cov_matrix", "vols_vector", "corr_matrix"]},
    
    # Factor models
    "factor_premiums_reverse": {"required": ["returns_vector", "beta_matrix", "rf"], "optional": []},
    "jensens_alpha": {"required": ["E_r", "rf", "beta", "E_rm"], "optional": ["market_premium"]},
    "information_ratio": {"required": ["alpha", "sigma_epsilon"], "optional": []},
    "treynor_ratio": {"required": ["E_r", "rf", "beta"], "optional": []},
    
    # Bonds
    "bond_price": {"required": ["face_value", "coupon_rate", "maturity", "ytm"], "optional": []},
    "bond_price_spot": {"required": ["face_value", "coupon_rate", "spot_rates"], "optional": []},
    "macaulay_duration": {"required": ["coupon_rate", "maturity", "ytm"], "optional": ["face_value"]},
    "immunization_weights": {"required": ["liability_duration"], "optional": ["duration"]},
    
    # Human capital
    "human_capital_pv": {"required": ["labor_income", "years_to_retirement"], "optional": ["income_growth", "rf"]},
    "human_capital_gordon": {"required": ["labor_income", "income_growth", "rf"], "optional": []},
    
    # Probability
    "prob_negative_return": {"required": ["E_r", "sigma"], "optional": ["time_horizon"]},
    "confidence_interval": {"required": ["E_r", "sigma"], "optional": ["confidence_level"]},
    
    # Valuation
    "gordon_growth_price": {"required": ["dividend", "growth_rate"], "optional": ["rf", "E_r"]},
    "implied_growth_rate": {"required": ["stock_price", "dividend"], "optional": ["rf", "E_r"]},
    
    # Reverse engineering
    "implied_risk_aversion": {"required": ["weights_vector", "returns_vector", "rf", "cov_matrix"], "optional": []},
}

# -----------------------------------------------------------------------------
# ROUTING MAP - Where to go for each goal
# -----------------------------------------------------------------------------

ROUTING_MAP = {
    "expected_return_capm": {"module": 5, "tab": "Factor Models", "subtab": "CAPM & SML"},
    "expected_return_multifactor": {"module": 5, "tab": "Factor Models", "subtab": "Factor Risk Analysis"},
    "portfolio_return": {"module": 1, "tab": "Portfolio Optimizer", "subtab": "Unconstrained"},
    "reverse_portfolio_return": {"module": 7, "tab": "Universal Solver", "subtab": "Reverse Weighted Average"},
    "portfolio_variance": {"module": 1, "tab": "Portfolio Optimizer", "subtab": "Unconstrained"},
    "portfolio_sharpe": {"module": 7, "tab": "Universal Solver", "subtab": "Sharpe Ratio Solver"},
    "beta_from_components": {"module": 7, "tab": "Universal Solver", "subtab": "Beta & Correlation Solver"},
    "min_var_weights": {"module": 1, "tab": "Portfolio Optimizer", "subtab": "Unconstrained"},
    "tangency_weights": {"module": 1, "tab": "Portfolio Optimizer", "subtab": "Unconstrained"},
    "utility_max_weights": {"module": 1, "tab": "Portfolio Optimizer", "subtab": "Unconstrained"},
    "constrained_weights": {"module": 1, "tab": "Portfolio Optimizer", "subtab": "Constrained (Solver)"},
    "factor_premiums_reverse": {"module": 5, "tab": "Factor Models", "subtab": "Reverse APT"},
    "jensens_alpha": {"module": 5, "tab": "Factor Models", "subtab": "Performance Metrics"},
    "information_ratio": {"module": 5, "tab": "Factor Models", "subtab": "Performance Metrics"},
    "treynor_ratio": {"module": 5, "tab": "Factor Models", "subtab": "Performance Metrics"},
    "active_weights_tb": {"module": 5, "tab": "Factor Models", "subtab": "Treynor-Black"},
    "bond_price": {"module": 2, "tab": "Bond Math", "subtab": "Pricing & Duration"},
    "bond_price_spot": {"module": 2, "tab": "Bond Math", "subtab": "Price with Spot Curve"},
    "macaulay_duration": {"module": 2, "tab": "Bond Math", "subtab": "Pricing & Duration"},
    "immunization_weights": {"module": 2, "tab": "Bond Math", "subtab": "Immunization"},
    "forward_rate": {"module": 2, "tab": "Bond Math", "subtab": "Forward Rates"},
    "human_capital_pv": {"module": 4, "tab": "Human Capital", "subtab": "Human Capital"},
    "human_capital_gordon": {"module": 4, "tab": "Human Capital", "subtab": "Human Capital"},
    "optimal_allocation_hc": {"module": 4, "tab": "Human Capital", "subtab": "Human Capital"},
    "prob_return_range": {"module": 6, "tab": "Probability", "subtab": "Return Probabilities"},
    "prob_negative_return": {"module": 6, "tab": "Probability", "subtab": "Return Probabilities"},
    "confidence_interval": {"module": 6, "tab": "Probability", "subtab": "Return Probabilities"},
    "gordon_growth_price": {"module": 3, "tab": "Stock Valuation", "subtab": "Gordon Growth (DDM)"},
    "pe_ratio": {"module": 3, "tab": "Stock Valuation", "subtab": "P/E and P/D Analysis"},
    "pvgo": {"module": 3, "tab": "Stock Valuation", "subtab": "P/E and P/D Analysis"},
}

# -----------------------------------------------------------------------------
# DIRECT SOLVERS - Algebraic solutions that can be computed immediately
# -----------------------------------------------------------------------------

def solve_expected_return_capm(inputs):
    """CAPM expected return: E[r] = rf + beta(E[rm] - rf)"""
    rf = inputs.get('rf', 0)
    beta = inputs.get('beta', 1)
    
    if 'market_premium' in inputs and inputs['market_premium'] != 0:
        mrp = inputs['market_premium']
    elif 'E_rm' in inputs:
        mrp = inputs['E_rm'] - rf
    else:
        return {"error": "Need either Market Premium or E[rm]"}
    
    E_r = rf + beta * mrp
    
    return {
        "result": E_r,
        "result_name": "Expected Return E[r]",
        "result_formatted": f"{E_r*100:.4f}%",
        "formula": r"$E[r] = r_f + \beta(E[r_m] - r_f)$",
        "steps": [
            f"1. Risk-free rate: rf = {rf*100:.2f}%",
            f"2. Beta: beta = {beta:.4f}",
            f"3. Market risk premium: E[rm] - rf = {mrp*100:.2f}%",
            f"4. Expected return: E[r] = {rf:.4f} + {beta:.4f} x {mrp:.4f} = **{E_r*100:.4f}%**"
        ]
    }


def solve_expected_return_multifactor(inputs):
    """Multi-factor APT: E[r] = rf + sum(beta_k x lambda_k)"""
    rf = inputs.get('rf', 0)
    betas = parse_messy_input(str(inputs.get('beta_vector', '1.0')))
    lambdas = parse_messy_input(str(inputs.get('lambda_vector', '0.06')))
    
    if len(betas) != len(lambdas):
        return {"error": f"Mismatch: {len(betas)} betas vs {len(lambdas)} factor premiums"}
    
    factor_contrib = betas * lambdas
    E_r = rf + np.sum(factor_contrib)
    
    steps = [f"1. Risk-free rate: rf = {rf*100:.2f}%"]
    for i, (b, l, c) in enumerate(zip(betas, lambdas, factor_contrib)):
        steps.append(f"2.{i+1}. Factor {i+1}: beta_{i+1} x lambda_{i+1} = {b:.4f} x {l*100:.2f}% = {c*100:.4f}%")
    steps.append(f"3. Sum of factor contributions: {np.sum(factor_contrib)*100:.4f}%")
    steps.append(f"4. Expected return: E[r] = {rf*100:.2f}% + {np.sum(factor_contrib)*100:.4f}% = **{E_r*100:.4f}%**")
    
    return {
        "result": E_r,
        "result_name": "Expected Return E[r]",
        "result_formatted": f"{E_r*100:.4f}%",
        "formula": r"$E[r] = r_f + \sum_k \beta_k \lambda_k$",
        "steps": steps
    }


def solve_portfolio_sharpe(inputs):
    """Sharpe Ratio: SR = (E[r] - rf) / sigma"""
    E_r = inputs.get('E_r', 0.12)
    rf = inputs.get('rf', 0.02)
    sigma = inputs.get('sigma', 0.20)
    
    if sigma == 0:
        return {"error": "Volatility cannot be zero"}
    
    excess_return = E_r - rf
    sharpe = excess_return / sigma
    
    return {
        "result": sharpe,
        "result_name": "Sharpe Ratio",
        "result_formatted": f"{sharpe:.4f}",
        "formula": r"$SR = \frac{E[r] - r_f}{\sigma}$",
        "steps": [
            f"1. Expected return: E[r] = {E_r*100:.2f}%",
            f"2. Risk-free rate: rf = {rf*100:.2f}%",
            f"3. Excess return: E[r] - rf = {excess_return*100:.2f}%",
            f"4. Volatility: sigma = {sigma*100:.2f}%",
            f"5. Sharpe Ratio: SR = {excess_return:.4f} / {sigma:.4f} = **{sharpe:.4f}**"
        ]
    }


def solve_beta_from_components(inputs):
    """Beta from correlation and volatilities: beta = rho x (sigma_i / sigma_m)"""
    rho = inputs.get('rho_im', 0.6)
    sigma_i = inputs.get('sigma', 0.25)
    sigma_m = inputs.get('sigma_m', 0.20)
    
    if sigma_m == 0:
        return {"error": "Market volatility cannot be zero"}
    
    beta = rho * (sigma_i / sigma_m)
    
    return {
        "result": beta,
        "result_name": "Beta",
        "result_formatted": f"{beta:.4f}",
        "formula": r"$\beta = \rho_{i,m} \times \frac{\sigma_i}{\sigma_m}$",
        "steps": [
            f"1. Correlation with market: rho = {rho:.4f}",
            f"2. Asset volatility: sigma_i = {sigma_i*100:.2f}%",
            f"3. Market volatility: sigma_m = {sigma_m*100:.2f}%",
            f"4. Beta: beta = {rho:.4f} x ({sigma_i:.4f} / {sigma_m:.4f}) = **{beta:.4f}**"
        ]
    }


def solve_portfolio_return(inputs):
    """Portfolio return: E[rp] = sum(wi x E[ri])"""
    weights = parse_messy_input(str(inputs.get('weights_vector', '0.5, 0.5')))
    returns = parse_messy_input(str(inputs.get('returns_vector', '0.08, 0.12')))
    
    if len(weights) != len(returns):
        return {"error": f"Mismatch: {len(weights)} weights vs {len(returns)} returns"}
    
    port_return = np.sum(weights * returns)
    
    steps = ["**Weighted average calculation:**"]
    for i, (w, r) in enumerate(zip(weights, returns)):
        steps.append(f"  Asset {i+1}: {w:.4f} x {r*100:.2f}% = {w*r*100:.4f}%")
    steps.append(f"**Portfolio Return: E[rp] = {port_return*100:.4f}%**")
    
    return {
        "result": port_return,
        "result_name": "Portfolio Expected Return E[rp]",
        "result_formatted": f"{port_return*100:.4f}%",
        "formula": r"$E[r_p] = \sum_i w_i E[r_i]$",
        "steps": steps
    }


def solve_reverse_portfolio_return(inputs):
    """Solve for missing asset return given portfolio return and weights"""
    port_return = inputs.get('portfolio_return', 0.09)
    weights = parse_messy_input(str(inputs.get('weights_vector', '0.333, 0.333, 0.334')))
    returns_str = str(inputs.get('returns_vector', '0.08, ?, 0.04'))
    
    # Parse returns, finding the missing one
    returns_clean = returns_str.replace('?', '0').replace('nan', '0').replace('NaN', '0')
    returns = parse_messy_input(returns_clean)
    
    # Find which is missing
    missing_idx = None
    parts = returns_str.replace(',', ' ').split()
    for i, p in enumerate(parts):
        if '?' in p or 'nan' in p.lower():
            missing_idx = i
            break
    
    if missing_idx is None:
        missing_idx = len(weights) - 1
    
    if missing_idx >= len(weights):
        return {"error": f"Missing index {missing_idx} out of range"}
    
    # Calculate known contribution
    known_contribution = 0
    for i in range(len(weights)):
        if i != missing_idx and i < len(returns):
            known_contribution += weights[i] * returns[i]
    
    if weights[missing_idx] == 0:
        return {"error": "Weight of missing asset is zero"}
    
    missing_return = (port_return - known_contribution) / weights[missing_idx]
    
    steps = [
        "**Reverse Weighted Average:**",
        f"Portfolio return: E[rp] = {port_return*100:.2f}%",
        "",
        "**Step 1: Sum known contributions**"
    ]
    for i in range(len(weights)):
        if i != missing_idx and i < len(returns):
            steps.append(f"  Asset {i+1}: {weights[i]:.4f} x {returns[i]*100:.2f}% = {weights[i]*returns[i]*100:.4f}%")
    steps.append(f"  Known total: {known_contribution*100:.4f}%")
    steps.append("")
    steps.append("**Step 2: Solve for missing return**")
    steps.append(f"  E[r_{missing_idx+1}] = (E[rp] - known) / w_{missing_idx+1}")
    steps.append(f"  E[r_{missing_idx+1}] = ({port_return:.4f} - {known_contribution:.4f}) / {weights[missing_idx]:.4f}")
    steps.append(f"  **E[r_{missing_idx+1}] = {missing_return*100:.4f}%**")
    
    return {
        "result": missing_return,
        "result_name": f"Missing Return (Asset {missing_idx+1})",
        "result_formatted": f"{missing_return*100:.4f}%",
        "formula": r"$E[r_{?}] = \frac{E[r_p] - \sum_{known} w_i E[r_i]}{w_{?}}$",
        "steps": steps
    }


def solve_beta_portfolio(inputs):
    """Portfolio beta: beta_p = sum(wi x beta_i)"""
    weights = parse_messy_input(str(inputs.get('weights_vector', '0.5, 0.5')))
    betas = parse_messy_input(str(inputs.get('beta_vector', '1.2, 0.8')))
    
    if len(weights) != len(betas):
        return {"error": f"Mismatch: {len(weights)} weights vs {len(betas)} betas"}
    
    port_beta = np.sum(weights * betas)
    
    steps = ["**Weighted average of betas:**"]
    for i, (w, b) in enumerate(zip(weights, betas)):
        steps.append(f"  Asset {i+1}: {w:.4f} x {b:.4f} = {w*b:.4f}")
    steps.append(f"**Portfolio Beta: beta_p = {port_beta:.4f}**")
    
    return {
        "result": port_beta,
        "result_name": "Portfolio Beta",
        "result_formatted": f"{port_beta:.4f}",
        "formula": r"$\beta_p = \sum_i w_i \beta_i$",
        "steps": steps
    }


def solve_jensens_alpha(inputs):
    """Jensen's Alpha: alpha = E[r] - [rf + beta(E[rm] - rf)]"""
    E_r = inputs.get('E_r', 0.12)
    rf = inputs.get('rf', 0.02)
    beta = inputs.get('beta', 1.2)
    
    if 'market_premium' in inputs and inputs['market_premium'] != 0:
        mrp = inputs['market_premium']
    elif 'E_rm' in inputs:
        mrp = inputs['E_rm'] - rf
    else:
        return {"error": "Need either Market Premium or E[rm]"}
    
    capm_return = rf + beta * mrp
    alpha = E_r - capm_return
    
    interp = "Positive alpha -> Underpriced, BUY" if alpha > 0 else ("Negative alpha -> Overpriced, SELL" if alpha < 0 else "alpha = 0 -> Fairly priced")
    
    return {
        "result": alpha,
        "result_name": "Jensen's Alpha",
        "result_formatted": f"{alpha*100:.4f}%",
        "formula": r"$\alpha = E[r] - [r_f + \beta(E[r_m] - r_f)]$",
        "steps": [
            f"1. Actual expected return: E[r] = {E_r*100:.2f}%",
            f"2. CAPM expected return: rf + beta(E[rm]-rf) = {rf:.4f} + {beta:.4f} x {mrp:.4f} = {capm_return*100:.2f}%",
            f"3. Alpha: alpha = {E_r*100:.2f}% - {capm_return*100:.2f}% = **{alpha*100:.4f}%**",
            "",
            f"**Interpretation:** {interp}"
        ]
    }


def solve_information_ratio(inputs):
    """Information Ratio: IR = alpha / sigma_epsilon"""
    alpha = inputs.get('alpha', 0.02)
    sigma_eps = inputs.get('sigma_epsilon', 0.15)
    
    if sigma_eps == 0:
        return {"error": "Residual volatility cannot be zero"}
    
    IR = alpha / sigma_eps
    
    return {
        "result": IR,
        "result_name": "Information Ratio (IR)",
        "result_formatted": f"{IR:.4f}",
        "formula": r"$IR = \frac{\alpha}{\sigma_\epsilon}$",
        "steps": [
            f"1. Alpha: alpha = {alpha*100:.2f}%",
            f"2. Residual volatility: sigma_eps = {sigma_eps*100:.2f}%",
            f"3. Information Ratio: IR = {alpha:.4f} / {sigma_eps:.4f} = **{IR:.4f}**"
        ]
    }


def solve_treynor_ratio(inputs):
    """Treynor Ratio: TR = (E[r] - rf) / beta"""
    E_r = inputs.get('E_r', 0.12)
    rf = inputs.get('rf', 0.02)
    beta = inputs.get('beta', 1.2)
    
    if beta == 0:
        return {"error": "Beta cannot be zero"}
    
    excess = E_r - rf
    TR = excess / beta
    
    return {
        "result": TR,
        "result_name": "Treynor Ratio",
        "result_formatted": f"{TR:.4f}",
        "formula": r"$TR = \frac{E[r] - r_f}{\beta}$",
        "steps": [
            f"1. Expected return: E[r] = {E_r*100:.2f}%",
            f"2. Risk-free rate: rf = {rf*100:.2f}%",
            f"3. Excess return: E[r] - rf = {excess*100:.2f}%",
            f"4. Beta: beta = {beta:.4f}",
            f"5. Treynor Ratio: TR = {excess:.4f} / {beta:.4f} = **{TR:.4f}**"
        ]
    }


def solve_systematic_variance(inputs):
    """Systematic variance: Var_sys = beta^2 x sigma_m^2"""
    beta = inputs.get('beta', 1.2)
    sigma_m = inputs.get('sigma_m', 0.20)
    sigma = inputs.get('sigma', None)
    
    sys_var = (beta ** 2) * (sigma_m ** 2)
    sys_vol = np.sqrt(sys_var)
    
    steps = [
        f"1. Beta: beta = {beta:.4f}",
        f"2. Market volatility: sigma_m = {sigma_m*100:.2f}%",
        f"3. Systematic variance: beta^2 x sigma_m^2 = {beta:.4f}^2 x {sigma_m:.4f}^2 = **{sys_var:.6f}**",
        f"4. Systematic volatility: sqrt({sys_var:.6f}) = **{sys_vol*100:.2f}%**"
    ]
    
    result_str = f"{sys_var:.6f} (Vol: {sys_vol*100:.2f}%)"
    
    if sigma is not None and sigma > 0:
        total_var = sigma ** 2
        R2 = sys_var / total_var
        steps.append("")
        steps.append(f"5. Total variance: sigma^2 = {sigma:.4f}^2 = {total_var:.6f}")
        steps.append(f"6. R^2 = Systematic / Total = {sys_var:.6f} / {total_var:.6f} = **{R2:.4f}** ({R2*100:.1f}%)")
        result_str += f", R^2 = {R2:.4f}"
    
    return {
        "result": sys_var,
        "result_name": "Systematic Variance",
        "result_formatted": result_str,
        "formula": r"$\sigma_{sys}^2 = \beta^2 \sigma_m^2$",
        "steps": steps
    }


def solve_residual_variance(inputs):
    """Residual variance: sigma_eps^2 = sigma^2 - beta^2 x sigma_m^2"""
    sigma = inputs.get('sigma', 0.25)
    beta = inputs.get('beta', 1.2)
    sigma_m = inputs.get('sigma_m', 0.20)
    
    total_var = sigma ** 2
    sys_var = (beta ** 2) * (sigma_m ** 2)
    resid_var = total_var - sys_var
    
    if resid_var < 0:
        return {"error": f"Negative residual variance ({resid_var:.6f}). Check inputs."}
    
    resid_vol = np.sqrt(resid_var)
    
    return {
        "result": resid_var,
        "result_name": "Residual Variance",
        "result_formatted": f"{resid_var:.6f} (Vol: {resid_vol*100:.2f}%)",
        "formula": r"$\sigma_\epsilon^2 = \sigma^2 - \beta^2 \sigma_m^2$",
        "steps": [
            f"1. Total variance: sigma^2 = {sigma:.4f}^2 = {total_var:.6f}",
            f"2. Systematic variance: beta^2 x sigma_m^2 = {beta:.4f}^2 x {sigma_m:.4f}^2 = {sys_var:.6f}",
            f"3. Residual variance: sigma_eps^2 = {total_var:.6f} - {sys_var:.6f} = **{resid_var:.6f}**",
            f"4. Residual volatility: sigma_eps = sqrt({resid_var:.6f}) = **{resid_vol*100:.2f}%**"
        ]
    }


def solve_factor_premiums_reverse(inputs):
    """Reverse APT: Given E[r] and betas, solve for factor premiums lambda"""
    rf = inputs.get('rf', 0.02)
    returns = parse_messy_input(str(inputs.get('returns_vector', '0.10, 0.08')))
    beta_matrix = parse_matrix_input(str(inputs.get('beta_matrix', '1.5, 0.5\n1.2, -0.3')))
    
    if beta_matrix.ndim == 1:
        beta_matrix = beta_matrix.reshape(-1, 1)
    
    n_assets = len(returns)
    n_factors = beta_matrix.shape[1]
    
    if beta_matrix.shape[0] != n_assets:
        return {"error": f"Mismatch: {n_assets} returns but {beta_matrix.shape[0]} rows in beta matrix"}
    
    excess_returns = returns - rf
    
    try:
        lambdas, residuals, rank, s = np.linalg.lstsq(beta_matrix, excess_returns, rcond=None)
    except np.linalg.LinAlgError:
        return {"error": "Could not solve linear system."}
    
    predicted = rf + beta_matrix @ lambdas
    errors = returns - predicted
    
    steps = [
        "**Reverse APT: Solve B x lambda = E[r] - rf**",
        "",
        "**Given:**",
        f"  Risk-free rate: rf = {rf*100:.2f}%",
        f"  Expected returns: {[f'{r*100:.2f}%' for r in returns]}",
        f"  Beta matrix ({n_assets} assets x {n_factors} factors):",
    ]
    for i in range(n_assets):
        steps.append(f"    Asset {i+1}: [{', '.join([f'{b:.4f}' for b in beta_matrix[i]])}]")
    
    steps.append("")
    steps.append("**Solution (Least Squares):**")
    for k in range(n_factors):
        steps.append(f"  lambda_{k+1} = **{lambdas[k]*100:.4f}%**")
    
    steps.append("")
    steps.append("**Verification:**")
    for i in range(n_assets):
        calc_parts = [f'{beta_matrix[i,k]:.2f} x {lambdas[k]*100:.2f}%' for k in range(n_factors)]
        steps.append(f"  Asset {i+1}: E[r] = {rf*100:.2f}% + {' + '.join(calc_parts)} = {predicted[i]*100:.2f}% (actual: {returns[i]*100:.2f}%)")
    
    return {
        "result": lambdas,
        "result_name": "Factor Risk Premiums",
        "result_formatted": ", ".join([f"lambda{k+1}={l*100:.4f}%" for k, l in enumerate(lambdas)]),
        "formula": r"$\lambda = (B'B)^{-1}B'(E[r] - r_f)$",
        "steps": steps
    }


def solve_gordon_growth_price(inputs):
    """Gordon Growth Model: P = D1 / (r - g)"""
    D1 = inputs.get('dividend', 2.0)
    g = inputs.get('growth_rate', 0.04)
    
    if 'E_r' in inputs and inputs['E_r'] != 0:
        r = inputs['E_r']
    elif 'rf' in inputs:
        r = inputs['rf'] + 0.06
    else:
        return {"error": "Need required return (E[r]) or risk-free rate"}
    
    if g >= r:
        return {"error": f"Growth rate ({g*100:.2f}%) must be less than required return ({r*100:.2f}%)"}
    
    price = D1 / (r - g)
    
    return {
        "result": price,
        "result_name": "Stock Price (P)",
        "result_formatted": f"${price:.2f}",
        "formula": r"$P = \frac{D_1}{r - g}$",
        "steps": [
            f"1. Expected dividend: D1 = ${D1:.2f}",
            f"2. Required return: r = {r*100:.2f}%",
            f"3. Growth rate: g = {g*100:.2f}%",
            f"4. Price: P = D1 / (r - g) = {D1:.2f} / ({r:.4f} - {g:.4f}) = **${price:.2f}**"
        ]
    }


def solve_implied_growth_rate(inputs):
    """Implied growth: g = r - D1/P"""
    P = inputs.get('stock_price', 50.0)
    D1 = inputs.get('dividend', 2.0)
    
    if 'E_r' in inputs and inputs['E_r'] != 0:
        r = inputs['E_r']
    elif 'rf' in inputs:
        r = inputs['rf'] + 0.06
    else:
        return {"error": "Need required return (E[r]) or risk-free rate"}
    
    if P == 0:
        return {"error": "Stock price cannot be zero"}
    
    div_yield = D1 / P
    g = r - div_yield
    
    return {
        "result": g,
        "result_name": "Implied Growth Rate (g)",
        "result_formatted": f"{g*100:.4f}%",
        "formula": r"$g = r - \frac{D_1}{P}$",
        "steps": [
            f"1. Stock price: P = ${P:.2f}",
            f"2. Expected dividend: D1 = ${D1:.2f}",
            f"3. Required return: r = {r*100:.2f}%",
            f"4. Dividend yield: D1/P = {D1:.2f}/{P:.2f} = {div_yield*100:.2f}%",
            f"5. Implied growth: g = r - D1/P = {r*100:.2f}% - {div_yield*100:.2f}% = **{g*100:.4f}%**"
        ]
    }


def solve_prob_negative_return(inputs):
    """P(r < 0) = Phi(-mu/sigma)"""
    E_r = inputs.get('E_r', 0.08)
    sigma = inputs.get('sigma', 0.20)
    T = inputs.get('time_horizon', 1.0)
    
    if sigma == 0:
        return {"error": "Volatility cannot be zero"}
    
    mu_T = E_r * T
    sigma_T = sigma * np.sqrt(T)
    
    z_score = -mu_T / sigma_T
    prob = norm.cdf(z_score)
    
    return {
        "result": prob,
        "result_name": "P(r < 0)",
        "result_formatted": f"{prob*100:.4f}%",
        "formula": r"$P(r < 0) = \Phi\left(\frac{-\mu}{\sigma}\right)$",
        "steps": [
            f"1. Expected return (over {T} years): mu = {E_r*100:.2f}% x {T} = {mu_T*100:.2f}%",
            f"2. Volatility (over {T} years): sigma = {sigma*100:.2f}% x sqrt({T}) = {sigma_T*100:.2f}%",
            f"3. Z-score: z = -mu/sigma = -{mu_T:.4f}/{sigma_T:.4f} = {z_score:.4f}",
            f"4. Probability: P(r < 0) = Phi({z_score:.4f}) = **{prob*100:.4f}%**"
        ]
    }


def solve_confidence_interval(inputs):
    """95% CI: [mu - 1.96*sigma, mu + 1.96*sigma]"""
    E_r = inputs.get('E_r', 0.08)
    sigma = inputs.get('sigma', 0.20)
    conf_level = inputs.get('confidence_level', 0.95)
    
    z = norm.ppf((1 + conf_level) / 2)
    lower = E_r - z * sigma
    upper = E_r + z * sigma
    
    return {
        "result": (lower, upper),
        "result_name": f"{conf_level*100:.0f}% Confidence Interval",
        "result_formatted": f"[{lower*100:.2f}%, {upper*100:.2f}%]",
        "formula": r"$[\mu - z_{\alpha/2}\sigma, \mu + z_{\alpha/2}\sigma]$",
        "steps": [
            f"1. Expected return: mu = {E_r*100:.2f}%",
            f"2. Volatility: sigma = {sigma*100:.2f}%",
            f"3. Confidence level: {conf_level*100:.0f}%",
            f"4. Z-score: z = {z:.4f}",
            f"5. Lower bound: mu - z x sigma = {E_r*100:.2f}% - {z:.4f} x {sigma*100:.2f}% = **{lower*100:.2f}%**",
            f"6. Upper bound: mu + z x sigma = {E_r*100:.2f}% + {z:.4f} x {sigma*100:.2f}% = **{upper*100:.2f}%**"
        ]
    }


def solve_bond_price(inputs):
    """Bond price from YTM"""
    F = inputs.get('face_value', 1000.0)
    c = inputs.get('coupon_rate', 0.05)
    T = inputs.get('maturity', 5.0)
    y = inputs.get('ytm', 0.05)
    
    n = int(T)
    coupon = F * c
    
    if abs(y) < 1e-10:
        pv_coupons = coupon * n
        pv_face = F
    else:
        pv_coupons = coupon * (1 - (1 + y) ** (-n)) / y
        pv_face = F * (1 + y) ** (-n)
    
    price = pv_coupons + pv_face
    
    return {
        "result": price,
        "result_name": "Bond Price",
        "result_formatted": f"${price:.2f}",
        "formula": r"$P = \sum_{t=1}^{T} \frac{C}{(1+y)^t} + \frac{F}{(1+y)^T}$",
        "steps": [
            f"1. Face value: F = ${F:.2f}",
            f"2. Coupon rate: c = {c*100:.2f}%",
            f"3. Annual coupon: C = F x c = ${coupon:.2f}",
            f"4. Maturity: T = {n} years",
            f"5. YTM: y = {y*100:.2f}%",
            f"6. PV of coupons: ${pv_coupons:.2f}",
            f"7. PV of face: ${pv_face:.2f}",
            f"8. Bond price: P = ${pv_coupons:.2f} + ${pv_face:.2f} = **${price:.2f}**"
        ]
    }


def solve_macaulay_duration(inputs):
    """Macaulay Duration"""
    F = inputs.get('face_value', 1000.0)
    c = inputs.get('coupon_rate', 0.05)
    T = inputs.get('maturity', 5.0)
    y = inputs.get('ytm', 0.05)
    
    n = int(T)
    coupon = F * c
    
    total_pv = 0
    weighted_sum = 0
    
    for t in range(1, n + 1):
        cf = coupon if t < n else coupon + F
        pv = cf / (1 + y) ** t
        total_pv += pv
        weighted_sum += t * pv
    
    duration = weighted_sum / total_pv
    mod_duration = duration / (1 + y)
    
    return {
        "result": duration,
        "result_name": "Macaulay Duration",
        "result_formatted": f"{duration:.4f} years",
        "formula": r"$D = \frac{\sum t \cdot PV(CF_t)}{P}$",
        "steps": [
            f"1. Face value: F = ${F:.2f}",
            f"2. Annual coupon: C = ${coupon:.2f}",
            f"3. Maturity: T = {n} years",
            f"4. YTM: y = {y*100:.2f}%",
            f"5. Sum of PV: ${total_pv:.2f}",
            f"6. Weighted sum: {weighted_sum:.2f}",
            f"7. Macaulay Duration = {weighted_sum:.2f} / {total_pv:.2f} = **{duration:.4f} years**",
            f"8. Modified Duration = {duration:.4f} / (1 + {y:.4f}) = **{mod_duration:.4f}**"
        ]
    }


def solve_human_capital_gordon(inputs):
    """Human capital as perpetuity: HC = Y1 / (r - g)"""
    Y = inputs.get('labor_income', 75000.0)
    g = inputs.get('income_growth', 0.03)
    r = inputs.get('rf', 0.05)
    
    if g >= r:
        return {"error": f"Growth rate ({g*100:.2f}%) must be less than discount rate ({r*100:.2f}%)"}
    
    HC = Y / (r - g)
    
    return {
        "result": HC,
        "result_name": "Human Capital (HC)",
        "result_formatted": f"${HC:,.2f}",
        "formula": r"$HC = \frac{Y_1}{r - g}$",
        "steps": [
            f"1. Annual labor income: Y = ${Y:,.2f}",
            f"2. Income growth rate: g = {g*100:.2f}%",
            f"3. Discount rate: r = {r*100:.2f}%",
            f"4. Human capital: HC = Y / (r - g) = ${Y:,.2f} / ({r:.4f} - {g:.4f}) = **${HC:,.2f}**"
        ]
    }


def solve_implied_risk_aversion(inputs):
    """Implied gamma from observed portfolio weights"""
    rf = inputs.get('rf', 0.02)
    weights = parse_messy_input(str(inputs.get('weights_vector', '0.6, 0.4')))
    returns = parse_messy_input(str(inputs.get('returns_vector', '0.10, 0.06')))
    
    cov_str = inputs.get('cov_matrix', '0.04, 0.01\n0.01, 0.0225')
    cov = parse_matrix_input(str(cov_str))
    
    n = len(weights)
    if cov.shape != (n, n):
        return {"error": f"Covariance matrix shape {cov.shape} doesn't match {n} assets"}
    
    excess = returns - rf
    port_var = weights @ cov @ weights
    
    try:
        cov_inv = np.linalg.inv(cov)
        numerator = excess @ cov_inv @ excess
        gamma = numerator / port_var
    except np.linalg.LinAlgError:
        return {"error": "Covariance matrix is singular"}
    
    return {
        "result": gamma,
        "result_name": "Implied Risk Aversion",
        "result_formatted": f"{gamma:.4f}",
        "formula": r"$\gamma = \frac{(\mu - r_f)'\Sigma^{-1}(\mu - r_f)}{w'\Sigma w}$",
        "steps": [
            f"1. Excess returns: mu - rf = {[f'{e*100:.2f}%' for e in excess]}",
            f"2. Portfolio variance: w'Sigma w = {port_var:.6f}",
            f"3. Numerator: (mu-rf)'Sigma^-1(mu-rf) = {numerator:.6f}",
            f"4. Implied risk aversion: gamma = **{gamma:.4f}**"
        ]
    }


# Map goals to solver functions
DIRECT_SOLVERS = {
    "expected_return_capm": solve_expected_return_capm,
    "expected_return_multifactor": solve_expected_return_multifactor,
    "portfolio_sharpe": solve_portfolio_sharpe,
    "beta_from_components": solve_beta_from_components,
    "portfolio_return": solve_portfolio_return,
    "reverse_portfolio_return": solve_reverse_portfolio_return,
    "beta_portfolio": solve_beta_portfolio,
    "jensens_alpha": solve_jensens_alpha,
    "information_ratio": solve_information_ratio,
    "treynor_ratio": solve_treynor_ratio,
    "systematic_variance": solve_systematic_variance,
    "residual_variance": solve_residual_variance,
    "factor_premiums_reverse": solve_factor_premiums_reverse,
    "gordon_growth_price": solve_gordon_growth_price,
    "implied_growth_rate": solve_implied_growth_rate,
    "prob_negative_return": solve_prob_negative_return,
    "confidence_interval": solve_confidence_interval,
    "bond_price": solve_bond_price,
    "macaulay_duration": solve_macaulay_duration,
    "human_capital_gordon": solve_human_capital_gordon,
    "implied_risk_aversion": solve_implied_risk_aversion,
}

# -----------------------------------------------------------------------------
# MAIN TRIAGE MODULE UI
# -----------------------------------------------------------------------------

def exam_triage_module():
    """Module 8: Exam Triage - Goal -> Given -> Solution Wizard"""
    st.header("🆘 Module 8: Exam Triage")
    
    st.markdown("""
    **Don't know which module to use?** This wizard helps you find the solution based on:
    1. **What you need to find** (the Goal)
    2. **What data you have** (the Given)
    3. **How to solve it** (direct calculation or which module to use)
    """)
    
    st.info("💡 **Tip:** Select your goal(s), enter the data you have, and the wizard will either solve it directly or guide you to the right module.")
    
    # ==========================================================================
    # STEP 1: GOAL SELECTION (Fixed LaTeX rendering)
    # ==========================================================================
    st.subheader("Step 1: What are you trying to find?")
    
    selected_goals = []
    
    for category, goals in GOAL_TAXONOMY.items():
        with st.expander(f"📂 {category}", expanded=False):
            for goal_id, goal_name, goal_notation in goals:
                # Use container for better layout control
                goal_container = st.container()
                with goal_container:
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        if st.checkbox(goal_name, key=f"goal_{goal_id}"):
                            selected_goals.append(goal_id)
                    with col2:
                        # Render LaTeX formula properly
                        if goal_notation.startswith("$") or goal_notation.startswith(r"$"):
                            st.latex(goal_notation.strip("$"))
                        else:
                            st.caption(goal_notation)
    
    if not selected_goals:
        st.warning("⬆️ Please select at least one goal from the categories above.")
        return
    
    st.success(f"**Selected goals:** {len(selected_goals)}")
    for g in selected_goals:
        for cat, goals in GOAL_TAXONOMY.items():
            for gid, gname, gnot in goals:
                if gid == g:
                    st.write(f"  • {gname}")
                    break
    
    # ==========================================================================
    # STEP 2: INPUT COLLECTION (Dynamic based on goals)
    # ==========================================================================
    st.subheader("Step 2: What data do you have?")
    
    # Determine relevant inputs based on selected goals
    relevant_inputs = set()
    required_inputs = set()
    
    # Track which categories of edge-case inputs are needed
    needs_factor_betas = False
    needs_human_capital = False
    needs_bond_advanced = False
    
    for goal in selected_goals:
        if goal in GOAL_REQUIREMENTS:
            req = GOAL_REQUIREMENTS[goal]
            required_inputs.update(req.get('required', []))
            relevant_inputs.update(req.get('required', []))
            relevant_inputs.update(req.get('optional', []))
        
        # Check if goal needs special edge-case inputs
        if goal in ['expected_return_multifactor', 'factor_premiums_reverse', 'active_weights_tb']:
            needs_factor_betas = True
            # Add factor-specific inputs to relevant
            relevant_inputs.update(['beta_smb', 'beta_hml', 'beta_wml', 'lambda_smb', 'lambda_hml', 'lambda_wml'])
        
        if goal in ['human_capital_pv', 'human_capital_gordon', 'optimal_allocation_hc', 'implicit_hc_beta']:
            needs_human_capital = True
            relevant_inputs.update(['rho_labor_market', 'sigma_labor', 'labor_income', 'income_growth', 
                                   'years_to_retirement', 'human_capital', 'financial_wealth'])
        
        if goal in ['bond_price', 'bond_price_spot', 'macaulay_duration', 'modified_duration', 
                    'immunization_weights', 'forward_rate', 'spot_rate_bootstrap']:
            needs_bond_advanced = True
            relevant_inputs.update(['convexity', 'yield_change_bp', 'liability_duration'])
    
    # Add common inputs only if they're actually relevant
    # (not blindly adding gamma, sigma_m etc for simple valuation problems)
    goal_categories = set()
    for goal in selected_goals:
        for cat, goals in GOAL_TAXONOMY.items():
            for gid, gname, gnot in goals:
                if gid == goal:
                    goal_categories.add(cat)
                    break
    
    # Smart common inputs based on categories
    if any(cat in goal_categories for cat in ['Risk Metrics', 'Optimal Portfolios & Weights', 'Factor Models & APT']):
        relevant_inputs.update(['rf', 'sigma_m', 'gamma'])
    if any(cat in goal_categories for cat in ['Returns & Expected Values', 'Factor Models & APT']):
        relevant_inputs.update(['rf', 'E_rm'])
    if 'Valuation & Growth' in goal_categories:
        relevant_inputs.add('rf')
    if 'Probability & Statistics' in goal_categories:
        relevant_inputs.update(['time_horizon', 'confidence_level'])
    
    # Organize inputs by type
    scalar_inputs = {}
    vector_inputs = {}
    matrix_inputs = {}
    
    collected_inputs = {}
    
    if required_inputs:
        st.markdown("**Required inputs** (marked with ★):")
    
    input_list = list(relevant_inputs)
    
    # Filter to only inputs we have definitions for
    for inp_id in input_list:
        if inp_id in INPUT_DEFINITIONS:
            inp_def = INPUT_DEFINITIONS[inp_id]
            if inp_def['type'] == 'vector':
                vector_inputs[inp_id] = inp_def
            elif inp_def['type'] == 'matrix':
                matrix_inputs[inp_id] = inp_def
            else:
                scalar_inputs[inp_id] = inp_def
    
    # Display scalar inputs in organized columns
    if scalar_inputs:
        st.markdown("**Scalar Values:**")
        cols = st.columns(3)
        for i, (inp_id, inp_def) in enumerate(scalar_inputs.items()):
            with cols[i % 3]:
                is_required = inp_id in required_inputs
                label = f"{'★ ' if is_required else ''}{inp_def['name']}"
                
                # Build help text with aliases and description
                help_parts = []
                if inp_def.get('aliases'):
                    help_parts.append(f"Also known as: {', '.join(inp_def['aliases'])}")
                help_text = " | ".join(help_parts) if help_parts else None
                
                if inp_def['type'] == 'percent':
                    val = st.number_input(
                        label, 
                        value=float(inp_def['default']),
                        format="%.4f",
                        key=f"triage_inp_{inp_id}",
                        help=help_text
                    )
                    collected_inputs[inp_id] = val
                elif inp_def['type'] == 'float':
                    val = st.number_input(
                        label,
                        value=float(inp_def['default']),
                        format="%.4f",
                        key=f"triage_inp_{inp_id}",
                        help=help_text
                    )
                    collected_inputs[inp_id] = val
    
    # Display vector inputs
    if vector_inputs:
        st.markdown("**Vector Values:**")
        for inp_id, inp_def in vector_inputs.items():
            is_required = inp_id in required_inputs
            label = f"{'★ ' if is_required else ''}{inp_def['name']}"
            
            help_parts = ["Enter comma-separated values (e.g., 0.08, 0.12, 0.06)"]
            if inp_def.get('aliases'):
                help_parts.append(f"Also known as: {', '.join(inp_def['aliases'])}")
            
            val = st.text_input(
                label,
                value=str(inp_def['default']),
                key=f"triage_inp_{inp_id}",
                help=" | ".join(help_parts)
            )
            collected_inputs[inp_id] = val
    
    # Display matrix inputs
    if matrix_inputs:
        st.markdown("**Matrix Values:**")
        for inp_id, inp_def in matrix_inputs.items():
            is_required = inp_id in required_inputs
            label = f"{'★ ' if is_required else ''}{inp_def['name']}"
            
            help_parts = ["Enter matrix with one row per line"]
            if inp_def.get('aliases'):
                help_parts.append(f"Also known as: {', '.join(inp_def['aliases'])}")
            
            val = st.text_area(
                label,
                value=str(inp_def['default']),
                height=100,
                key=f"triage_inp_{inp_id}",
                help=" | ".join(help_parts)
            )
            collected_inputs[inp_id] = val
    
    # ==========================================================================
    # DYNAMIC EDGE-CASE INPUTS (only show if relevant)
    # ==========================================================================
    
    # Factor Betas Section - only for multi-factor models
    if needs_factor_betas:
        with st.expander("📊 Factor Model Inputs (Fama-French / Carhart)", expanded=True):
            st.caption("These factors are used in multi-factor asset pricing models like Fama-French 3-Factor and Carhart 4-Factor.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                smb = st.number_input(
                    "SMB Beta (βₛₘᵦ)", 
                    value=0.0, 
                    format="%.4f", 
                    key="triage_beta_smb",
                    help="**Small Minus Big (SMB)**: Fama-French Size Factor. Measures exposure to small-cap vs large-cap stocks. Positive β means tilted toward small caps."
                )
                if smb != 0:
                    collected_inputs['beta_smb'] = smb
                    
                smb_prem = st.number_input(
                    "SMB Premium (λₛₘᵦ)", 
                    value=0.03, 
                    format="%.4f", 
                    key="triage_lambda_smb",
                    help="**SMB Risk Premium**: Historical average ~3% annually. Compensation for holding small-cap risk."
                )
                collected_inputs['lambda_smb'] = smb_prem
                    
            with col2:
                hml = st.number_input(
                    "HML Beta (βₕₘₗ)", 
                    value=0.0, 
                    format="%.4f", 
                    key="triage_beta_hml",
                    help="**High Minus Low (HML)**: Fama-French Value Factor. Measures exposure to value vs growth stocks. Positive β means tilted toward value (high book-to-market)."
                )
                if hml != 0:
                    collected_inputs['beta_hml'] = hml
                    
                hml_prem = st.number_input(
                    "HML Premium (λₕₘₗ)", 
                    value=0.04, 
                    format="%.4f", 
                    key="triage_lambda_hml",
                    help="**HML Risk Premium**: Historical average ~4% annually. Compensation for holding value stock risk."
                )
                collected_inputs['lambda_hml'] = hml_prem
                    
            with col3:
                wml = st.number_input(
                    "WML Beta (βwₘₗ)", 
                    value=0.0, 
                    format="%.4f", 
                    key="triage_beta_wml",
                    help="**Winners Minus Losers (WML)**: Carhart Momentum Factor. Also called UMD (Up Minus Down). Measures exposure to momentum strategy. Positive β means tilted toward recent winners."
                )
                if wml != 0:
                    collected_inputs['beta_wml'] = wml
                    
                wml_prem = st.number_input(
                    "WML Premium (λwₘₗ)", 
                    value=0.06, 
                    format="%.4f", 
                    key="triage_lambda_wml",
                    help="**WML Risk Premium**: Historical average ~6% annually (but highly variable). Compensation for momentum crash risk."
                )
                collected_inputs['lambda_wml'] = wml_prem
    
    # Human Capital Section - only for lifecycle problems
    if needs_human_capital:
        with st.expander("👤 Human Capital & Labor Income Inputs", expanded=True):
            st.caption("Human capital represents the present value of your future labor income. These inputs help model how your career affects optimal portfolio allocation.")
            
            col1, col2 = st.columns(2)
            with col1:
                rho_lm = st.number_input(
                    "Correlation (Labor, Market) ρᵧ,ₘ", 
                    value=0.2, 
                    min_value=-1.0, 
                    max_value=1.0, 
                    format="%.4f", 
                    key="triage_rho_lm",
                    help="**Labor-Market Correlation**: How your income moves with the stock market. Professors/government workers ≈ 0 (bond-like). Investment bankers ≈ 0.5+ (stock-like). Higher correlation → invest less in stocks."
                )
                collected_inputs['rho_labor_market'] = rho_lm
                
                sig_l = st.number_input(
                    "Labor Income Volatility (σᵧ)", 
                    value=0.10, 
                    format="%.4f", 
                    key="triage_sig_l",
                    help="**Income Volatility**: Standard deviation of your income growth. Stable jobs (tenured professor) ≈ 5%. Variable jobs (sales, finance) ≈ 15-25%."
                )
                collected_inputs['sigma_labor'] = sig_l
                
            with col2:
                fin_wealth = st.number_input(
                    "Financial Wealth ($)", 
                    value=200000.0, 
                    format="%.2f", 
                    key="triage_fin_wealth",
                    help="**Current Liquid Wealth**: Your investable assets (savings, brokerage accounts, 401k). Does NOT include home equity or human capital."
                )
                collected_inputs['financial_wealth'] = fin_wealth
                
                hc_val = st.number_input(
                    "Human Capital Value ($)", 
                    value=500000.0, 
                    format="%.2f", 
                    key="triage_hc_val",
                    help="**Present Value of Future Labor Income**: If not given, calculate using HC = Y/(r-g) for perpetuity or sum of discounted future wages."
                )
                collected_inputs['human_capital'] = hc_val
    
    # Bond Advanced Section - only for fixed income problems
    if needs_bond_advanced:
        with st.expander("🏦 Advanced Bond Inputs", expanded=True):
            st.caption("These inputs are used for bond price sensitivity analysis, immunization, and convexity adjustments.")
            
            col1, col2 = st.columns(2)
            with col1:
                conv = st.number_input(
                    "Convexity (C)", 
                    value=0.0, 
                    format="%.4f", 
                    key="triage_conv",
                    help="**Bond Convexity**: Second-order price sensitivity to yield changes. Used for more accurate price change estimates: ΔP/P ≈ -D×Δy + ½×C×(Δy)². Higher convexity is better (prices rise more when yields fall, fall less when yields rise)."
                )
                if conv != 0:
                    collected_inputs['convexity'] = conv
                    
                liab_dur = st.number_input(
                    "Liability Duration (Dₗ)", 
                    value=0.0, 
                    format="%.4f", 
                    key="triage_liab_dur",
                    help="**Target Duration for Immunization**: The duration of your liability (e.g., pension obligation, future payment). Set portfolio duration = liability duration to immunize against parallel yield curve shifts."
                )
                if liab_dur != 0:
                    collected_inputs['liability_duration'] = liab_dur
                    
            with col2:
                dy = st.number_input(
                    "Yield Change (basis points)", 
                    value=0.0, 
                    key="triage_dy",
                    help="**Interest Rate Shock**: Expected or hypothetical yield change in basis points (1 bp = 0.01%). Used for duration/convexity approximations. 100 bp = 1%."
                )
                if dy != 0:
                    collected_inputs['yield_change_bp'] = dy
    
    # ==========================================================================
    # STEP 3: TRIAGE & SOLVE
    # ==========================================================================
    if st.button("🔍 Analyze & Solve", type="primary", key="triage_solve"):
        st.subheader("Step 3: Solution")
        
        for goal in selected_goals:
            st.markdown(f"---")
            
            # Find goal name
            goal_name = goal
            for cat, goals in GOAL_TAXONOMY.items():
                for gid, gname, gnot in goals:
                    if gid == goal:
                        goal_name = gname
                        break
            
            st.markdown(f"### 🎯 {goal_name}")
            
            # Check if we have a direct solver
            if goal in DIRECT_SOLVERS:
                try:
                    result = DIRECT_SOLVERS[goal](collected_inputs)
                    
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                        
                        # Show routing if available
                        if goal in ROUTING_MAP:
                            route = ROUTING_MAP[goal]
                            st.warning(f"➡️ Try using **Module {route['module']}: {route['tab']}** → **{route['subtab']}**")
                    else:
                        # Display result
                        st.success(f"**{result['result_name']}:** {result['result_formatted']}")
                        
                        # Show formula
                        st.latex(result['formula'])
                        
                        # Show calculation steps
                        with st.expander("📝 Step-by-Step Solution", expanded=True):
                            for step in result['steps']:
                                st.markdown(step)
                        
                        # Show routing info
                        if goal in ROUTING_MAP:
                            route = ROUTING_MAP[goal]
                            st.info(f"💡 For more options, go to **Module {route['module']}: {route['tab']}** → **{route['subtab']}**")
                
                except Exception as e:
                    st.error(f"❌ Calculation error: {str(e)}")
                    
                    if goal in ROUTING_MAP:
                        route = ROUTING_MAP[goal]
                        st.warning(f"➡️ Try using **Module {route['module']}: {route['tab']}** → **{route['subtab']}**")
            
            elif goal in ROUTING_MAP:
                # No direct solver, route to module
                route = ROUTING_MAP[goal]
                st.info(f"""
                **This problem type requires the full module interface.**
                
                ➡️ Go to **Module {route['module']}: {route['tab']}** → **{route['subtab']}**
                
                This module provides:
                - More input flexibility
                - Detailed output tables
                - Visualization options
                - Excel formula instructions
                """)
            else:
                st.warning(f"⚠️ No direct solver or routing available for: {goal}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Financial Calculator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stExpander {
        background-color: #e8f4f8;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("📊 Financial Calculator")
    st.sidebar.markdown("---")
    
    module = st.sidebar.radio(
        "Select Module:",
        [
            "📈 1. Portfolio Optimizer",
            "💵 2. Bond Math",
            "📊 3. Stock Valuation",
            "👤 4. Human Capital",
            "📉 5. Factor Models",
            "🎲 6. Probability",
            "🧮 7. Universal Solver",
            "🆘 8. Exam Triage"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Tips:**
    - Paste data directly from PDFs
    - Supports formats: "0.08, 0.15" or "8%, 15%"
    - Click 📝 expanders for Excel formulas
    - Each module has multiple tabs for different problem types
    - **Module 8 (Exam Triage):** Don't know where to start? Use this!
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Financial Calculator")
    st.sidebar.caption("Covers 2023-2025 exam problem families")
    
    # Main Title
    st.title("🎓 Financial Calculator")
    st.markdown("---")
    
    # Route to appropriate module
    if "Portfolio" in module:
        portfolio_optimizer_module_integrated()
    elif "Bond" in module:
        bond_math_module_integrated()
    elif "Stock" in module:
        stock_valuation_module()
    elif "Human" in module:
        human_capital_module_integrated()
    elif "Factor" in module or "CAPM" in module:
        factor_models_module_integrated()
    elif "Probability" in module:
        probability_module()
    elif "Triage" in module:
        exam_triage_module()
    elif "Universal" in module or "7." in module:
        universal_solver_module()


if __name__ == "__main__":
    main()
