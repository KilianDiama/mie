#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OMEGA PRIME CORE 7.0 -- moteur ouvert de conception, calibration et
quantification d'incertitudes pour nanocapteurs plasmoniques (LSPR) a ADN.

numpy + scipy uniquement. Monolithique.

Contenu :
  * Permittivite de l'or Rakic (Lorentz-Drude) + Kreibig additif.
  * Mie exact : sphere pleine et coeur-coquille (Bohren & Huffman).
  * Pic LSPR raffine (Brent), FWHM, sensibilite en volume centree, FOM.
  * Decalage lie a la liaison : coeur-coquille + Maxwell-Garnett.
  * ADN nearest-neighbor SantaLucia 1998, mismatch parametrable,
    propagation Monte-Carlo de l'incertitude sur DeltaG.
  * Thermique analytique (particule, Kapitza, collectif, transitoire).
  * Calibration bayesienne Langmuir (S_max marginalise analytiquement),
    controle predictif a posteriori, identifiabilite, LOD probabiliste.
  * Monte-Carlo optique (rayon, indice, Kreibig), sensibilite OAT.
  * Exploration Pareto, auto-tests, rapports JSON/CSV traces (SHA-256).
  * Champ `epistemic` sur chaque resultat.

Ce logiciel n'est PAS un dispositif medical. Il ne prouve ni efficacite
biologique ni securite. Toute prediction est conditionnelle au modele.
Aucune validation experimentale n'est revendiquee.

Usage :
    python omega_prime_core.py selftest
    python omega_prime_core.py demo
    python omega_prime_core.py spectrum --radius-nm 40
    python omega_prime_core.py shift --radius-nm 40 --theta 1
    python omega_prime_core.py binding --probe ACGTTAGC --target GCTAACGT --ddg-sigma 0.8
    python omega_prime_core.py thermal --radius-nm 40 --intensity 5e5
    python omega_prime_core.py mc-optical --radius-nm 40 --radius-sigma 2 --medium-sigma 0.005
    python omega_prime_core.py sensitivity --radius-nm 40
    python omega_prime_core.py calibrate --data calib.csv --sigma-nm 0.05
    python omega_prime_core.py predict --data calib.csv --conc-nm 50 --sigma-nm 0.05
    python omega_prime_core.py design --target-nm 650
CSV de calibration : conc_nm,shift_nm[,sigma_nm]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import sys
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import erfc, log_ndtr, spherical_jn, spherical_yn
from scipy.stats import truncnorm

__version__ = "7.0.0"
VERSION = __version__
log = logging.getLogger("omega_prime")

# ----------------------------------------------------------------------------
# Constantes physiques
# ----------------------------------------------------------------------------
HC_EV_NM = 1239.841984
HBAR_EV_S = 6.582119569e-16
V_F_GOLD = 1.39e6
R_KCAL = 1.987204e-3
KAPPA_WATER = 0.60
DIFFUSIVITY_WATER = 1.43e-7
DEFAULT_OUT = Path("OMEGA_PRIME_RESULTS")


def _pos(x: float, name: str) -> None:
    if not math.isfinite(float(x)) or float(x) <= 0:
        raise ValueError(f"{name} doit etre fini et strictement positif.")


def epistemic() -> dict:
    return {"status": "MODEL_BASED",
            "internal_validation": "ANALYTIC_LIMITS_AND_CONSISTENCY_TESTS",
            "experimental_validation": "NO",
            "usage_disclaimer": "Sortie conditionnelle au modele ; aucune "
                                "validation experimentale, biologique ou clinique."}


# ----------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------
@dataclass
class OpticalConfig:
    radius_nm: float = 40.0
    medium_index: float = 1.333
    wl_min_nm: float = 400.0
    wl_max_nm: float = 1000.0
    wl_points: int = 601
    use_kreibig: bool = True
    kreibig_a: float = 0.75

    def validate(self) -> None:
        _pos(self.radius_nm, "radius_nm")
        _pos(self.medium_index, "medium_index")
        if not 2.0 <= self.radius_nm <= 500.0:
            raise ValueError("radius_nm hors domaine [2, 500] nm.")
        if not 1.0 <= self.medium_index <= 2.5:
            raise ValueError("medium_index hors domaine [1, 2.5].")
        if not 200.0 <= self.wl_min_nm < self.wl_max_nm <= 2500.0:
            raise ValueError("Fenetre spectrale invalide (domaine 200-2500 nm).")
        if self.wl_points < 101:
            raise ValueError("Au moins 101 points spectraux.")
        if self.kreibig_a < 0:
            raise ValueError("kreibig_a doit etre >= 0.")


@dataclass
class LayerConfig:
    """Couche moleculaire ajoutee a saturation (theta = 1)."""
    thickness_nm: float = 5.0
    n_biomolecule: float = 1.50
    vf_full: float = 0.35

    def validate(self) -> None:
        _pos(self.thickness_nm, "thickness_nm")
        _pos(self.n_biomolecule, "n_biomolecule")
        if not 0.0 <= self.vf_full <= 1.0:
            raise ValueError("vf_full doit etre dans [0, 1].")


@dataclass
class DnaConfig:
    probe: str = "ACGTTAGC"
    target: str = "GCTAACGT"
    conc_nm: float = 50.0
    na_mol_l: float = 0.15
    temp_c: float = 25.0
    mismatch_ddg_kcal: float = 2.0     # heuristique moyenne, a remplacer
    surface_ddg_kcal: float = 0.0      # penalite de surface, a mesurer
    kon_per_m_s: float = 1.0e5         # hypothese, a mesurer

    def validate(self) -> None:
        _pos(self.kon_per_m_s, "kon_per_m_s")
        _pos(self.na_mol_l, "na_mol_l")
        if not math.isfinite(self.conc_nm) or self.conc_nm < 0:
            raise ValueError("conc_nm doit etre fini et >= 0.")
        if not math.isfinite(self.temp_c) or not -20.0 <= self.temp_c <= 120.0:
            raise ValueError("temp_c hors domaine [-20, 120] C.")


@dataclass
class ThermalConfig:
    intensity_w_m2: float = 5.0e5
    wavelength_nm: Optional[float] = None
    heat_fraction: float = 1.0
    kappa_w_mk: float = KAPPA_WATER
    diffusivity_m2_s: float = DIFFUSIVITY_WATER
    interface_conductance_w_m2k: Optional[float] = None
    duration_s: float = 5.0e-3
    particles_in_focus: int = 1
    focus_radius_um: Optional[float] = None
    max_delta_t_c: float = 4.0

    def validate(self) -> None:
        for v, n in ((self.kappa_w_mk, "kappa"), (self.diffusivity_m2_s, "D"),
                     (self.duration_s, "duration_s")):
            _pos(v, n)
        if not math.isfinite(self.intensity_w_m2) or self.intensity_w_m2 < 0:
            raise ValueError("Intensite invalide.")
        if not 0 < self.heat_fraction <= 1:
            raise ValueError("heat_fraction dans ]0, 1].")
        if self.particles_in_focus < 1:
            raise ValueError("particles_in_focus >= 1.")
        if self.particles_in_focus > 1 and not self.focus_radius_um:
            raise ValueError("focus_radius_um requis si plusieurs particules.")
        if self.interface_conductance_w_m2k is not None:
            _pos(self.interface_conductance_w_m2k, "interface_conductance")


@dataclass
class CalibrationConfig:
    kd_min_nm: float = 1.0e-3
    kd_max_nm: float = 1.0e7
    kd_points: int = 600
    smax_prior_nm: float = 50.0
    draws: int = 20000
    seed: int = 20261112
    detection_k_sigma: float = 3.0
    detection_prob: float = 0.95

    def validate(self) -> None:
        if not 0 < self.kd_min_nm < self.kd_max_nm:
            raise ValueError("Bornes Kd invalides.")
        if self.kd_points < 50 or self.draws < 1000:
            raise ValueError("kd_points >= 50 et draws >= 1000.")
        _pos(self.smax_prior_nm, "smax_prior_nm")
        _pos(self.detection_k_sigma, "detection_k_sigma")
        if not 0.5 < self.detection_prob < 1:
            raise ValueError("detection_prob dans ]0.5, 1[.")


# ----------------------------------------------------------------------------
# 1. Permittivite de l'or
# ----------------------------------------------------------------------------
_RAKIC_WP, _RAKIC_F0, _RAKIC_G0 = 9.03, 0.760, 0.053
_RAKIC_F = np.array([0.024, 0.010, 0.071, 0.601, 4.384])
_RAKIC_W = np.array([0.415, 0.830, 2.969, 4.304, 13.32])
_RAKIC_G = np.array([0.241, 0.345, 0.870, 2.494, 2.214])


def _drude(e: np.ndarray, gamma_ev: float) -> np.ndarray:
    return -(_RAKIC_F0 * _RAKIC_WP ** 2) / (e * (e + 1j * gamma_ev))


def gold_permittivity(wl_nm, radius_nm: Optional[float] = None,
                      kreibig_a: float = 0.75) -> np.ndarray:
    """Rakic LD (convention exp(-i w t), Im(eps) > 0). Taille : Gamma0 est
    remplace par Gamma0 + A*vF/R (additif)."""
    wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
    if not np.all(np.isfinite(wl)) or np.any(wl <= 0):
        raise ValueError("Longueurs d'onde invalides.")
    e = (HC_EV_NM / wl).astype(complex)
    eps = 1.0 + _drude(e, _RAKIC_G0)
    for f, w, g in zip(_RAKIC_F, _RAKIC_W, _RAKIC_G):
        eps = eps + f * _RAKIC_WP ** 2 / (w ** 2 - e ** 2 - 1j * g * e)
    if radius_nm is not None:
        _pos(radius_nm, "radius_nm")
        gamma_size = HBAR_EV_S * kreibig_a * V_F_GOLD / (radius_nm * 1e-9)
        eps = eps - _drude(e, _RAKIC_G0) + _drude(e, _RAKIC_G0 + gamma_size)
    if not np.all(np.isfinite(eps)):
        raise FloatingPointError("Permittivite non finie.")
    return eps


# ----------------------------------------------------------------------------
# 2. Mie : sphere pleine et coeur-coquille
# ----------------------------------------------------------------------------
def _psi(n: int, z: np.ndarray):
    j = spherical_jn(n, z)
    return z * j, j + z * spherical_jn(n, z, derivative=True)


def _chi(n: int, z: np.ndarray):
    y = spherical_yn(n, z)
    return -z * y, -(y + z * spherical_yn(n, z, derivative=True))


def _nstop(size: np.ndarray) -> np.ndarray:
    return np.ceil(size + 4.0 * size ** (1.0 / 3.0) + 2.0).astype(int)


def _bare_q(x: np.ndarray, m: np.ndarray):
    qe, qs = np.zeros(len(x)), np.zeros(len(x))
    ns = _nstop(x)
    for n in range(1, int(ns.max()) + 1):
        i = np.nonzero(ns >= n)[0]
        xx, mm = x[i].astype(complex), m[i]
        px, dpx = _psi(n, xx)
        pm, dpm = _psi(n, mm * xx)
        cx, dcx = _chi(n, xx)
        xi, dxi = px - 1j * cx, dpx - 1j * dcx
        da, db = mm * pm * dxi - xi * dpm, pm * dxi - mm * xi * dpm
        if np.any(np.abs(da) < 1e-300) or np.any(np.abs(db) < 1e-300):
            raise FloatingPointError("Coefficient de Mie singulier.")
        an = (mm * pm * dpx - px * dpm) / da
        bn = (pm * dpx - mm * px * dpm) / db
        w = 2 * n + 1
        qe[i] += w * (an + bn).real
        qs[i] += w * (np.abs(an) ** 2 + np.abs(bn) ** 2)
    return 2 * qe / x ** 2, 2 * qs / x ** 2


def _coated_q(x: np.ndarray, y: np.ndarray, m1: np.ndarray, m2: np.ndarray):
    """Efficacites rapportees au rayon externe (y = k*b)."""
    qe, qs = np.zeros(len(x)), np.zeros(len(x))
    ns = _nstop(y)
    for n in range(1, int(ns.max()) + 1):
        i = np.nonzero(ns >= n)[0]
        xx, yy = x[i].astype(complex), y[i].astype(complex)
        a1, a2 = m1[i], m2[i]
        p1, dp1 = _psi(n, a1 * xx)
        p2x, dp2x = _psi(n, a2 * xx)
        c2x, dc2x = _chi(n, a2 * xx)
        p2y, dp2y = _psi(n, a2 * yy)
        c2y, dc2y = _chi(n, a2 * yy)
        py, dpy = _psi(n, yy)
        cy, dcy = _chi(n, yy)
        xy, dxy = py - 1j * cy, dpy - 1j * dcy
        an_ = (a2 * p2x * dp1 - a1 * dp2x * p1) / (a2 * c2x * dp1 - a1 * dc2x * p1)
        bn_ = (a2 * p1 * dp2x - a1 * p2x * dp1) / (a2 * dc2x * p1 - a1 * dp1 * c2x)
        ta, tb = dp2y - an_ * dc2y, dp2y - bn_ * dc2y
        ua, ub = p2y - an_ * c2y, p2y - bn_ * c2y
        da = xy * ta - a2 * dxy * ua
        db = a2 * xy * tb - dxy * ub
        if np.any(np.abs(da) < 1e-300) or np.any(np.abs(db) < 1e-300):
            raise FloatingPointError("Coefficient de Mie coeur-coquille singulier.")
        an = (py * ta - a2 * dpy * ua) / da
        bn = (a2 * py * tb - dpy * ub) / db
        w = 2 * n + 1
        qe[i] += w * (an + bn).real
        qs[i] += w * (np.abs(an) ** 2 + np.abs(bn) ** 2)
    return 2 * qe / y ** 2, 2 * qs / y ** 2


def mie_cross_sections(wl_nm, radius_nm: float, medium_index: float = 1.333,
                       use_kreibig: bool = True, kreibig_a: float = 0.75,
                       shell_nm: float = 0.0,
                       shell_index: Optional[float] = None) -> dict:
    """Sections efficaces (nm^2) d'une sphere d'or (+ coquille dielectrique)
    dans un milieu non absorbant."""
    wl = np.atleast_1d(np.asarray(wl_nm, dtype=float))
    _pos(radius_nm, "radius_nm")
    _pos(medium_index, "medium_index")
    eps = gold_permittivity(wl, radius_nm if use_kreibig else None, kreibig_a)
    m1 = np.sqrt(eps)
    m1 = np.where(m1.imag < 0, -m1, m1) / medium_index
    k = 2.0 * np.pi * medium_index / wl
    x = k * radius_nm
    if shell_nm > 0:
        if shell_index is None:
            raise ValueError("shell_index requis avec shell_nm > 0.")
        _pos(shell_index, "shell_index")
        outer = radius_nm + shell_nm
        m2 = np.full(len(wl), shell_index / medium_index, dtype=complex)
        qe, qs = _coated_q(x, k * outer, m1, m2)
        area = math.pi * outer ** 2
    else:
        qe, qs = _bare_q(x, m1)
        area = math.pi * radius_nm ** 2
    ext, sca = area * qe, area * qs
    ab = ext - sca
    if np.any(ab < -1e-6 * np.maximum(np.abs(ext), 1e-12)):
        raise FloatingPointError("Absorption negative : conservation d'energie violee.")
    if not (np.all(np.isfinite(ext)) and np.all(np.isfinite(sca))):
        raise FloatingPointError("Sections efficaces non finies.")
    return {"extinction_nm2": ext, "scattering_nm2": sca,
            "absorption_nm2": np.maximum(ab, 0.0)}


# ----------------------------------------------------------------------------
# 3. Spectre, pic, FWHM, sensibilite
# ----------------------------------------------------------------------------
def _grid(c: OpticalConfig) -> np.ndarray:
    return np.linspace(c.wl_min_nm, c.wl_max_nm, c.wl_points)


def locate_peak(fun: Callable[[np.ndarray], np.ndarray], wl: np.ndarray,
                y: np.ndarray) -> tuple[float, float, str]:
    """Maximum global sur la grille, raffine par Brent dans les 2 cellules."""
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return float(wl[i]), float(y[i]), "BOUNDARY_PEAK"
    r = minimize_scalar(lambda w: -float(fun(np.array([w]))[0]),
                        bounds=(float(wl[i - 1]), float(wl[i + 1])),
                        method="bounded", options={"xatol": 1e-6})
    return float(r.x), float(-r.fun), "REFINED_PEAK"


def _crossing(x0: float, y0: float, x1: float, y1: float, level: float) -> float:
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def fwhm_nm(wl: np.ndarray, y: np.ndarray, peak_y: float) -> Optional[float]:
    """Largeur a mi-hauteur de l'extinction absolue (sans ligne de base) ;
    None si le demi-maximum n'est pas encadre dans la fenetre."""
    half, i = 0.5 * peak_y, int(np.argmax(y))
    lo = hi = i
    while lo > 0 and y[lo] > half:
        lo -= 1
    while hi < len(y) - 1 and y[hi] > half:
        hi += 1
    if y[lo] > half or y[hi] > half:
        return None
    left = _crossing(wl[lo], y[lo], wl[lo + 1], y[lo + 1], half)
    right = _crossing(wl[hi], y[hi], wl[hi - 1], y[hi - 1], half)
    return float(right - left)


def _ext_fn(c: OpticalConfig, medium_index: Optional[float] = None,
            shell_nm: float = 0.0, shell_index: Optional[float] = None):
    n = medium_index if medium_index is not None else c.medium_index
    return lambda w: mie_cross_sections(w, c.radius_nm, n, c.use_kreibig, c.kreibig_a,
                                        shell_nm, shell_index)["extinction_nm2"]


def extinction_peak(c: OpticalConfig, medium_index: Optional[float] = None,
                    shell_nm: float = 0.0, shell_index: Optional[float] = None):
    f = _ext_fn(c, medium_index, shell_nm, shell_index)
    wl = _grid(c)
    y = f(wl)
    return (*locate_peak(f, wl, y), wl, y)


def compute_spectrum(c: OpticalConfig) -> dict:
    c.validate()
    wl = _grid(c)
    m = mie_cross_sections(wl, c.radius_nm, c.medium_index, c.use_kreibig, c.kreibig_a)

    def fe(w):
        return mie_cross_sections(w, c.radius_nm, c.medium_index, c.use_kreibig,
                                  c.kreibig_a)["extinction_nm2"]

    def fa(w):
        return mie_cross_sections(w, c.radius_nm, c.medium_index, c.use_kreibig,
                                  c.kreibig_a)["absorption_nm2"]

    pe = locate_peak(fe, wl, m["extinction_nm2"])
    pa = locate_peak(fa, wl, m["absorption_nm2"])
    return {"wavelength_nm": wl, **m,
            "peak_extinction_nm": pe[0], "peak_extinction_nm2": pe[1],
            "peak_extinction_status": pe[2],
            "peak_absorption_nm": pa[0], "peak_absorption_nm2": pa[1],
            "peak_absorption_status": pa[2],
            "fwhm_nm": fwhm_nm(wl, m["extinction_nm2"], pe[1]),
            "epistemic": epistemic()}


def bulk_sensitivity(c: OpticalConfig, delta_n: float = 0.01) -> dict:
    """dLambda/dn par difference finie centree, pics raffines."""
    _pos(delta_n, "delta_n")
    c.validate()
    lo = extinction_peak(c, c.medium_index - delta_n)
    hi = extinction_peak(c, c.medium_index + delta_n)
    bad = "BOUNDARY_PEAK" in (lo[2], hi[2])
    return {"sensitivity_nm_per_riu": (hi[0] - lo[0]) / (2 * delta_n),
            "delta_n": delta_n, "peak_low_nm": lo[0], "peak_high_nm": hi[0],
            "status": "UNRELIABLE_PEAK_AT_WINDOW_EDGE" if bad else "OK",
            "epistemic": epistemic()}


# ----------------------------------------------------------------------------
# 4. Couche moleculaire et decalage LSPR
# ----------------------------------------------------------------------------
def layer_index(vf: float, n_medium: float, n_bio: float) -> float:
    """Maxwell-Garnett : inclusions biomoleculaires dans le MILIEU reel."""
    if not 0.0 <= vf <= 1.0:
        raise ValueError("Fraction volumique hors [0, 1].")
    em, ei = n_medium ** 2, n_bio ** 2
    den = ei + 2 * em - vf * (ei - em)
    if abs(den) < 1e-12:
        raise FloatingPointError("Singularite Maxwell-Garnett.")
    eff = em * (ei + 2 * em + 2 * vf * (ei - em)) / den
    if eff <= 0:
        raise FloatingPointError("Indice effectif non physique.")
    return math.sqrt(eff)


def bound_shift(c: OpticalConfig, layer: LayerConfig, theta: float,
                with_estimate: bool = True) -> dict:
    c.validate()
    layer.validate()
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta hors [0, 1].")
    n_layer = layer_index(layer.vf_full * theta, c.medium_index, layer.n_biomolecule)
    base = extinction_peak(c)
    out = {"theta": theta, "layer_index": n_layer, "peak_bare_nm": base[0],
           "status": "COMPUTED_CORE_SHELL_MIE", "epistemic": epistemic()}
    if theta == 0.0:
        out.update(shift_nm=0.0, peak_coated_nm=base[0])
        return out
    coat = extinction_peak(c, shell_nm=layer.thickness_nm, shell_index=n_layer)
    out.update(shift_nm=coat[0] - base[0], peak_coated_nm=coat[0])
    if "BOUNDARY_PEAK" in (base[2], coat[2]):
        out["status"] = "UNRELIABLE_PEAK_AT_WINDOW_EDGE"
    if with_estimate:
        s = bulk_sensitivity(c)["sensitivity_nm_per_riu"]
        frac = 1.0 - (c.radius_nm / (c.radius_nm + layer.thickness_nm)) ** 3
        out["quasi_static_estimate_nm"] = s * (n_layer - c.medium_index) * frac
    return out


# ----------------------------------------------------------------------------
# 5. ADN : nearest-neighbor SantaLucia 1998
# ----------------------------------------------------------------------------
def _revcomp(s: str) -> str:
    return s.translate(str.maketrans("ATGC", "TACG"))[::-1]


def validate_dna(seq: str) -> str:
    s = str(seq).strip().upper()
    if not s or set(s) - set("ATGC"):
        raise ValueError("Sequence ADN invalide (A, T, G, C uniquement).")
    return s


_NN_BASE = {"AA": (-7.9, -22.2), "AT": (-7.2, -20.4), "TA": (-7.2, -21.3),
            "CA": (-8.5, -22.7), "GT": (-8.4, -22.4), "CT": (-7.8, -21.0),
            "GA": (-8.2, -22.2), "CG": (-10.6, -27.2), "GC": (-9.8, -24.4),
            "GG": (-8.0, -19.9)}                       # kcal/mol, cal/(mol.K)
NN = {**_NN_BASE, **{_revcomp(k): v for k, v in _NN_BASE.items()}}


def duplex_thermo(seq: str, na_mol_l: float = 0.15) -> tuple[float, float]:
    """(dH kcal/mol, dS cal/mol/K) d'un duplex parfait, correction de sel."""
    s = validate_dna(seq)
    if len(s) < 2:
        raise ValueError("Au moins 2 bases.")
    _pos(na_mol_l, "na_mol_l")
    dh = ds = 0.0
    for b in (s[0], s[-1]):
        dh, ds = (dh + 0.1, ds - 2.8) if b in "GC" else (dh + 2.3, ds + 4.1)
    for i in range(len(s) - 1):
        dh += NN[s[i:i + 2]][0]
        ds += NN[s[i:i + 2]][1]
    if s == _revcomp(s):
        ds -= 1.4
    ds += 0.368 * (len(s) - 1) * math.log(na_mol_l)
    return dh, ds


def duplex_dg(seq: str, na_mol_l: float, temp_c: float) -> float:
    dh, ds = duplex_thermo(seq, na_mol_l)
    return dh - (temp_c + 273.15) * ds / 1000.0


def binding_occupancy(cfg: DnaConfig) -> dict:
    cfg.validate()
    probe, target = validate_dna(cfg.probe), validate_dna(cfg.target)
    expected = _revcomp(probe)
    if len(expected) != len(target):
        return {"status": "INDETERMINATE_LENGTH_MISMATCH", "probe": probe,
                "target": target, "occupancy": None, "epistemic": epistemic()}
    mm = sum(a != b for a, b in zip(expected, target))
    rt = R_KCAL * (cfg.temp_c + 273.15)
    dg = (duplex_dg(expected, cfg.na_mol_l, cfg.temp_c)
          + mm * cfg.mismatch_ddg_kcal + cfg.surface_ddg_kcal)
    kd_m = math.exp(dg / rt)
    kd_nm = kd_m * 1e9
    theta = cfg.conc_nm / (cfg.conc_nm + kd_nm)
    c_m = cfg.conc_nm * 1e-9
    tau = 1.0 / (cfg.kon_per_m_s * (c_m + kd_m))
    return {"status": "NEAREST_NEIGHBOR_SANTALUCIA_1998", "probe": probe,
            "target": target, "expected_target": expected, "mismatches": mm,
            "dg_kcal_mol": dg, "kd_nm": kd_nm, "occupancy": theta,
            "tau_equilibration_s": tau, "t95_s": math.log(20.0) * tau,
            "limitation": "Penalite de mismatch additive heuristique ; pas de "
                          "boucles, G-U, structures secondaires ; solution != "
                          "surface (surface_ddg_kcal a mesurer).",
            "epistemic": epistemic()}


def binding_mc(cfg: DnaConfig, ddg_sigma_kcal: float, n_draws: int = 4000,
               seed: int = 7) -> dict:
    """Propage une incertitude gaussienne sur DeltaG total vers Kd et theta."""
    if not math.isfinite(ddg_sigma_kcal) or ddg_sigma_kcal < 0:
        raise ValueError("ddg_sigma_kcal doit etre >= 0.")
    if n_draws < 100:
        raise ValueError("n_draws >= 100.")
    base = binding_occupancy(cfg)
    if base.get("occupancy") is None:
        return {"status": "INDETERMINATE", "epistemic": epistemic()}
    rng = np.random.default_rng(seed)
    rt = R_KCAL * (cfg.temp_c + 273.15)
    dg = base["dg_kcal_mol"] + ddg_sigma_kcal * rng.standard_normal(n_draws)
    kd_nm = np.exp(np.clip(dg / rt, -700, 700)) * 1e9
    theta = cfg.conc_nm / (cfg.conc_nm + kd_nm)
    q = lambda v: [float(t) for t in np.quantile(v, [0.025, 0.5, 0.975])]
    return {"status": "MC_DNA_DG_PROPAGATION", "ddg_sigma_kcal": ddg_sigma_kcal,
            "kd_nm_ci95_low_median_high": q(kd_nm),
            "occupancy_ci95_low_median_high": q(theta),
            "limitation": "Sigma sur DeltaG suppose gaussien et fourni par "
                          "l'utilisateur ; non estime par le moteur.",
            "epistemic": epistemic()}


# ----------------------------------------------------------------------------
# 6. Thermique analytique
# ----------------------------------------------------------------------------
def thermal_report(radius_nm: float, cfg: ThermalConfig, opt: OpticalConfig) -> dict:
    cfg.validate()
    o = replace(opt, radius_nm=radius_nm)
    o.validate()
    if cfg.wavelength_nm is None:
        wl = compute_spectrum(o)["peak_absorption_nm"]
    else:
        _pos(cfg.wavelength_nm, "wavelength_nm")
        wl = float(cfg.wavelength_nm)
    cabs = float(mie_cross_sections([wl], radius_nm, o.medium_index, o.use_kreibig,
                                    o.kreibig_a)["absorption_nm2"][0]) * 1e-18
    r = radius_nm * 1e-9
    p = cabs * cfg.intensity_w_m2 * cfg.heat_fraction
    k, d = cfg.kappa_w_mk, cfg.diffusivity_m2_s
    dt_medium = p / (4 * math.pi * k * r)
    dt_kapitza = (p / (4 * math.pi * r ** 2 * cfg.interface_conductance_w_m2k)
                  if cfg.interface_conductance_w_m2k else 0.0)
    tau = r ** 2 / d
    at_t = dt_medium * float(erfc(r / (2 * math.sqrt(d * cfg.duration_s))))
    out = {"status": "ANALYTICAL_CONTINUUM_MODEL", "wavelength_nm": wl,
           "absorption_cross_section_m2": cabs, "absorbed_power_w": p,
           "delta_t_medium_surface_c": dt_medium,
           "delta_t_interface_jump_c": dt_kapitza,
           "delta_t_particle_c": dt_medium + dt_kapitza,
           "delta_t_at_duration_c": at_t, "tau_single_particle_s": tau,
           "delta_t_collective_center_c": 0.0, "tau_collective_s": None,
           "collective_steady_state_reached": None, "epistemic": epistemic()}
    if cfg.particles_in_focus > 1:
        big_l = cfg.focus_radius_um * 1e-6
        out["delta_t_collective_center_c"] = (
            1.5 * cfg.particles_in_focus * p / (4 * math.pi * k * big_l))
        out["tau_collective_s"] = big_l ** 2 / d
        out["collective_steady_state_reached"] = cfg.duration_s >= 3 * big_l ** 2 / d
    total = out["delta_t_particle_c"] + out["delta_t_collective_center_c"]
    out["delta_t_total_upper_bound_c"] = total
    out["threshold_c"] = cfg.max_delta_t_c
    out["threshold_status"] = ("BELOW_CONFIGURED_THRESHOLD"
                               if total <= cfg.max_delta_t_c
                               else "ABOVE_CONFIGURED_THRESHOLD")
    out["limitation"] = ("Continuum, regime permanent pour le collectif, pas de "
                         "convection, source collective uniforme, facteur erfc "
                         "transitoire heuristique (pas la solution exacte).")
    return out


# ----------------------------------------------------------------------------
# 7. Calibration bayesienne : S(c) = S_max * c/(c+Kd)
# ----------------------------------------------------------------------------
def read_calibration_csv(path: str):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    conc, shift, sig = [], [], []
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames or not {"conc_nm", "shift_nm"} <= set(rd.fieldnames):
            raise ValueError("CSV requis : conc_nm,shift_nm[,sigma_nm].")
        has_sig = "sigma_nm" in rd.fieldnames
        for ln, row in enumerate(rd, start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            try:
                conc.append(float(row["conc_nm"]))
                shift.append(float(row["shift_nm"]))
                if has_sig:
                    sig.append(float(row["sigma_nm"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Ligne {ln} invalide : {exc}") from exc
    return np.asarray(conc), np.asarray(shift), (np.asarray(sig) if has_sig else None)


def _log_ndtr_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log(Phi(b) - Phi(a)) stable, b > a."""
    a2, b2 = np.where(a > 0, -b, a), np.where(a > 0, -a, b)
    la, lb = log_ndtr(a2), log_ndtr(b2)
    return lb + np.log1p(-np.exp(la - lb))


def calibrate(conc, shift, sigma, cfg: CalibrationConfig) -> dict:
    cfg.validate()
    refuse = lambda why: {"status": "REFUSED", "reason": why, "epistemic": epistemic()}
    c, y = np.asarray(conc, float), np.asarray(shift, float)
    if c.ndim != 1 or c.shape != y.shape or len(c) < 3:
        return refuse("Au moins 3 paires (conc_nm, shift_nm) requises.")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(y))) or np.any(c < 0):
        return refuse("Donnees non finies ou concentration negative.")
    if sigma is None:
        return refuse("Incertitude de mesure absente (sigma_nm CSV ou --sigma-nm) : "
                      "le moteur n'invente pas la precision experimentale.")
    s = np.asarray(sigma, float)
    if s.shape != y.shape or not np.all(np.isfinite(s)) or np.any(s <= 0):
        return refuse("Incertitudes invalides.")
    if len(np.unique(c[c > 0])) < 2:
        return refuse("Au moins 2 concentrations non nulles distinctes requises.")

    kd = np.exp(np.linspace(math.log(cfg.kd_min_nm), math.log(cfg.kd_max_nm),
                            cfg.kd_points))
    g = c[None, :] / (c[None, :] + kd[:, None])
    w = 1.0 / s ** 2
    a_, b_, c_ = (w * g * g).sum(1), (w * g * y).sum(1), float((w * y * y).sum())
    mu, sd = b_ / a_, 1.0 / np.sqrt(a_)
    hi = cfg.smax_prior_nm
    lo_z, hi_z = -mu / sd, (hi - mu) / sd
    logz = (-0.5 * (c_ - b_ ** 2 / a_) - 0.5 * np.log(a_)
            + _log_ndtr_diff(lo_z, hi_z))
    wts = np.exp(logz - logz.max())
    wts /= wts.sum()

    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(kd), size=cfg.draws, replace=True, p=wts)
    smax = truncnorm.rvs(lo_z[idx], hi_z[idx], loc=mu[idx], scale=sd[idx],
                         random_state=rng)
    kd_s = kd[idx]

    model = smax[:, None] * c[None, :] / (c[None, :] + kd_s[:, None])
    chi_obs = (((y[None, :] - model) / s) ** 2).sum(1)
    chi_rep = (rng.standard_normal(model.shape) ** 2).sum(1)
    ppc_p = float(np.mean(chi_rep >= chi_obs))

    q = lambda v: [float(t) for t in np.quantile(v, [0.025, 0.5, 0.975])]
    kd_ci, s_ci = q(kd_s), q(smax)
    edge = max(2, int(0.02 * cfg.kd_points))
    edge_mass = float(np.mean((idx < edge) | (idx >= cfg.kd_points - edge)))
    ratio = kd_ci[2] / kd_ci[0]
    ident = ("PRIOR_BOUND_DOMINATED" if edge_mass > 0.05
             else "WEAK" if ratio > 30 else "IDENTIFIED")
    pred_med = s_ci[1] * c / (c + kd_ci[1])
    dof = max(1, len(c) - 2)
    informative = float(np.mean((c >= 0.1 * kd_ci[1]) & (c <= 10 * kd_ci[1])))
    return {
        "status": "CALIBRATED",
        "method": "Langmuir ; S_max marginalise (gaussienne tronquee), "
                  "grille log(Kd), priors log-uniforme Kd / uniforme S_max",
        "kd_nm_ci95_low_median_high": kd_ci,
        "smax_nm_ci95_low_median_high": s_ci,
        "identifiability": ident, "kd_ci_ratio": ratio, "kd_edge_mass": edge_mass,
        "informative_points_fraction": informative,
        "posterior_predictive_p": ppc_p,
        "posterior_predictive_ok": bool(0.025 <= ppc_p <= 0.975),
        "reduced_chi_square_at_median": float(np.sum(((y - pred_med) / s) ** 2) / dof),
        "n_observations": int(len(c)), "measurement_sigma_nm": s,
        "seed": cfg.seed, "draws": cfg.draws,
        "limitations": ["Reponse de Langmuir a un site, bruit gaussien independant.",
                        "Derive, erreurs systematiques et correlations non modelisees.",
                        "Un posterior etroit ne prouve pas que le modele est vrai.",
                        "Le prior Kd/S_max influence le resultat si les donnees "
                        "n'encadrent pas Kd."],
        "epistemic": epistemic(),
        "_kd": kd_s, "_smax": smax, "_cfg": cfg,
    }


def predict(cal: dict, conc_nm: float) -> dict:
    if cal.get("status") != "CALIBRATED":
        return {"verdict": "INCONNU", "reason": "Calibration valide absente.",
                "epistemic": epistemic()}
    if not math.isfinite(conc_nm) or conc_nm < 0:
        raise ValueError("Concentration invalide.")
    cfg: CalibrationConfig = cal["_cfg"]
    kd, sm = cal["_kd"], cal["_smax"]
    sig = float(np.median(cal["measurement_sigma_nm"]))
    thr = cfg.detection_k_sigma * sig
    resp = sm * conc_nm / (conc_nm + kd)
    rng = np.random.default_rng(cfg.seed + 1)
    pred = resp + sig * rng.standard_normal(len(resp))
    qs = lambda v: [float(t) for t in np.quantile(v, [0.025, 0.5, 0.975])]
    p_det = float(np.mean(resp > thr))
    with np.errstate(divide="ignore", invalid="ignore"):
        c_lod = np.where(sm > thr, thr * kd / (sm - thr), np.inf)
    verdict = ("DETECTABLE" if p_det >= cfg.detection_prob
               else "NON_DETECTABLE" if p_det <= 1 - cfg.detection_prob
               else "INCONCLUSIVE")
    if cal["identifiability"] != "IDENTIFIED" and verdict != "INCONCLUSIVE":
        verdict += "_SOUS_RESERVE_IDENTIFIABILITE"
    return {"verdict": verdict, "concentration_nm": float(conc_nm),
            "mean_response_nm_ci95": qs(resp),
            "predictive_shift_nm_ci95": qs(pred),
            "detection_threshold_nm": thr,
            "probability_signal_above_threshold": p_det,
            "lod_nm_median": float(np.quantile(c_lod, 0.5)),
            "lod_nm_conservative": float(np.quantile(c_lod, cfg.detection_prob)),
            "warning": "Conditionnel au modele et a la calibration ; "
                       "pas une preuve de detection biologique.",
            "epistemic": epistemic()}


# ----------------------------------------------------------------------------
# 8. Pareto
# ----------------------------------------------------------------------------
def pareto_mask(obj: np.ndarray) -> np.ndarray:
    """Objectifs a MINIMISER, une ligne par candidat."""
    keep = np.ones(len(obj), dtype=bool)
    for i in range(len(obj)):
        dom = np.all(obj <= obj[i], axis=1) & np.any(obj < obj[i], axis=1)
        keep[i] = not dom.any()
    return keep


def design_pareto(medium_index: float, layer: LayerConfig, rmin: float = 10.0,
                  rmax: float = 80.0, steps: int = 15,
                  target_nm: Optional[float] = None) -> dict:
    if not 2 <= rmin < rmax <= 500 or steps < 3:
        raise ValueError("Plage de rayons invalide.")
    cands = []
    for r in np.linspace(rmin, rmax, steps):
        c = OpticalConfig(radius_nm=float(r), medium_index=medium_index, wl_points=401)
        sp, se = compute_spectrum(c), bulk_sensitivity(c)
        sh = bound_shift(c, layer, 1.0, with_estimate=False)
        ok = (sp["peak_extinction_status"] == "REFINED_PEAK"
              and se["status"] == "OK" and sh["status"].startswith("COMPUTED")
              and sp["fwhm_nm"] is not None)
        fom = (se["sensitivity_nm_per_riu"] / sp["fwhm_nm"]) if sp["fwhm_nm"] else 0.0
        cands.append({"radius_nm": float(r), "peak_nm": sp["peak_extinction_nm"],
                      "fwhm_nm": sp["fwhm_nm"],
                      "sensitivity_nm_per_riu": se["sensitivity_nm_per_riu"],
                      "fom": fom, "shift_full_nm": sh["shift_nm"],
                      "gold_volume_nm3": 4 / 3 * math.pi * float(r) ** 3,
                      "valid": ok})
    valid = [x for x in cands if x["valid"]]
    if not valid:
        raise RuntimeError("Aucun candidat valide dans la fenetre spectrale.")
    cols = [[-x["shift_full_nm"], -x["fom"], x["gold_volume_nm3"]] +
            ([abs(x["peak_nm"] - target_nm)] if target_nm else []) for x in valid]
    mask = pareto_mask(np.asarray(cols))
    return {"status": "COMPUTATIONAL_DESIGN_SCREENING",
            "objectives": ["max decalage a saturation (Mie coeur-coquille)",
                           "max FOM = sensibilite / FWHM",
                           "min volume d'or"] +
                          ([f"min |pic - {target_nm} nm|"] if target_nm else []),
            "candidates": cands,
            "pareto_front": [v for v, k in zip(valid, mask) if k],
            "excluded_constraints": ["toxicite", "stabilite colloidale", "cout",
                                     "fonctionnalisation reelle", "polydispersite"],
            "epistemic": epistemic()}


# ----------------------------------------------------------------------------
# 9. Monte-Carlo optique et sensibilite OAT
# ----------------------------------------------------------------------------
def _peak_and_sens(c: OpticalConfig) -> tuple[float, float]:
    sp = compute_spectrum(c)
    se = bulk_sensitivity(c)
    pk = sp["peak_extinction_nm"] if sp["peak_extinction_status"] == "REFINED_PEAK" else np.nan
    sn = se["sensitivity_nm_per_riu"] if se["status"] == "OK" else np.nan
    return float(pk), float(sn)


def optical_mc_uncertainty(base: OpticalConfig, radius_sigma_nm: float = 0.0,
                           medium_sigma: float = 0.0, kreibig_sigma: float = 0.0,
                           n_draws: int = 200, seed: int = 42) -> dict:
    """Propage les incertitudes de rayon, d'indice et de facteur de Kreibig
    sur le pic d'extinction et la sensibilite en volume."""
    base.validate()
    if min(radius_sigma_nm, medium_sigma, kreibig_sigma) < 0:
        raise ValueError("Les sigmas doivent etre >= 0.")
    if n_draws < 20:
        raise ValueError("n_draws >= 20.")
    rng = np.random.default_rng(seed)
    peaks, sens = np.full(n_draws, np.nan), np.full(n_draws, np.nan)
    fails = 0
    for i in range(n_draws):
        r = float(np.clip(base.radius_nm + radius_sigma_nm * rng.standard_normal(),
                          2.0, 500.0))
        nm = float(np.clip(base.medium_index + medium_sigma * rng.standard_normal(),
                           1.0, 2.5))
        ka = float(max(0.0, base.kreibig_a + kreibig_sigma * rng.standard_normal()))
        try:
            peaks[i], sens[i] = _peak_and_sens(
                replace(base, radius_nm=r, medium_index=nm, kreibig_a=ka))
        except (ValueError, FloatingPointError, ArithmeticError):
            fails += 1
    ok = np.isfinite(peaks) & np.isfinite(sens)
    if ok.sum() < 5:
        return {"status": "TOO_FEW_VALID_DRAWS", "n_draws": n_draws,
                "n_failed": fails, "n_valid": int(ok.sum()), "epistemic": epistemic()}
    p, s_ = peaks[ok], sens[ok]
    q = lambda v: [float(t) for t in np.quantile(v, [0.025, 0.5, 0.975])]
    return {"status": "MC_PROPAGATION", "n_draws": n_draws, "n_failed": fails,
            "n_valid": int(ok.sum()), "radius_sigma_nm": radius_sigma_nm,
            "medium_sigma": medium_sigma, "kreibig_sigma": kreibig_sigma,
            "peak_nm_ci95_low_median_high": q(p),
            "sensitivity_nm_per_riu_ci95_low_median_high": q(s_),
            "peak_nm_std": float(np.std(p, ddof=1)),
            "sensitivity_nm_per_riu_std": float(np.std(s_, ddof=1)),
            "limitation": "Sigmas supposes gaussiens et independants ; "
                          "incertitudes de calibration non propagees.",
            "epistemic": epistemic()}


def sensitivity_analysis(base: OpticalConfig, layer: Optional[LayerConfig] = None,
                         rel_step: float = 0.05) -> dict:
    """One-at-a-time : +/- rel_step sur chaque parametre."""
    base.validate()
    if not 0.001 <= rel_step <= 0.5:
        raise ValueError("rel_step dans [0.001, 0.5].")
    p0, s0 = _peak_and_sens(base)
    rows = []
    for name in ("radius_nm", "medium_index", "kreibig_a"):
        v = getattr(base, name)
        lo = _peak_and_sens(replace(base, **{name: max(v * (1 - rel_step), 0.0)
                                             if name == "kreibig_a" else v * (1 - rel_step)}))
        hi = _peak_and_sens(replace(base, **{name: v * (1 + rel_step)}))
        ok = all(np.isfinite([lo[0], hi[0], lo[1], hi[1]]))
        rows.append({"param": name, "value": float(v),
                     "peak_lo_nm": lo[0], "peak_hi_nm": hi[0],
                     "peak_delta_nm": hi[0] - lo[0] if ok else None,
                     "sens_lo": lo[1], "sens_hi": hi[1],
                     "sens_delta": hi[1] - lo[1] if ok else None})
    if layer is not None:
        sh = bound_shift(base, layer, 1.0)
        rows.append({"param": "layer_shift_full_nm", "value": sh["shift_nm"],
                     "note": "decalage coeur-coquille a saturation"})
    return {"status": "OAT_SENSITIVITY", "rel_step": rel_step,
            "baseline_peak_nm": p0, "baseline_sensitivity_nm_per_riu": s0,
            "rows": rows, "epistemic": epistemic()}


# ----------------------------------------------------------------------------
# 10. Export et tracabilite
# ----------------------------------------------------------------------------
def json_safe(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return json_safe(asdict(o))
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, np.generic):
        return json_safe(o.item())
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, TypeError):
        return "unavailable"


def provenance(config: Any = None) -> dict:
    try:
        script = _file_sha256(Path(__file__))
    except NameError:
        script = "unavailable"
    return {"engine": "OMEGA PRIME CORE", "version": VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "script_sha256": script, "experimental_validation": False,
            "config": json_safe(config)}


def save_json(out: Path, name: str, data: Any) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False,
                            allow_nan=False), encoding="utf-8")
    return p


def save_csv(out: Path, name: str, cols: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    keys = list(cols)
    n = len(cols[keys[0]])
    if any(len(cols[k]) != n for k in keys):
        raise ValueError("Colonnes CSV de longueurs differentes.")
    p = out / name
    with p.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(keys)
        for i in range(n):
            wr.writerow([float(cols[k][i]) for k in keys])
    return p


def _scalars(d: dict) -> dict:
    return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}


# ----------------------------------------------------------------------------
# 11. Auto-tests
# ----------------------------------------------------------------------------
def _synthetic_calibration(smax=1.2, kd=30.0, sigma=0.05):
    c = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])
    noise = np.array([0.02, -0.03, 0.04, -0.02, 0.01, 0.03, -0.04])
    return c, smax * c / (c + kd) + noise, np.full_like(c, sigma)


def selftest() -> dict:
    res: list[dict] = []

    def check(name: str, fn: Callable[[], Any]) -> None:
        try:
            ok = fn()
            res.append({"name": name, "passed": ok is not False})
        except Exception as exc:  # noqa: BLE001
            res.append({"name": name, "passed": False,
                        "error": f"{type(exc).__name__}: {exc}"})

    def t_eps():
        e = gold_permittivity([632.8])[0]
        assert -13 < e.real < -9 and 0.5 < e.imag < 2.5
        i = [gold_permittivity([600.0], r)[0].imag for r in (10, 40, 1000)]
        assert i[0] > i[1] > i[2] > gold_permittivity([600.0])[0].imag - 1e-3

    def t_rayleigh():
        r, nm, wl = 5.0, 1.333, np.array([520.0])
        eps = gold_permittivity(wl, r)[0]
        k = 2 * math.pi * nm / wl[0]
        ref = k * (4 * math.pi * r ** 3 * (eps - nm ** 2) / (eps + 2 * nm ** 2)).imag
        got = mie_cross_sections(wl, r, nm)["absorption_nm2"][0]
        assert abs(got / ref - 1) < 0.02, (got, ref)

    def t_lossless():
        qe, qs = _bare_q(np.array([1.0, 3.0]), np.array([1.5 + 0j, 1.33 + 0j]))
        assert np.allclose(qe, qs, rtol=1e-9)

    def t_coated_limits():
        wl = np.linspace(450, 800, 15)
        a = mie_cross_sections(wl, 30.0, 1.333)["extinction_nm2"]
        b = mie_cross_sections(wl, 30.0, 1.333, shell_nm=4.0,
                               shell_index=1.333)["extinction_nm2"]
        assert np.allclose(a, b, rtol=1e-6)
        eps = gold_permittivity(wl, 30.0)
        m1 = np.sqrt(eps)
        m1 = np.where(m1.imag < 0, -m1, m1) / 1.333
        k = 2 * math.pi * 1.333 / wl
        qe_c, _ = _coated_q(k * 30.0, k * 34.0, m1, m1.copy())
        qe_b, _ = _bare_q(k * 34.0, m1)
        assert np.allclose(qe_c, qe_b, rtol=1e-6)

    def t_coated_quasistatic():
        a, t, nm, ns, wl = 5.0, 1.0, 1.333, 1.5, np.array([540.0])
        e1 = gold_permittivity(wl, a)[0]
        e2, em, b = ns ** 2, nm ** 2, a + t
        f = (a / b) ** 3
        alpha = 4 * math.pi * b ** 3 * (
            (e2 - em) * (e1 + 2 * e2) + f * (e1 - e2) * (em + 2 * e2)) / (
            (e2 + 2 * em) * (e1 + 2 * e2) + 2 * f * (e2 - em) * (e1 - e2))
        ref = 2 * math.pi * nm / wl[0] * alpha.imag
        got = mie_cross_sections(wl, a, nm, shell_nm=t,
                                 shell_index=ns)["absorption_nm2"][0]
        assert abs(got / ref - 1) < 0.03, (got, ref)

    def t_energy():
        for r in (2.0, 20.0, 100.0, 250.0, 500.0):
            wl = np.linspace(400, 1000, 61)
            m = mie_cross_sections(wl, r, 1.333)
            assert np.all(m["absorption_nm2"] >= 0)
            assert np.allclose(m["extinction_nm2"],
                               m["scattering_nm2"] + m["absorption_nm2"], rtol=1e-6)

    def t_peak():
        sp = compute_spectrum(OpticalConfig(radius_nm=40.0))
        assert 520 < sp["peak_extinction_nm"] < 600, sp["peak_extinction_nm"]
        assert sp["fwhm_nm"] is None or sp["fwhm_nm"] > 0
        sp2 = compute_spectrum(OpticalConfig(radius_nm=12.0))
        assert 505 < sp2["peak_extinction_nm"] < 545, sp2["peak_extinction_nm"]

    def t_sensitivity():
        s = bulk_sensitivity(OpticalConfig(radius_nm=40.0))
        assert s["status"] == "OK" and 50 < s["sensitivity_nm_per_riu"] < 600, s

    def t_shift():
        c, ly = OpticalConfig(radius_nm=30.0), LayerConfig()
        s0, s5, s1 = (bound_shift(c, ly, t)["shift_nm"] for t in (0.0, 0.5, 1.0))
        assert s0 == 0.0 and 0 < s5 < s1 < 20
        e = bound_shift(c, ly, 1.0)
        assert 0.3 < s1 / e["quasi_static_estimate_nm"] < 3.0, e

    def t_mg():
        assert abs(layer_index(0.0, 1.333, 1.5) - 1.333) < 1e-12
        assert layer_index(0.5, 1.333, 1.5) > layer_index(0.2, 1.333, 1.5)

    def t_dna_symmetry():
        for s in ("GCGATAACG", "ACGTTAGC", "CCATTGGA"):
            a = duplex_dg(s, 0.15, 25.0)
            assert abs(a - duplex_dg(_revcomp(s), 0.15, 25.0)) < 1e-9
        assert set(NN) == {a + b for a in "ACGT" for b in "ACGT"}

    def t_dna_trends():
        base = DnaConfig(probe="ACGTTAGC", target="GCTAACGT")
        mm = DnaConfig(probe="ACGTTAGC", target="GCTTACGT")
        hot = DnaConfig(probe="ACGTTAGC", target="GCTAACGT", temp_c=45.0)
        kb, km, kh = (binding_occupancy(x)["kd_nm"] for x in (base, mm, hot))
        assert km > kb and kh > kb
        assert binding_occupancy(DnaConfig(target="GCTAAC"))["occupancy"] is None

    def t_dna_mc():
        r = binding_mc(DnaConfig(), 0.8, n_draws=500)
        lo, med, hi = r["occupancy_ci95_low_median_high"]
        assert 0 <= lo <= med <= hi <= 1
        r0 = binding_mc(DnaConfig(), 0.0, n_draws=200)
        lo0, _, hi0 = r0["occupancy_ci95_low_median_high"]
        assert abs(hi0 - lo0) < 1e-9

    def t_thermal():
        o = OpticalConfig()
        a = thermal_report(40.0, ThermalConfig(intensity_w_m2=1e5), o)
        b = thermal_report(40.0, ThermalConfig(intensity_w_m2=1e6), o)
        assert abs(b["delta_t_particle_c"] / a["delta_t_particle_c"] - 10) < 1e-9
        assert 0.99 < a["delta_t_at_duration_c"] / a["delta_t_medium_surface_c"] <= 1.0
        z = thermal_report(40.0, ThermalConfig(intensity_w_m2=0.0), o)
        assert z["delta_t_particle_c"] == 0.0
        col = thermal_report(40.0, ThermalConfig(particles_in_focus=1000,
                                                 focus_radius_um=5.0), o)
        assert col["delta_t_collective_center_c"] > 0

    def t_calibration():
        c, y, s = _synthetic_calibration()
        cal = calibrate(c, y, s, CalibrationConfig())
        assert cal["status"] == "CALIBRATED"
        lo, _, hi = cal["kd_nm_ci95_low_median_high"]
        slo, _, shi = cal["smax_nm_ci95_low_median_high"]
        assert lo <= 30.0 <= hi and slo <= 1.2 <= shi, cal
        assert cal["posterior_predictive_ok"]
        assert cal["identifiability"] in ("IDENTIFIED", "WEAK")
        p = predict(cal, 100.0)
        assert p["verdict"].startswith("DETECTABLE") and p["lod_nm_conservative"] < 100.0
        lo2, med, hi2 = p["mean_response_nm_ci95"]
        assert lo2 <= med <= hi2

    def t_reproducible():
        c, y, s = _synthetic_calibration()
        a = calibrate(c, y, s, CalibrationConfig())
        b = calibrate(c, y, s, CalibrationConfig())
        assert a["kd_nm_ci95_low_median_high"] == b["kd_nm_ci95_low_median_high"]

    def t_refusals():
        c, y, s = _synthetic_calibration()
        assert calibrate(c, y, None, CalibrationConfig())["status"] == "REFUSED"
        assert calibrate(c[:2], y[:2], s[:2], CalibrationConfig())["status"] == "REFUSED"
        assert calibrate(c * 0, y, s, CalibrationConfig())["status"] == "REFUSED"
        assert predict({"status": "REFUSED"}, 5.0)["verdict"] == "INCONNU"

    def t_misfit_detected():
        c = np.array([1.0, 10.0, 100.0, 1000.0, 5000.0])
        y = np.array([0.0, 2.0, 0.0, 2.0, 0.0])
        cal = calibrate(c, y, np.full_like(c, 0.05), CalibrationConfig())
        assert not cal["posterior_predictive_ok"]

    def t_bad_inputs():
        for bad in (OpticalConfig(radius_nm=-1), OpticalConfig(medium_index=0.5),
                    OpticalConfig(wl_points=10)):
            try:
                compute_spectrum(bad)
            except ValueError:
                continue
            raise AssertionError("entree invalide acceptee")

    def t_pareto():
        m = pareto_mask(np.array([[1, 1], [2, 2], [0, 3], [3, 0]]))
        assert m.tolist() == [True, False, True, True]

    def t_json():
        s = json.dumps(json_safe({"a": np.bool_(True), "b": np.array([1.0, np.nan]),
                                  "_x": 1, "c": float("inf")}), allow_nan=False)
        assert "_x" not in s

    def t_mc_optical():
        r = optical_mc_uncertainty(OpticalConfig(radius_nm=40.0), 1.0, 0.005, 0.05,
                                   n_draws=20, seed=1)
        assert r["status"] == "MC_PROPAGATION"
        lo, med, hi = r["peak_nm_ci95_low_median_high"]
        assert lo <= med <= hi and 500 < med < 620

    def t_oat():
        r = sensitivity_analysis(OpticalConfig(radius_nm=40.0), LayerConfig(), 0.05)
        assert r["status"] == "OAT_SENSITIVITY" and len(r["rows"]) == 4
        assert np.isfinite(r["baseline_peak_nm"])

    def t_epistemic():
        sp = compute_spectrum(OpticalConfig(radius_nm=40.0))
        assert sp["epistemic"]["experimental_validation"] == "NO"

    tests = (
        ("Permittivite Rakic + Kreibig additif", t_eps),
        ("Limite de Rayleigh (analytique)", t_rayleigh),
        ("Sphere sans pertes : Qext = Qsca", t_lossless),
        ("Coeur-coquille : limites", t_coated_limits),
        ("Coeur-coquille vs quasi-statique", t_coated_quasistatic),
        ("Conservation d'energie, R = 2 a 500 nm", t_energy),
        ("Pic LSPR plausible", t_peak),
        ("Sensibilite en volume", t_sensitivity),
        ("Decalage coeur-coquille ~ estimation QS", t_shift),
        ("Maxwell-Garnett", t_mg),
        ("ADN : symetrie, 16 doublets", t_dna_symmetry),
        ("ADN : Kd croit avec T et mismatch", t_dna_trends),
        ("ADN : propagation Monte-Carlo", t_dna_mc),
        ("Thermique : lineaire, nul, collectif", t_thermal),
        ("Calibration : recouvre les parametres + PPC", t_calibration),
        ("Calibration : reproductible", t_reproducible),
        ("Refus : sigma absent, <3 points, donnees nulles", t_refusals),
        ("PPC detecte un modele faux", t_misfit_detected),
        ("Rejet des entrees invalides", t_bad_inputs),
        ("Pareto : dominance", t_pareto),
        ("Serialisation JSON sure", t_json),
        ("Monte-Carlo optique", t_mc_optical),
        ("Sensibilite OAT", t_oat),
        ("Statut epistemique present", t_epistemic))
    for name, fn in tests:
        check(name, fn)

    passed = sum(r["passed"] for r in res)
    return {"version": VERSION, "passed": passed, "total": len(res),
            "success": passed == len(res), "tests": res, "epistemic": epistemic(),
            "note": "Tests de coherence interne et limites analytiques ; "
                    "aucune validation experimentale."}


# ----------------------------------------------------------------------------
# 12. Demo et CLI
# ----------------------------------------------------------------------------
def _emit(obj: Any) -> None:
    print(json.dumps(json_safe(obj), indent=2, ensure_ascii=False, allow_nan=False))


def _spectrum_csv(out: Path, sp: dict, name: str = "spectrum.csv") -> Path:
    return save_csv(out, name, {k: sp[k] for k in (
        "wavelength_nm", "extinction_nm2", "scattering_nm2", "absorption_nm2")})


def run_demo(out: Path) -> dict:
    opt, layer, dna, th = OpticalConfig(), LayerConfig(), DnaConfig(), ThermalConfig()
    print("[1/8] Spectre Mie ...")
    sp = compute_spectrum(opt)
    fw = sp["fwhm_nm"]
    print(f"      pic {sp['peak_extinction_nm']:.3f} nm, FWHM "
          f"{'n/a' if fw is None else f'{fw:.1f} nm'}")
    print("[2/8] Sensibilite en volume ...")
    se = bulk_sensitivity(opt)
    print(f"      {se['sensitivity_nm_per_riu']:.1f} nm/RIU")
    print("[3/8] ADN nearest-neighbor ...")
    dn = binding_occupancy(dna)
    print(f"      Kd {dn['kd_nm']:.3g} nM, occupation {dn['occupancy']:.4f}")
    dmc = binding_mc(dna, 0.8)
    print(f"[4/8] Incertitude DeltaG : occupation IC95 "
          f"{[round(v, 4) for v in dmc['occupancy_ci95_low_median_high']]}")
    print("[5/8] Decalage coeur-coquille ...")
    sh = bound_shift(opt, layer, dn["occupancy"])
    print(f"      decalage {sh['shift_nm']:.4f} nm "
          f"(QS {sh['quasi_static_estimate_nm']:.4f} nm)")
    print("[6/8] Thermique ...")
    tr = thermal_report(opt.radius_nm, th, opt)
    print(f"      dT {tr['delta_t_particle_c']:.4g} K [{tr['threshold_status']}]")
    print("[7/8] Monte-Carlo optique ...")
    mc = optical_mc_uncertainty(opt, 2.0, 0.005, 0.05, n_draws=100)
    print(f"      pic IC95 {mc.get('peak_nm_ci95_low_median_high')}")
    print("[8/8] Calibration sur donnees SYNTHETIQUES ...")
    c, y, s = _synthetic_calibration()
    save_csv(out, "synthetic_calibration.csv",
             {"conc_nm": c, "shift_nm": y, "sigma_nm": s})
    cal = calibrate(c, y, s, CalibrationConfig())
    pr = predict(cal, 50.0)
    print(f"      Kd IC95 {cal['kd_nm_ci95_low_median_high']}, "
          f"LOD conservateur {pr['lod_nm_conservative']:.3g} nM")
    tests = selftest()
    print(f"Tests : {tests['passed']}/{tests['total']}")
    report = {"provenance": provenance({"optical": opt, "layer": layer,
                                        "dna": dna, "thermal": th}),
              "optical": _scalars(sp), "sensitivity": se, "binding": dn,
              "binding_mc": dmc, "shift": sh, "thermal": tr, "mc_optical": mc,
              "calibration_on_synthetic_data": cal, "prediction": pr,
              "selftest": tests,
              "limitations": ["Donnees de calibration SYNTHETIQUES : demonstration.",
                              "Aucune validation experimentale, biologique ou clinique."],
              "epistemic": epistemic()}
    save_json(out, "complete_report.json", report)
    _spectrum_csv(out, sp)
    print(f"Rapports : {out.resolve()}\nVALIDATION EXPERIMENTALE : NON EFFECTUEE")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"OMEGA PRIME CORE {VERSION}")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="dossier de sortie")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)
    for n in ("selftest", "demo"):
        sub.add_parser(n)

    def optical(sp):
        sp.add_argument("--radius-nm", type=float, default=40.0)
        sp.add_argument("--medium-index", type=float, default=1.333)

    def layer(sp):
        sp.add_argument("--shell-nm", type=float, default=5.0)
        sp.add_argument("--n-bio", type=float, default=1.50)
        sp.add_argument("--vf-full", type=float, default=0.35)

    s = sub.add_parser("spectrum"); optical(s)
    s = sub.add_parser("shift"); optical(s); layer(s)
    s.add_argument("--theta", type=float, default=1.0)
    s = sub.add_parser("binding")
    s.add_argument("--probe", default="ACGTTAGC")
    s.add_argument("--target", default="GCTAACGT")
    s.add_argument("--conc-nm", type=float, default=50.0)
    s.add_argument("--temp-c", type=float, default=25.0)
    s.add_argument("--na-mol-l", type=float, default=0.15)
    s.add_argument("--mismatch-ddg", type=float, default=2.0)
    s.add_argument("--surface-ddg", type=float, default=0.0)
    s.add_argument("--ddg-sigma", type=float, default=None)
    s = sub.add_parser("thermal"); optical(s)
    s.add_argument("--intensity", type=float, default=5e5)
    s.add_argument("--duration-ms", type=float, default=5.0)
    s.add_argument("--wavelength-nm", type=float, default=None)
    s.add_argument("--particles", type=int, default=1)
    s.add_argument("--focus-radius-um", type=float, default=None)
    s.add_argument("--interface-conductance", type=float, default=None)
    s.add_argument("--max-dt", type=float, default=4.0)
    s = sub.add_parser("mc-optical"); optical(s)
    s.add_argument("--radius-sigma", type=float, default=0.0)
    s.add_argument("--medium-sigma", type=float, default=0.0)
    s.add_argument("--kreibig-sigma", type=float, default=0.0)
    s.add_argument("--n-draws", type=int, default=200)
    s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("sensitivity"); optical(s); layer(s)
    s.add_argument("--rel-step", type=float, default=0.05)
    for n in ("calibrate", "predict"):
        s = sub.add_parser(n); optical(s); layer(s)
        s.add_argument("--data", required=True)
        s.add_argument("--sigma-nm", type=float, default=None)
        s.add_argument("--smax-prior-nm", type=float, default=50.0)
        s.add_argument("--seed", type=int, default=20261112)
        if n == "predict":
            s.add_argument("--conc-nm", type=float, required=True)
    s = sub.add_parser("design")
    s.add_argument("--medium-index", type=float, default=1.333); layer(s)
    s.add_argument("--rmin-nm", type=float, default=10.0)
    s.add_argument("--rmax-nm", type=float, default=80.0)
    s.add_argument("--steps", type=int, default=15)
    s.add_argument("--target-nm", type=float, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, a.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out: Path = a.out
    try:
        if a.command == "selftest":
            r = selftest()
            _emit(r)
            return 0 if r["success"] else 1
        if a.command == "demo":
            run_demo(out)
            return 0
        if a.command == "binding":
            cfg = DnaConfig(a.probe, a.target, a.conc_nm, a.na_mol_l, a.temp_c,
                            a.mismatch_ddg, a.surface_ddg)
            r = binding_occupancy(cfg)
            if a.ddg_sigma is not None and r.get("occupancy") is not None:
                r["monte_carlo"] = binding_mc(cfg, a.ddg_sigma)
            save_json(out, "binding_report.json", {"provenance": provenance(cfg), **r})
            _emit(r)
            return 0 if r.get("occupancy") is not None else 2
        if a.command == "design":
            ly = LayerConfig(a.shell_nm, a.n_bio, a.vf_full)
            r = design_pareto(a.medium_index, ly, a.rmin_nm, a.rmax_nm, a.steps,
                              a.target_nm)
            save_json(out, "design_pareto.json", {"provenance": provenance(ly), **r})
            _emit(r)
            return 0

        opt = OpticalConfig(radius_nm=a.radius_nm, medium_index=a.medium_index)
        if a.command == "spectrum":
            sp = compute_spectrum(opt)
            _spectrum_csv(out, sp)
            r = {**_scalars(sp), "sensitivity": bulk_sensitivity(opt)}
            save_json(out, "spectrum_report.json", {"provenance": provenance(opt), **r})
            _emit(r)
            return 0
        if a.command == "shift":
            ly = LayerConfig(a.shell_nm, a.n_bio, a.vf_full)
            r = bound_shift(opt, ly, a.theta)
            save_json(out, "shift_report.json", {"provenance": provenance([opt, ly]), **r})
            _emit(r)
            return 0
        if a.command == "thermal":
            th = ThermalConfig(intensity_w_m2=a.intensity, wavelength_nm=a.wavelength_nm,
                               duration_s=a.duration_ms * 1e-3,
                               particles_in_focus=a.particles,
                               focus_radius_um=a.focus_radius_um,
                               interface_conductance_w_m2k=a.interface_conductance,
                               max_delta_t_c=a.max_dt)
            r = thermal_report(a.radius_nm, th, opt)
            save_json(out, "thermal_report.json", {"provenance": provenance(th), **r})
            _emit(r)
            return 0
        if a.command == "mc-optical":
            r = optical_mc_uncertainty(opt, a.radius_sigma, a.medium_sigma,
                                       a.kreibig_sigma, a.n_draws, a.seed)
            save_json(out, "mc_optical_report.json", {"provenance": provenance(opt), **r})
            _emit(r)
            return 0 if r["status"] == "MC_PROPAGATION" else 2
        if a.command == "sensitivity":
            ly = LayerConfig(a.shell_nm, a.n_bio, a.vf_full)
            r = sensitivity_analysis(opt, ly, a.rel_step)
            save_json(out, "sensitivity_report.json",
                      {"provenance": provenance([opt, ly]), **r})
            _emit(r)
            return 0
        if a.command in ("calibrate", "predict"):
            conc, y, sig = read_calibration_csv(a.data)
            if sig is None and a.sigma_nm is not None:
                _pos(a.sigma_nm, "sigma_nm")
                sig = np.full_like(y, a.sigma_nm)
            ccfg = CalibrationConfig(smax_prior_nm=a.smax_prior_nm, seed=a.seed)
            cal = calibrate(conc, y, sig, ccfg)
            if cal["status"] == "CALIBRATED":
                ly = LayerConfig(a.shell_nm, a.n_bio, a.vf_full)
                mie = bound_shift(opt, ly, 1.0, with_estimate=False)["shift_nm"]
                cal["mie_full_occupancy_shift_nm"] = mie
                cal["smax_over_mie_ratio"] = (
                    cal["smax_nm_ci95_low_median_high"][1] / mie
                    if abs(mie) > 1e-12 else None)
                cal["input_sha256"] = hashlib.sha256(
                    Path(a.data).read_bytes()).hexdigest()
            result = cal
            if a.command == "predict":
                result = {"calibration": cal, "prediction": predict(cal, a.conc_nm)}
            save_json(out, f"{a.command}_report.json",
                      {"provenance": provenance(ccfg), "result": result})
            _emit(result)
            return 0 if cal["status"] == "CALIBRATED" else 2
        return 2
    except (ValueError, RuntimeError, ArithmeticError, OSError, KeyError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        print(f"ERREUR : {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())tu peux me donner le read me github de ce code s'il te plais, en quelque phrases courte, en anglais , avec des perspectives
