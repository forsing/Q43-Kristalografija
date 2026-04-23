#!/usr/bin/env python3

"""
Q43 Kristalografija (granica sa QFT / Bloch bazisom) — periodični potencijal,
Bravais rešetka, Bloch teorema, recipročna rešetka, band struktura — čisto
kvantno.

Paradigma:
  Kristalografija / fizika čvrstog tela proučava kvantnu dinamiku u periodičnim
  sistemima. Ključne konstrukcije:

    • Bravais rešetka (1D): {j · a : j ∈ ℤ},  a = konstanta rešetke (unit cell).
    • Periodični potencijal:  V(j + a) = V(j),  za sve j.
    • Bloch teorema:  eigenstanja Hamiltonijana sa periodičnim potencijalom
      imaju formu  ψ_{n,k}(j) = e^{i k j} · u_{n,k}(j),  gde je u_{n,k}
      periodično sa periodom a, a k ∈ Brillouin zone [−π/a, π/a).
    • Recipročna rešetka: G_m = (2π · m) / a,  m ∈ ℤ.
    • Band struktura: energijska funkcija E_n(k), gde je n indeks opsega,
      k kvazi-impuls. Spektar je organizovan u opsege (bands) razdvojene
      zabranjenim zonama (band gaps).

  Bloch bazis kao Fourier transformacija:
      |k⟩ = (1/√N) · Σ_{j=0..N−1}  e^{2πi k j / N} · |j⟩,   k ∈ {0..N−1}
  Kvantna Fourier transformacija (QFT) preslikava pozicioni bazis |j⟩ u
  Bloch (momentum) bazis |k⟩. Za translaciono-invariantne operatore
  (čisto kinetski H_kin), QFT diagonalizuje: H_kin → diag(ε(k)).

Granica sa QFT paradigmom:
  QFT je tehnički alat; Q28 (QFR) koristi Fourier bazis za FITTING regresione
  amplitude, Q30 (QSP) koristi polinomske funkcije Hamiltonijana preko
  signal-processing tehnike. Q43 koristi QFT samo kao DIJAGNOSTIKU prelaska
  između pozicionog i Bloch bazisa, a centralni objekat je PERIODIČNI
  POTENCIJAL nad 1D Bravais rešetkom — kristalografska paradigma koja nije
  pokrivena prethodnim modelima. Dinamika se određuje kroz spektralno
  rešavanje Mathieu-tipskog problema (tight-binding + kosinusni potencijal +
  harmonic trap), tj. preko band strukture + lokalizacije.

Mapiranje na loto:
  Za svaku poziciju i ∈ {1..7}:
    1) j_target (strukturalni cilj, nije frekvencija):
           target_i(prev) = prev + (N_MAX − prev) / (N_NUMBERS − i + 2)
           j_target = round(target_i) − i  ∈ [0, 32]
    2) Kristalna rešetka: 64 sajta sa konstantom a = 4 (16 unit cells).
       Periodični potencijal:
           V_period(j) = V_0 · cos(2π · j / a)
       To je Mathieu-like potencijal — tipičan za optičke rešetke u ultra-
       hladnim atomima.
    3) Lokalizacioni (Gaussian-trap) potencijal oko j_target:
           V_loc(j) = α · ((j − j_target) / σ)²
       Ograničava wavepacket u okolini cilja (single-cell localization oko
       ciljne unit cell-e).
    4) Tight-binding kinetski član (NN hopping — Chevalley E/F generatori su(64)):
           H_kin = −t · (T̂_+ + T̂_−)
       Energetski spektar slobodne česteke: ε(k) = −2t · cos(k).
    5) Puni Hamiltonijan:
           H = H_kin + V_period(X̂) + V_loc(X̂)
       Za t = 1.0, V_0 = 0.8: potencijal je perturbacija pored tight-binding
       opsega, stvarajući mini-bands unutar glavnog opsega ±2t.
    6) Ground state (najniža energija iz band strukture + trap):
           H = V · diag(Λ) · V†
           |ψ_GS⟩ = V[:, 0]   (najniži eigenstate)
       Ground state je lokalizovan Wannier-like paket oko najbliže unit cell-e
       do j_target, modulisan Bloch periodičnošću.
    7) QFT dijagnostika (granica sa Fourier paradigmom):
           |k⟩ = (1/√N) Σ_j e^{2πi k j / N} |j⟩   (DFT/QFT matrica F_N)
           |ψ_k⟩ = F_N · |ψ_GS⟩         (momentum raspodela)
           P_k = |ψ_k|²                 (Bloch band-occupancy indikator)
    8) Born sempling u pozicionom bazisu:
           P(j) = |⟨j|ψ_GS⟩|²
       Maskovanje: num > prev_pick, num ∈ [i, i+32]; renormalize; rng.choice.

Dijagnostika po poziciji:
  • ⟨j⟩, σ_j                          — mean/std u j-prostoru (Wannier centar)
  • ⟨k⟩, σ_k                          — mean/std u momentum-prostoru (Bloch)
  • E_0                               — ground state energija (najniža band)
  • IPR_j = Σ_j |ψ(j)|⁴               — Inverse Participation Ratio (lokalizacija)
                                        — veliko IPR_j = lokalizovan,
                                          malo IPR_j = delokalizovan (Bloch talas)

(okruženje): Python 3.11.13, qiskit 1.4.4, macOS M1, seed = 39.
CSV = /data/loto7hh_4602_k32.csv
CSV u celini (S̄ kao info).
DeprecationWarning / FutureWarning se gase.
NQ = 6 qubit-a po poziciji (DIM = 64), reciklirani registar.
"""


from __future__ import annotations

import csv
import math
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass


# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4602_k32.csv")
N_NUMBERS = 7
N_MAX = 39

NQ = 6                              
DIM = 1 << NQ                       # 64
POS_RANGE = 33                      # Num_i ∈ [i, i + 32]

LATTICE_A = 4                       # unit cell size (period u j-sajtima)
V0_PERIOD = 0.8                     # jačina periodičnog kosinusnog potencijala
T_HOP = 1.0                         # tight-binding NN hopping (Chevalley)
ALPHA_LOC = 0.3                     # jačina lokalizacionog (harmonic) trap-a
SIGMA_LOC = 3.5                     # širina lokalizacije u j-sajtima


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def sort_rows_asc(H: np.ndarray) -> np.ndarray:
    return np.sort(H, axis=1)


# =========================
# Structural target (bez frekvencije)
# =========================
def target_num_structural(position_1based: int, prev_pick: int) -> float:
    denom = float(N_NUMBERS - position_1based + 2)
    return float(prev_pick) + float(N_MAX - prev_pick) / denom


def compute_j_target(position_1based: int, prev_pick: int) -> Tuple[int, float]:
    target = target_num_structural(position_1based, prev_pick)
    j = int(round(target)) - position_1based
    j = max(0, min(POS_RANGE - 1, j))
    return j, target


# =========================
# Tight-binding shift operatori T̂_±  (sa PERIODIČNIM graničnim uslovima
# — standardni kristalografski setup, Bravais torus)
# =========================
def shift_plus_periodic(n: int) -> np.ndarray:
    T = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        T[(j + 1) % n, j] = 1.0
    return T


T_PLUS = shift_plus_periodic(DIM)
T_MINUS = T_PLUS.conj().T

# H_kin = -t (T_+ + T_-), spektar: ε(k) = -2t cos(2π k / N), k = 0..N-1
H_KIN = -T_HOP * (T_PLUS + T_MINUS)


# =========================
# Periodični potencijal V_period(j) = V_0 · cos(2π j / a)
# =========================
def build_v_period(n: int, a: int, v0: float) -> np.ndarray:
    V = np.zeros((n, n), dtype=np.complex128)
    for j in range(n):
        V[j, j] = v0 * math.cos(2.0 * math.pi * float(j) / float(a))
    return V


V_PERIOD = build_v_period(DIM, LATTICE_A, V0_PERIOD)


# =========================
# Lokalizacioni harmonic potencijal
# =========================
def build_v_loc(j_target: int, alpha: float, sigma: float) -> np.ndarray:
    V = np.zeros((DIM, DIM), dtype=np.complex128)
    for j in range(DIM):
        V[j, j] = alpha * ((float(j) - float(j_target)) / sigma) ** 2
    return V


# =========================
# QFT matrica F_N (Bloch bazis dijagnostika)
# =========================
def build_qft_matrix(n: int) -> np.ndarray:
    F = np.zeros((n, n), dtype=np.complex128)
    for k in range(n):
        for j in range(n):
            F[k, j] = np.exp(-2j * math.pi * float(k * j) / float(n)) / math.sqrt(float(n))
    return F


F_QFT = build_qft_matrix(DIM)


# =========================
# Ground state Hamiltonijana
# =========================
def ground_state(H: np.ndarray) -> Tuple[np.ndarray, float]:
    Hh = (H + H.conj().T) / 2.0
    evals, evecs = np.linalg.eigh(Hh)
    return evecs[:, 0].astype(np.complex128), float(evals[0])


# =========================
# Predikcija jedne pozicije
# =========================
def bloch_pick_one_position(
    position_1based: int,
    prev_pick: int,
    rng: np.random.Generator,
) -> Tuple[int, int, float, float, float, float, float, float, float]:
    j_target, target = compute_j_target(position_1based, prev_pick)

    V_loc = build_v_loc(j_target, ALPHA_LOC, SIGMA_LOC)
    H = H_KIN + V_PERIOD + V_loc

    psi_gs, e_gs = ground_state(H)
    probs_j = np.abs(psi_gs) ** 2
    probs_j = np.clip(np.real(probs_j), 0.0, None)

    # Dijagnostika u j-prostoru
    js = np.arange(DIM, dtype=np.float64)
    mean_j = float(np.sum(js * probs_j))
    var_j = float(np.sum(((js - mean_j) ** 2) * probs_j))
    ipr_j = float(np.sum(probs_j ** 2))

    # Dijagnostika u Bloch / momentum bazisu (granica sa QFT paradigmom)
    psi_k = F_QFT @ psi_gs
    probs_k = np.abs(psi_k) ** 2
    probs_k = np.clip(np.real(probs_k), 0.0, None)
    ks = np.arange(DIM, dtype=np.float64)
    mean_k = float(np.sum(ks * probs_k))
    var_k = float(np.sum(((ks - mean_k) ** 2) * probs_k))

    mask = np.zeros(DIM, dtype=np.float64)
    for j in range(POS_RANGE):
        num = position_1based + j
        if 1 <= num <= N_MAX and num > prev_pick:
            mask[j] = 1.0

    probs_valid = probs_j * mask
    s = float(probs_valid.sum())
    if s < 1e-15:
        for j in range(POS_RANGE):
            num = position_1based + j
            if 1 <= num <= N_MAX and num > prev_pick:
                return (
                    num, j_target, target, mean_j, var_j,
                    mean_k, var_k, e_gs, ipr_j,
                )
        return (
            max(prev_pick + 1, position_1based),
            j_target, target, mean_j, var_j,
            mean_k, var_k, e_gs, ipr_j,
        )

    probs_valid /= s
    j_sampled = int(rng.choice(DIM, p=probs_valid))
    num = position_1based + j_sampled
    return (
        num, j_target, target, mean_j, var_j,
        mean_k, var_k, e_gs, ipr_j,
    )


# =========================
# Autoregresivni run
# =========================
def run_bloch_autoregressive() -> List[int]:
    rng = np.random.default_rng(SEED)
    picks: List[int] = []
    prev_pick = 0

    for i in range(1, N_NUMBERS + 1):
        (num, j_t, target, mean_j, var_j,
         mean_k, var_k, e_gs, ipr_j) = bloch_pick_one_position(
            i, prev_pick, rng
        )
        picks.append(int(num))
        print(
            f"  [pos {i}]  target={target:.3f}  j_target={j_t:2d}  "
            f"⟨j⟩={mean_j:5.2f}  σ_j={math.sqrt(max(var_j,0)):.3f}  "
            f"⟨k⟩={mean_k:5.2f}  σ_k={math.sqrt(max(var_k,0)):.3f}  "
            f"E_0={e_gs:+.3f}  IPR={ipr_j:.3f}  num={num:2d}"
        )
        prev_pick = int(num)

    return picks


# =========================
# Main
# =========================
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Nema CSV: {CSV_PATH}")

    H = load_rows(CSV_PATH)
    H_sorted = sort_rows_asc(H)
    S_bar = float(H_sorted.sum(axis=1).mean())

    n_cells = DIM // LATTICE_A

    print("=" * 88)
    print("Q43 Kristalografija — Bravais rešetka + Bloch bazis + band struktura")
    print("=" * 88)
    print(f"CSV:            {CSV_PATH}")
    print(f"Broj redova:    {H.shape[0]}")
    print(f"Qubit budget:   {NQ} po poziciji  (Hilbert dim={DIM})")
    print(f"Rešetka:        N = {DIM} sajtova,  a = {LATTICE_A} (unit cell),  "
          f"{n_cells} unit cells,  periodični BC (torus)")
    print(f"Hamiltonijan:   H = −t(T̂_+ + T̂_−) + V₀·cos(2πj/a)·I_j + α·((j−j_tgt)/σ)²·I_j")
    print(f"Parametri:      t = {T_HOP}   V₀ = {V0_PERIOD}   α = {ALPHA_LOC}   σ = {SIGMA_LOC}")
    print(f"Bloch bazis:    |k⟩ = (1/√N) Σ_j e^(2πi kj/N)|j⟩   (QFT)")
    print(f"Recip. rešetka: G_m = 2π m / a,  m ∈ ℤ")
    print(f"Stanje:         |ψ_GS⟩ = lowest-energy eigenstate H (band bottom)")
    print(f"Srednja suma S̄: {S_bar:.3f}  (CSV info, nije driver)")
    print(f"Seed:           {SEED}")
    print()
    print("Pokretanje kristalografije (Bloch GS + QFT dijagnostika) po pozicijama:")

    picks = run_bloch_autoregressive()

    n_odd = sum(1 for v in picks if v % 2 == 1)
    gaps = [picks[i + 1] - picks[i] for i in range(N_NUMBERS - 1)]

    print()
    print("=" * 88)
    print("REZULTAT Q43 (NEXT kombinacija)")
    print("=" * 88)
    print(f"Suma:   {sum(picks)}   (S̄={S_bar:.2f})")
    print(f"#odd:   {n_odd}")
    print(f"Gaps:   {gaps}")
    print(f"Predikcija NEXT: {picks}")


if __name__ == "__main__":
    main()



"""
========================================================================================
Q43 Kristalografija — Bravais rešetka + Bloch bazis + band struktura
========================================================================================
CSV:            /data/loto7hh_4602_k32.csv
Broj redova:    4602
Qubit budget:   6 po poziciji  (Hilbert dim=64)
Rešetka:        N = 64 sajtova,  a = 4 (unit cell),  16 unit cells,  periodični BC (torus)
Hamiltonijan:   H = −t(T̂_+ + T̂_−) + V₀·cos(2πj/a)·I_j + α·((j−j_tgt)/σ)²·I_j
Parametri:      t = 1.0   V₀ = 0.8   α = 0.3   σ = 3.5
Bloch bazis:    |k⟩ = (1/√N) Σ_j e^(2πi kj/N)|j⟩   (QFT)
Recip. rešetka: G_m = 2π m / a,  m ∈ ℤ
Stanje:         |ψ_GS⟩ = lowest-energy eigenstate H (band bottom)
Srednja suma S̄: 140.509  (CSV info, nije driver)
Seed:           39

Pokretanje kristalografije (Bloch GS + QFT dijagnostika) po pozicijama:
  [pos 1]  target=4.875  j_target= 4  ⟨j⟩= 4.14  σ_j=1.954  ⟨k⟩=28.06  σ_k=28.643  E_0=-1.999  IPR=0.174  num= 5
  [pos 2]  target=9.857  j_target= 8  ⟨j⟩= 8.00  σ_j=1.991  ⟨k⟩=27.80  σ_k=28.744  E_0=-2.003  IPR=0.171  num=11
  [pos 3]  target=15.667  j_target=13  ⟨j⟩=13.54  σ_j=1.466  ⟨k⟩=28.40  σ_k=28.348  E_0=-2.023  IPR=0.273  num=15
  [pos 4]  target=19.800  j_target=16  ⟨j⟩=16.00  σ_j=1.991  ⟨k⟩=27.80  σ_k=28.744  E_0=-2.003  IPR=0.171  num=18
  [pos 5]  target=23.250  j_target=18  ⟨j⟩=18.00  σ_j=1.238  ⟨k⟩=28.67  σ_k=28.155  E_0=-2.038  IPR=0.310  num=23
  [pos 6]  target=28.333  j_target=22  ⟨j⟩=22.00  σ_j=1.238  ⟨k⟩=28.67  σ_k=28.155  E_0=-2.038  IPR=0.310  num=29
  [pos 7]  target=34.000  j_target=27  ⟨j⟩=26.46  σ_j=1.466  ⟨k⟩=28.40  σ_k=28.348  E_0=-2.023  IPR=0.273  num=32

========================================================================================
REZULTAT Q43 (NEXT kombinacija)
========================================================================================
Suma:   133   (S̄=140.51)
#odd:   5
Gaps:   [6, 4, 3, 5, 6, 3]
Predikcija NEXT: [5, 11, x, y, z, 29, 32]
"""



"""
REZULTAT — Q43 Kristalografija / Bravais rešetka + Bloch bazis + band struktura
-------------------------------------------------------------------------------
(Popunjava se iz printa main()-a nakon pokretanja.)

Koncept:
  • Čisto kvantno: tight-binding Hamiltonijan na 1D periodičnoj
    rešetki, kosinusni periodični potencijal, harmonic lokalizacija, ground
    state dekompozicija preko eigh, QFT dijagnostika u Bloch bazisu, Born
    sempling. Bez klasičnog ML-a.
  • Kristalografska paradigma: Bravais rešetka (64 sajta, a = 4, 16 unit
    cells), Bloch teorema (eigenstanja = e^{ikj}·u_{n,k}(j)), recipročna
    rešetka (G_m = 2π m / a), band struktura (ε_n(k)).
  • Bloch bazis je tehnički Fourier bazis — QFT preslikava |j⟩ ↔ |k⟩.
    Stoga granica sa QFT paradigmom: QFT je korišćen isključivo kao DIAGNOSTIKA
    momentuma, a centralna fizika je u pozicionoj band-structure + lokalizaciji.
  • različito od Q28 (QFR — Fourier regression fitting) i Q30
    (QSP — polinomske funkcije preko signal-processing). Q43 koristi
    PERIODIČAN potencijal + Bloch opseg kao glavni strukturni element.
  • 64 sajta = prirodna diskretna rešetka; unit cell a = 4 daje
    16 ćelija; Mathieu-like eigenstanja su lokalizovani Wannier-tipski
    paketi sa band modulacijom — prirodna diskretna varijanta kristalne
    fizike.
  • NQ = 6 qubit-a, reciklirani 64-dim registar.
  • deterministički seed + fiksni a, V₀, t, α, σ + seeded Born.

Tehnike:
  • T̂_+ sa PERIODIČNIM graničnim uslovima (torus) — standardni kristalografski
    setup.
  • V_period = V_0 cos(2π j / a) — Mathieu potencijal (optička rešetka).
  • V_loc = α ((j − j_target)/σ)² — Gaussian-trap lokalizacija oko cilja.
  • eigh dijagonalizacija → ground state + svi opsezi.
  • F_QFT matrica = DFT transformacija u Bloch bazis.
  • Born sempling iz maskovane pozicione distribucije.

Dijagnostike:
  • ⟨j⟩, σ_j — Wannier centar i širina u j-prostoru.
  • ⟨k⟩, σ_k — Bloch momentum distribucija (kroz QFT).
  • E_0 — ground state energija (band bottom).
  • IPR_j = Σ_j |ψ(j)|⁴ — inverse participation ratio (lokalizacija vs
    delokalizacija: veliko = lokalizovano, malo = Bloch talas).
"""
