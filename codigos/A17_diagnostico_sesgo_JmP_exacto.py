# -*- coding: utf-8 -*-
"""Diagnóstico exacto de J_{m,P}(u*) sobre los pares fijos de A16.

No entrena una red; aplica el mismo generador PCG64 de pares a
Delta u* = -2*pi^2 sin(pi*x) sin(pi*y), con E_m(u*)=0.
Los resultados archivados de A16 solo se usan como contraste, sin
re-ejecutar ni alterar ningún entrenamiento.

Uso desde Tesis_PUCP/:
  python codigos/A17_diagnostico_sesgo_JmP_exacto.py
"""
from pathlib import Path
import csv
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'resultados' / 'ABC_Jm_full' / 'resultados_ABC_JmP.csv'
OUTPUT = ROOT / 'resultados' / 'diagnostico_JmP_solucion_exacta.csv'
A_CONST = 2 * math.sqrt(2) / (3 * math.pi)


def crear_pares_np(n, seed):
    """Traducción sin cambios de la lógica de A16.crear_pares.

    Deja intactos el orden de llamadas del generador y las escalas.
    """
    rng = np.random.default_rng(int(seed))
    a = rng.uniform(-1., 1., size=(n, 2))
    b = np.empty_like(a)
    n_lejos = n // 4
    b[:n_lejos] = rng.uniform(-1., 1., size=(n_lejos, 2))
    radios = np.array([0.002, 0.02, 0.1, 0.4, 1.0])
    k = n - n_lejos
    radii = rng.choice(radios, size=k) * rng.uniform(0.5, 1.5, size=k)
    angles = rng.uniform(0, 2*np.pi, size=k)
    b[n_lejos:] = np.clip(a[n_lejos:] + radii[:, None] *
                          np.stack((np.cos(angles), np.sin(angles)), axis=1), -1., 1.)
    close = np.linalg.norm(a-b, axis=1) < 1e-8
    b[close] = np.clip(a[close] + 0.01, -1., 1.)
    return a, b


def lap_exacta(z):
    return -2 * np.pi**2 * np.sin(np.pi * z[:, 0]) * np.sin(np.pi * z[:, 1])


def rp_exacta(seed):
    a, b = crear_pares_np(1024, seed + 900)
    dist = np.linalg.norm(a-b, axis=1).clip(1e-8)
    return float(np.max((lap_exacta(a) - lap_exacta(b))**2 / dist))


def main():
    records=[]
    with RESULTS.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['perdida_entrenada'] != 'JmP' or r['protocolo'] not in ('B_C', 'C'):
                continue
            # para B_C, cada m=1521/W64; para C m=5476 y 11881
            m, seed = int(r['m']), int(r['seed'])
            rp = rp_exacta(seed)
            am = A_CONST * m**(-.75)
            j_exact = am * rp
            j_train = float(r['J_train_P'])
            records.append(dict(protocolo=r['protocolo'],m=m,ancho=int(r['ancho']),
                                seed=seed,n_pares=1024, R_P_u_exacta=rp,
                                a_m=am,E_m_u_exacta=0.,J_mP_u_exacta=j_exact,
                                J_mP_red_entrenada=j_train,
                                diferencia_J_exacto_menos_entrenada=j_exact-j_train,
                                L2_red_entrenada=float(r['L2_rel']),
                                fuente='A16.crear_pares con seed+900; A16 CSV original'))
    OUTPUT.parent.mkdir(exist_ok=True,parents=True)
    fields=list(records[0].keys())
    with OUTPUT.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader();w.writerows(records)
    for r in records:
        print(r['m'],r['seed'],f"R={r['R_P_u_exacta']:.6f}",
              f"J_exact={r['J_mP_u_exacta']:.6f}",
              f"J_trained={r['J_mP_red_entrenada']:.6f}",
              f"diff={r['diferencia_J_exacto_menos_entrenada']:.6f}")


if __name__=='__main__': main()
