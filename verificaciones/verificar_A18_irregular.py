"""Verificacion independiente de A18 con conjuntos de nodos no cartesianos.

Desde la raiz del proyecto:
    python verificaciones/verificar_A18_irregular.py

No entrena redes ni certifica una seminorma continua. Comprueba la
interpolacion puntual, la arquitectura MLP y la evaluacion polinomica.
"""
import importlib.util
from pathlib import Path
import mpmath as mp

mp.mp.dps = 100
ruta = Path(__file__).resolve().parents[1] / 'codigos' / 'A18_comparadores_hermite.py'
spec = importlib.util.spec_from_file_location('a18_hermite', ruta)
a18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a18)

casos = [
    (['-0.71', '0.37'], ['-0.84', '-0.14', '0.48']),
    (['-0.91', '-0.88', '0.13'], ['-0.65', '0.52']),
    (['-0.6'], ['-0.9', '-0.1', '0.8']),
    (['-0.62', '-0.6199999999', '0.39'], ['-0.87', '-0.33', '0.55']),
]
for i, (xraw, yraw) in enumerate(casos, 1):
    xs, ys = [mp.mpf(s) for s in xraw], [mp.mpf(s) for s in yraw]
    zx, cx = a18.hermite_newton(xs)
    zy, cy = a18.hermite_newton(ys)
    red = a18.construir_mlp_hermite(zx, cx, zy, cy)
    def pol(x, y):
        return a18.B(zx, cx, x)[0] * a18.B(zy, cy, y)[0]
    err = max(abs(red(x, y) - pol(x, y))
              for x in [mp.mpf(k)/7 for k in (-6,-2,0,3,6)]
              for y in [mp.mpf(k)/9 for k in (-7,-1,2,5,8)])
    resid = max(abs(a18.B(zx,cx,x)[2]*a18.B(zy,cy,y)[0] +
                    a18.B(zx,cx,x)[0]*a18.B(zy,cy,y)[2] +
                    2*mp.pi**2*mp.sin(mp.pi*x)*mp.sin(mp.pi*y))
                for x in xs for y in ys)
    assert max(red.arquitectura[1:-1]) <= 8
    assert len(red.capas) == 8*max(len(xs), len(ys)) + 4
    assert err < mp.mpf('1e-75')
    assert resid < mp.mpf('1e-75')
    print(f'Caso {i} ({len(xs)}x{len(ys)} nodos): MLP={len(red.capas)} capas, '
          f'error_MLP={mp.nstr(err,4)}, residuo={mp.nstr(resid,4)}')
print('Todas las verificaciones independientes pasaron.')
