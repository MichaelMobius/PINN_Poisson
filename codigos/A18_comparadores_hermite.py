"""Comparadores de Hermite para Poisson 2D: ejemplo corto y reproducible.

Ejecutar desde la raiz del proyecto:
    python codigos/A18_comparadores_hermite.py

Requiere solamente mpmath. No realiza entrenamiento ni usa GPU.
Ademas instancia las matrices W_l y los sesgos b_l de una MLP genuina
con activacion sigma(t)=t**2 en todas las capas ocultas, y una salida lineal.
No se entrena la red: los pesos se construyen directamente.
"""

import csv
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 85


def a(t):
    """sin(pi*t)/(1-t**2); los nodos de ejemplo no son extremos."""
    return mp.sin(mp.pi * t) / (1 - t * t)


def hermite_newton(nodos):
    """Datos de a, a', a'', a''' por nodo; diferencias confluyentes."""
    z = [t for t in nodos for _ in range(4)]
    N = len(z)
    D = [[mp.mpf('0')] * (i + 1) for i in range(N)]
    for i in range(N):
        D[i][0] = a(z[i])
        for j in range(1, i + 1):
            if z[i] == z[i - j]:
                D[i][j] = mp.diff(a, z[i], j) / mp.factorial(j)
            else:
                D[i][j] = (D[i][j - 1] - D[i - 1][j - 1]) / (z[i] - z[i - j])
    return z, [D[i][i] for i in range(N)]


def eval_newton(z, c, t):
    """Valor y las primeras tres derivadas por Horner generalizado."""
    p0, p1, p2, p3 = c[-1], mp.mpf('0'), mp.mpf('0'), mp.mpf('0')
    for k in range(len(c) - 2, -1, -1):
        v = t - z[k]
        p3, p2, p1, p0 = (
            p3 * v + 3 * p2,
            p2 * v + 2 * p1,
            p1 * v + p0,
            p0 * v + c[k],
        )
    return p0, p1, p2, p3


def B(z, c, t):
    """b=(1-t^2)q; devuelve b, b', b'', b''' ."""
    q, q1, q2, q3 = eval_newton(z, c, t)
    v = 1 - t * t
    return (v * q,
            -2*t*q + v*q1,
            -2*q - 4*t*q1 + v*q2,
            -6*q1 - 6*t*q2 + v*q3)


def mul_sq(u, v):
    """Multiplicacion usando solamente sumas y sigma(s)=s^2."""
    return ((u+v)**2 - (u-v)**2) / 4


def eval_newton_square(z, c, t):
    s = c[-1]
    for k in range(len(c) - 2, -1, -1):
        s = mul_sq(s, t-z[k]) + c[k]
    return s



# Una MLP de verdad: todas las capas ocultas aplican sigma(t)=t^2.
# La ultima capa es afin. Las matrices se construyen, no se entrenan.
class MLP_Cuadratica:
    def __init__(self):
        self.capas = []  # pares (W_l, b_l), con W_l matriz explicita
        self.salida = None

    def capa(self, filas):
        # Cada fila es (lista_de_pesos, sesgo).
        n_in = len(filas[0][0])
        assert all(len(w) == n_in for w, _ in filas)
        self.capas.append(([[mp.mpf(a) for a in w] for w, _ in filas],
                           [mp.mpf(b) for _, b in filas]))

    def __call__(self, x, y):
        z = [mp.mpf(x), mp.mpf(y)]
        for W, b in self.capas:
            z = [(sum((a*v for a, v in zip(fila, z)), mp.mpf('0')) + bi)**2
                 for fila, bi in zip(W, b)]
        W, b = self.salida
        return sum((a*v for a, v in zip(W, z)), b)

    @property
    def arquitectura(self):
        return [2] + [len(b) for _, b in self.capas] + [1]


def af(n, terms=(), bias=0):
    """Forma afin explicita en R^n: (fila de W, componente de b)."""
    w = [mp.mpf('0')] * n
    for i, a in terms:
        w[i] += a
    return w, mp.mpf(bias)


def suma(F, G, a=1, b=1, c=0):
    """Devuelve a*F+b*G+c, manteniendo la representacion matricial."""
    assert len(F[0]) == len(G[0])
    return [a*x+b*y for x, y in zip(F[0], G[0])], a*F[1]+b*G[1]+c


def shift(F, c):
    return F[0][:], F[1] + c


def enc(F):
    """Codifica r en las dos neuronas (r+1)^2, (r-1)^2."""
    return [shift(F, 1), shift(F, -1)]


def dec(n, i):
    """Decodifica el registro i: r=(z[2i]-z[2i+1])/4."""
    return af(n, [(2*i, mp.mpf('0.25')),
                  (2*i+1, -mp.mpf('0.25'))])


def construir_mlp_hermite(zx, cx, zy, cy):
    """MLP cuadratica estrictamente por capas para psi*qx(x)*qy(y).

    Cada capa guarda registros con E(r)=((r+1)^2,(r-1)^2).
    Un paso Horner usa DOS capas ocultas y ancho maximo 8.
    No hay productos ni funciones especiales fuera de las capas.
    """
    red = MLP_Cuadratica()
    # Registros iniciales: x, y, qx=cx[-1], qy=cy[-1].
    estados = [af(2, [(0, 1)]), af(2, [(1, 1)]),
               af(2, bias=cx[-1]), af(2, bias=cy[-1])]
    red.capa([f for F in estados for f in enc(F)])

    # Horner paralelo, permitiendo diferentes nodos por coordenada.
    for j in range(max(len(cx), len(cy)) - 1):
        x, y, qx, qy = [dec(8, i) for i in range(4)]
        ix, iy = len(cx)-2-j, len(cy)-2-j
        kx = shift(x, -zx[ix]) if ix >= 0 else af(8)
        ky = shift(y, -zy[iy]) if iy >= 0 else af(8)
        A = enc(x) + enc(y)
        A += ([suma(qx, kx), suma(qx, kx, b=-1)] if ix >= 0 else enc(qx))
        A += ([suma(qy, ky), suma(qy, ky, b=-1)] if iy >= 0 else enc(qy))
        red.capa(A)
        # A4-A5 = 4*qx*(x-zx[ix]); si inactivo codifica el qx previo.
        qx2 = shift(af(8, [(4, mp.mpf('0.25')),
                               (5, -mp.mpf('0.25'))]), cx[ix] if ix >= 0 else 0)
        qy2 = shift(af(8, [(6, mp.mpf('0.25')),
                               (7, -mp.mpf('0.25'))]), cy[iy] if iy >= 0 else 0)
        red.capa(enc(dec(8, 0)) + enc(dec(8, 1)) + enc(qx2) + enc(qy2))

    # Factor polinomico de frontera: (1-x^2)(1-y^2).
    x, y, qx, qy = [dec(8, i) for i in range(4)]
    red.capa([x, y] + enc(qx) + enc(qy))  # x^2, y^2, E(qx), E(qy)
    red.capa(enc(af(6, [(0, -1)], bias=1)) +
             enc(af(6, [(1, -1)], bias=1)) +
             enc(dec(6, 1)) + enc(dec(6, 2)))

    # Multiplicar (1-x^2)*qx y (1-y^2)*qy en paralelo.
    u, v, qx, qy = [dec(8, i) for i in range(4)]
    red.capa([suma(u, qx), suma(u, qx, b=-1),
              suma(v, qy), suma(v, qy, b=-1)])
    prodx = af(4, [(0, mp.mpf('0.25')), (1, -mp.mpf('0.25'))])
    prody = af(4, [(2, mp.mpf('0.25')), (3, -mp.mpf('0.25'))])
    red.capa(enc(prodx) + enc(prody))
    t1, t2 = dec(4, 0), dec(4, 1)
    red.capa([suma(t1, t2), suma(t1, t2, b=-1)])
    red.salida = ([mp.mpf('0.25'), -mp.mpf('0.25')], mp.mpf('0'))
    assert max(red.arquitectura[1:-1]) <= 8
    return red


def guardar_mlp_json(red, ruta):
    """Exporta literalmente todas las matrices y sesgos en decimal."""
    d = {'sigma': 't**2', 'capa_final': 'lineal', 'mp_dps': mp.mp.dps,
         'arquitectura': red.arquitectura,
         'capas': [{'W': [[mp.nstr(v, mp.mp.dps) for v in fila] for fila in W],
                    'b': [mp.nstr(v, mp.mp.dps) for v in b]} for W, b in red.capas],
         'W_salida': [mp.nstr(v, mp.mp.dps) for v in red.salida[0]],
         'b_salida': mp.nstr(red.salida[1], mp.mp.dps)}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')


def leer_mlp_json(ruta):
    d = json.loads(ruta.read_text(encoding='utf-8'))
    red = MLP_Cuadratica()
    for capa in d['capas']:
        red.capa(list(zip(capa['W'], capa['b'])))
    red.salida = ([mp.mpf(v) for v in d['W_salida']],
                  mp.mpf(d['b_salida']))
    assert red.arquitectura == d['arquitectura']
    return red


def prueba(n, puntos_malla=21):
    # Nodos racionales dentro del cuadrado; m=n^2 puntos interiores.
    nodos = [mp.mpf(-4)/5 + mp.mpf(8)*i/(5*(n-1)) for i in range(n)]
    z, c = hermite_newton(nodos)
    mlp = construir_mlp_hermite(z, c, z, c)
    max_res = mp.mpf('0')
    for x in nodos:
        for y in nodos:
            bx, by = B(z,c,x), B(z,c,y)
            lap = bx[2]*by[0] + bx[0]*by[2]
            f = -2*mp.pi**2*mp.sin(mp.pi*x)*mp.sin(mp.pi*y)
            max_res = max(max_res, abs(lap - f))

    numer, denom, max_grad = mp.mpf('0'), mp.mpf('0'), mp.mpf('0')
    for i in range(puntos_malla):
        x = -1 + mp.mpf(2)*i/(puntos_malla-1)
        bx = B(z,c,x)
        for j in range(puntos_malla):
            y = -1 + mp.mpf(2)*j/(puntos_malla-1)
            by = B(z,c,y)
            pred = bx[0]*by[0]
            exacta = mp.sin(mp.pi*x)*mp.sin(mp.pi*y)
            numer += (pred-exacta)**2
            denom += exacta**2
            gx = bx[3]*by[0] + bx[1]*by[2]
            gy = bx[2]*by[1] + bx[0]*by[3]
            max_grad = max(max_grad, mp.sqrt(gx**2+gy**2))

    # Evaluacion directa de la MLP completa contra el polinomio.
    max_error_mlp = mp.mpf('0')
    for i in range(5):
        x = -mp.mpf('0.9') + mp.mpf('1.8')*i/4
        for j in range(5):
            y = -mp.mpf('0.9') + mp.mpf('1.8')*j/4
            pol = B(z, c, x)[0]*B(z, c, y)[0]
            max_error_mlp = max(max_error_mlp, abs(mlp(x, y) - pol))
    assert max_error_mlp < mp.mpf('1e-45')

    # Test de serializacion de los pesos en el ejemplo mas pequeno.
    if n == 2:
        ruta = Path('resultados/ejemplo_hermite_mlp_n2.json')
        guardar_mlp_json(mlp, ruta)
        red_leida = leer_mlp_json(ruta)
        assert abs(red_leida(mp.mpf('0.125'), mp.mpf('-0.375')) -
                   mlp(mp.mpf('0.125'), mp.mpf('-0.375'))) < mp.mpf('1e-55')

    t0 = mp.mpf('0.127')
    q_directo = eval_newton(z,c,t0)[0]
    q_cuadrados = eval_newton_square(z,c,t0)
    assert abs(q_directo-q_cuadrados) < mp.mpf('1e-45')
    return {'n':n, 'm':n*n, 'grado_q':len(c)-1,
            'residuo_nodal_max':mp.nstr(max_res,8),
            'error_relativo_L2_malla':mp.nstr(mp.sqrt(numer/denom),8),
            'max_grad_laplaciano_malla':mp.nstr(max_grad,8),
            'diferencia_circuito_cuadratico':mp.nstr(abs(q_directo-q_cuadrados),5),
            'capas_ocultas_mlp':len(mlp.capas),
            'ancho_maximo_mlp':max(mlp.arquitectura[1:-1]),
            'max_diferencia_mlp_polinomio':mp.nstr(max_error_mlp,6)}


def main():
    filas = [prueba(n) for n in (2,3,4)]
    dest = Path('resultados/ejemplo_hermite_comparadores.csv')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0]))
        writer.writeheader()
        writer.writerows(filas)
    for fila in filas:
        print(fila)
    print('Guardado:', dest)
    print('Pesos explicitos n=2:', 'resultados/ejemplo_hermite_mlp_n2.json')


if __name__ == '__main__':
    main()
