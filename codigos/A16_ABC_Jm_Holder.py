# -*- coding: utf-8 -*-
"""A-C revisados: pérdida de Poisson 2D con regularización Hölder DISCRETA.

J_m(h) = E_m(h) + a_m [Delta h]_{C^{0,1/2}(U)}^2 es el objeto teórico.
Este programa optimiza J_{m,P}(h) = E_m(h) + a_m R_P(h), donde
R_P = max_{(x,y) en P} |Delta h(x)-Delta h(y)|² / ||x-y||.
R_P es una COTA INFERIOR (nunca una certificación superior) de la
seminorma continua al cuadrado. No certifica minimización global.

Uso: python codigos/A16_ABC_Jm_Holder.py --profile smoke --loss both
     python codigos/A16_ABC_Jm_Holder.py --profile full --loss both --output resultados/ABC_Jm_nuevos
En Colab con GPU, copiar todo el proyecto y lanzar desde su raíz.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ALPHA = 0.5
A_CONST = 2 * math.sqrt(2) / (3 * math.pi)


def a_m(m):
    return A_CONST * m**(-0.75)


def exacta(x):
    return torch.sin(torch.pi*x[:, :1])*torch.sin(torch.pi*x[:, 1:2])


def fuente(x):
    return -2*torch.pi**2*exacta(x)


class RedFuerte(nn.Module):
    def __init__(self, ancho):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, ancho), nn.Tanh(),
                                 nn.Linear(ancho, ancho), nn.Tanh(),
                                 nn.Linear(ancho, 1))
    def forward(self,x):
        return (1-x[:, :1]**2)*(1-x[:, 1:2]**2)*self.net(x)


def lap_y_grad(model, points, create_graph=True):
    x = points.detach().requires_grad_(True)
    u = model(x)
    g = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    gxx = torch.autograd.grad(g[:, :1], x, torch.ones_like(g[:, :1]), create_graph=create_graph, retain_graph=True)[0][:, :1]
    gyy = torch.autograd.grad(g[:, 1:2], x, torch.ones_like(g[:, 1:2]), create_graph=create_graph)[0][:, 1:2]
    return u, g, gxx+gyy


def crear_pares(n, seed, device, dtype):
    """Pares aleatorios y de varias escalas, fijos por semilla y presupuesto."""
    rng = np.random.default_rng(int(seed))
    a = rng.uniform(-1., 1., size=(n,2))
    b = np.empty_like(a)
    n_lejos = n // 4
    b[:n_lejos] = rng.uniform(-1., 1., size=(n_lejos,2))
    radios = np.array([0.002, 0.02, 0.1, 0.4, 1.0])
    k = n-n_lejos
    radii = rng.choice(radios, size=k)*rng.uniform(0.5, 1.5, size=k)
    angles = rng.uniform(0, 2*np.pi, size=k)
    b[n_lejos:] = np.clip(a[n_lejos:] + radii[:,None]*np.stack((np.cos(angles),np.sin(angles)),axis=1),-1.,1.)
    close = np.linalg.norm(a-b,axis=1)<1e-8
    b[close] = np.clip(a[close]+0.01,-1.,1.)
    return (torch.as_tensor(a,device=device,dtype=dtype),
            torch.as_tensor(b,device=device,dtype=dtype))


def holder2_pares(model, pairs, batch=512, train=False):
    """Máximo discreto R_P; diferenciable casi por doquier respecto a theta."""
    x,y = pairs
    valores = []
    for start in range(0,len(x),batch):
        xx=x[start:start+batch]
        yy=y[start:start+batch]
        _,_,lap_x=lap_y_grad(model,xx,create_graph=train)
        _,_,lap_y=lap_y_grad(model,yy,create_graph=train)
        dist=torch.linalg.norm(xx-yy,dim=1,keepdim=True).clamp_min(1e-8)
        valores.append(((lap_x-lap_y)**2/dist).max())
    return torch.stack(valores).max()


def loss_j(model, points, pairs, regulariza, alpha_weight):
    _,_,lap=lap_y_grad(model,points,create_graph=True)
    e=((lap-fuente(points))**2).mean()
    reg=holder2_pares(model,pairs,batch=512,train=True) if regulariza else e.new_zeros(())
    return e + alpha_weight*reg,e,reg


def medir(model, puntos_train, puntos_test, pares_train, pares_eval, peso, max_eval=600):
    model.eval()
    def evaluacion_pde(puntos):
        suma=0.; norm_e=0.; norm_u=0.; norm_g_e=0.; norm_g_u=0.
        for start in range(0,len(puntos),max_eval):
            x=puntos[start:start+max_eval]
            h,dh,lap=lap_y_grad(model,x,create_graph=False)
            e=exacta(x)
            grad_ex=torch.cat((torch.pi*torch.cos(torch.pi*x[:,:1])*torch.sin(torch.pi*x[:,1:2]),
                               torch.pi*torch.sin(torch.pi*x[:,:1])*torch.cos(torch.pi*x[:,1:2])),dim=1)
            suma+=float(((lap-fuente(x))**2).sum().detach())
            norm_e+=float(((h-e)**2).sum().detach())
            norm_u+=float((e**2).sum().detach())
            norm_g_e+=float(((dh-grad_ex)**2).sum().detach())
            norm_g_u+=float((grad_ex**2).sum().detach())
        return (suma/len(puntos),math.sqrt(norm_e/norm_u),
                math.sqrt((norm_e+norm_g_e)/(norm_u+norm_g_u)))
    er_tr,_,_=evaluacion_pde(puntos_train)
    er_te,e_l2,e_h1=evaluacion_pde(puntos_test)
    # Gradientes de theta desactivados en el resultado numérico; seguimos
    # utilizando diferenciación respecto de las coordenadas para Delta.
    with torch.enable_grad():
        r_train=float(holder2_pares(model,pares_train,batch=256,train=False).detach())
        r1200=float(holder2_pares(model,(pares_eval[0][:1200],pares_eval[1][:1200]),batch=256,train=False).detach())
        r4800=float(holder2_pares(model,pares_eval,batch=256,train=False).detach())
    return dict(E_train=er_tr,E_test=er_te,L2_rel=e_l2,H1_rel=e_h1,
                R_P_train=r_train,J_train_P=er_tr+peso*r_train,
                holder_eval_1200=math.sqrt(r1200),holder_eval_4800=math.sqrt(r4800),
                a_m=peso, E_sobre_a=er_tr/peso, J_sobre_a=(er_tr+peso*r_train)/peso)


def run_one(n, width, seed, method, config, out, device, dtype, protocol, warm=False):
    torch.manual_seed(seed)
    model=RedFuerte(width).to(device=device,dtype=dtype)
    train=torch.as_tensor(np.random.default_rng(seed+5).uniform(-1,1,size=(n,2)),device=device,dtype=dtype)
    test=torch.as_tensor(np.random.default_rng(seed+500).uniform(-1,1,size=(config['n_test'],2)),device=device,dtype=dtype)
    pairs=crear_pares(config['train_pairs'],seed+900,device,dtype)
    pairs_eval=crear_pares(4800,seed+70000,device,dtype)
    # Las mismas condiciones iniciales y muestras para residual y J_{m,P}.
    reg=(method=='JmP')
    weight=a_m(n)
    rows=[]; start=time.perf_counter()
    opt=torch.optim.Adam(model.parameters(),lr=1.e-3)
    stages=([config['adam_short'],config['adam_long']] if warm else [config['adam_long']])
    last=0
    for total in stages:
        for step in range(last,total):
            idx=torch.randperm(n,device=device)[:min(config['adam_batch'],n)]
            # Reemplazo estocástico de P durante Adam: optimización por minilotes.
            a,b=pairs
            pidx=torch.randint(len(a),(min(config['adam_pairs'],len(a)),),device=device)
            opt.zero_grad(set_to_none=True)
            obj,_,_=loss_j(model,train[idx],(a[pidx],b[pidx]),reg,weight)
            obj.backward()
            opt.step()
        last=total
        if warm and total==config['adam_short']:
            m=medir(model,train,test,pairs,pairs_eval,weight)
            rows.append(dict(protocolo=protocol+'-corto',perdida_entrenada=method,m=n,ancho=width,
                             seed=seed,adam_pasos=last,lbfgs_iter=0,segundos=time.perf_counter()-start,**m))
    # L-BFGS: objetivo DETERMINISTA sobre TODO el conjunto interior y P fijo.
    # Requiere más memoria; en Colab A100 se usa el perfil completo.
    if config['lbfgs']:
        opt2=torch.optim.LBFGS(model.parameters(),lr=1.0,max_iter=config['lbfgs'],
                               history_size=30,line_search_fn='strong_wolfe')
        def closure():
            opt2.zero_grad(set_to_none=True)
            obj,_,_=loss_j(model,train,pairs,reg,weight)
            obj.backward()
            return obj
        opt2.step(closure)
        it=int(opt2.state[opt2._params[0]].get('n_iter',0))
    else:
        it=0
    result=medir(model,train,test,pairs,pairs_eval,weight)
    rows.append(dict(protocolo=protocol+('-largo' if warm else ''),perdida_entrenada=method,
                     m=n,ancho=width,seed=seed,adam_pasos=last,lbfgs_iter=it,
                     segundos=time.perf_counter()-start,**result))
    model_path=out / f'{protocol}_{method}_m{n}_w{width}_s{seed}.pt'
    torch.save({'state_dict':model.state_dict(),'config':config,'n':n,'width':width,
                'seed':seed,'method':method,'a_m':weight,
                'nota':'JmP contiene cota inferior muestral; no seminorma continua'},model_path)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',choices=['smoke','pilot','full'],default='smoke')
    ap.add_argument('--loss',choices=['residual','regularized','both'],default='both')
    ap.add_argument('--output',default='resultados/ABC_Jm_nuevos')
    args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype=torch.float64
    torch.set_num_threads(2)
    profiles={
      'smoke':dict(adam_short=1,adam_long=2,lbfgs=0,n_test=20,train_pairs=16,adam_pairs=8,adam_batch=24),
      'pilot':dict(adam_short=100,adam_long=400,lbfgs=25,n_test=1024,train_pairs=256,adam_pairs=96,adam_batch=256),
      'full':dict(adam_short=900,adam_long=4000,lbfgs=120,n_test=4096,train_pairs=1024,adam_pairs=256,adam_batch=512),
    }
    cfg=profiles[args.profile]
    if args.profile=='smoke':
        settings=[('A',16,8,True),('B',16,12,False),('C',25,16,False)]
        seeds=[7]
    else:
        # B64 es el mismo punto de partida de C(1521,64): se calcula UNA vez
        settings=[('A',1521,32,True),('A',1521,64,True),('B',1521,32,False),('B_C',1521,64,False),
                  ('B',1521,128,False),('C',5476,128,False),('C',11881,192,False)]
        seeds=[7,17,27]
    methods= ['Em','JmP'] if args.loss=='both' else (['Em'] if args.loss=='residual' else ['JmP'])
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for protocol,n,width,warm in settings:
        for seed in seeds:
            for method in methods:
                print(f'{protocol} m={n} W={width} seed={seed} loss={method}',flush=True)
                rows=run_one(n,width,seed,method,cfg,out,device,dtype,protocol,warm=warm)
                allrows+=rows
                # Guardado incremental para reanudar extracción de resultados.
                with open(out/'resultados_ABC_JmP.csv','w',newline='',encoding='utf8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(allrows[0].keys()));writer.writeheader();writer.writerows(allrows)
    (out/'protocolo.json').write_text(json.dumps(dict(profile=args.profile,loss=args.loss,config=cfg,
        device=str(device),torch=torch.__version__,seeds=seeds,settings=settings,
        dtype=str(dtype),continuous_holder_certified=False,
        holder_pairs_train=cfg['train_pairs'],holder_pairs_eval=4800,
        comparison='mismas semillas, datos, inicialización, pasos y arquitectura por método'),
        indent=2,ensure_ascii=False),encoding='utf8')
    print('Finalizado; filas:',len(allrows),'salida:',out)

if __name__=='__main__':
    main()
