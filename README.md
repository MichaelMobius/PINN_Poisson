# PINN para la ecuación de Poisson — código experimental

Implementaciones en Python de los experimentos de Poisson 1D y 2D, la comparación PINN/FEM/FVM, la regularización Hölder discreta y los comparadores de Hermite.

## Instalación

Requiere Python 3.10 o superior. Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
```

PyTorch utiliza la GPU si está disponible. Para instalar una versión de PyTorch con CUDA, consultar las instrucciones de instalación de PyTorch para el sistema correspondiente.

## Ejecuciones de comprobación

```bash
python codigos/A18_comparadores_hermite.py
python codigos/A16_ABC_Jm_Holder.py --profile smoke --loss both
python verificaciones/verificar_A18_irregular.py
python reproducibilidad/verificar_resultados_ABC.py
python codigos/A17_diagnostico_sesgo_JmP_exacto.py
```

A16 admite `--profile smoke`, `pilot` y `full`; para reproducir el ensayo ampliado se usa `--profile full --loss both`. Los perfiles completos y otras ejecuciones de entrenamiento pueden requerir GPU y tiempos de cómputo elevados.

Los programas A01–A12 se ejecutan desde la raíz con `python codigos/NOMBRE.py`. A13–A15 son extractos dependientes de variables definidas en el cuaderno `cuadernos/Experimentos_PINN_Poisson.ipynb`; ejecutar sus secciones dentro de ese cuaderno, no como archivos Python independientes.

Para los protocolos originales A–C, ejecutar `cuadernos/Protocolos_ABC.ipynb` desde la raíz del repositorio. Por defecto usa el perfil `rapido`; cambiar la variable de entorno `PINN_JURADO_PERFIL` a `completo` para el presupuesto documentado. Los cuatro checkpoints de inicio del protocolo A están en `reproducibilidad/checkpoints_origen_A/`. El perfil completo está diseñado para GPU.

## Archivos

- `codigos/`: implementaciones A01–A18. A08 corresponde al objetivo con peso basado en el número total de puntos interiores.
- `cuadernos/`: experimentos integrados y protocolos A–C.
- `resultados/`: cuatro CSV de referencia utilizados por A17 y las verificaciones.
- `reproducibilidad/checkpoints_origen_A/`: cuatro estados iniciales para prolongar el protocolo A.
- `verificaciones/`: prueba de la construcción de Hermite.

Las nuevas ejecuciones guardan sus resultados en las carpetas de salida definidas en cada programa. El archivo `resultados/ABC_Jm_full/resultados_ABC_JmP.csv` contiene los resultados de referencia necesarios para A17, no nuevos cálculos.

## Alcance de la reproducción

Los catorce estados finales originales de A–C no se incluyen: se proporciona el código y los cuatro estados de partida de A para volver a ejecutar el entrenamiento. Los resultados tabulados del experimento oscilatorio no tienen una trazabilidad ejecutable completa desde el código disponible. La ejecución del programa A08 genera resultados nuevos, que deben distinguirse de los publicados en la tesis.
