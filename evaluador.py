"""
Evaluador de startup - Nivel 5 JDL
Funcion completa que toma 5 variables de entrada y entrega:
  - Score numerico (FIS)
  - Clasificacion lingüistica (BAJO/MEDIO/ALTO/CRITICO)
  - Color del semaforo
  - Recomendacion accionable (matriz de decision)
  - Palancas sugeridas (que mover para mejorar)
"""

import numpy as np
from skfuzzy import control as ctrl
from fis_quiebra import construir_fis, clasificar

# ============================================================
# MATRIZ DE DECISION (Nivel 5 JDL)
# ============================================================
MATRIZ_DECISION = {
    'BAJO': {
        'color': 'verde',
        'hex': '#2ca02c',
        'accion': 'INVERTIR',
        'descripcion': 'Startup saludable. Recomendado proceder con term sheet.',
        'urgencia': 'Bajo - proceso estandar de due diligence',
    },
    'MEDIO': {
        'color': 'amarillo',
        'hex': '#f0c419',
        'accion': 'DUE DILIGENCE PROFUNDO',
        'descripcion': 'Senales mixtas. Requiere revision detallada antes de decidir.',
        'urgencia': 'Medio - revision adicional en 2 semanas',
    },
    'ALTO': {
        'color': 'naranja',
        'hex': '#ff7f0e',
        'accion': 'NEGOCIAR TERMINOS',
        'descripcion': 'Riesgo alto. Considerar valuacion menor, hitos o tranches.',
        'urgencia': 'Alto - decision condicional con metas trimestrales',
    },
    'CRITICO': {
        'color': 'rojo',
        'hex': '#d62728',
        'accion': 'RECHAZAR',
        'descripcion': 'Riesgo critico de quiebra. No invertir en esta ronda.',
        'urgencia': 'Critico - reevaluar solo si cambian fundamentales',
    },
}


def palancas_mejora(entradas, score):
    """Sugiere que palancas mover para mejorar el score."""
    if score < 30:
        return ['Mantener trayectoria actual']

    sugerencias = []
    if entradas['runway'] < 9:
        sugerencias.append(
            f"Extender runway (actual: {entradas['runway']:.0f} meses). "
            "Levantar puente o reducir burn."
        )
    if entradas['burn_ratio'] > 2.0:
        sugerencias.append(
            f"Reducir burn ratio (actual: {entradas['burn_ratio']:.1f}x). "
            "Recortar costos o acelerar ingresos."
        )
    if entradas['growth_mom'] < 5:
        sugerencias.append(
            f"Acelerar crecimiento (actual: {entradas['growth_mom']:+.0f}% MoM). "
            "Revisar product-market fit y canales."
        )
    if entradas['gross_margin'] < 30:
        sugerencias.append(
            f"Mejorar margen bruto (actual: {entradas['gross_margin']:.0f}%). "
            "Optimizar COGS o subir precios."
        )
    if entradas['team_exp'] < 4:
        sugerencias.append(
            f"Fortalecer equipo (actual: {entradas['team_exp']:.1f}/10). "
            "Sumar advisors o senior hires."
        )
    return sugerencias if sugerencias else ['Ninguna palanca critica identificada']


def evaluar_startup(runway, burn_ratio, growth_mom, team_exp, gross_margin,
                    nombre='Startup', verbose=True):
    """
    Funcion principal del Nivel 5 JDL.
    Recibe 5 metricas y retorna evaluacion completa.
    """
    sistema, _ = construir_fis()
    sim = ctrl.ControlSystemSimulation(sistema)

    entradas = {
        'runway': runway, 'burn_ratio': burn_ratio,
        'growth_mom': growth_mom, 'team_exp': team_exp,
        'gross_margin': gross_margin,
    }

    for k, v in entradas.items():
        sim.input[k] = float(v)
    sim.compute()
    score = sim.output['quiebra_score']

    clase = clasificar(score)
    decision = MATRIZ_DECISION[clase]
    palancas = palancas_mejora(entradas, score)

    resultado = {
        'nombre': nombre,
        'entradas': entradas,
        'score': score,
        'clasificacion': clase,
        'color': decision['color'],
        'accion': decision['accion'],
        'descripcion': decision['descripcion'],
        'urgencia': decision['urgencia'],
        'palancas': palancas,
    }

    if verbose:
        print('=' * 65)
        print(f'Evaluacion JDL: {nombre}')
        print('=' * 65)
        print(f'  Runway:       {runway:>5} meses')
        print(f'  Burn ratio:   {burn_ratio:>5.1f}x  (gasto/ingreso)')
        print(f'  Crec. MoM:    {growth_mom:>5.0f}%')
        print(f'  Experiencia:  {team_exp:>5.1f}/10')
        print(f'  Margen bruto: {gross_margin:>5.0f}%')
        print('-' * 65)
        print(f'  SCORE DE QUIEBRA:  {score:>5.1f}/100  ({clase})')
        print(f'  SEMAFORO:          {decision["color"].upper()}')
        print(f'  DECISION:          {decision["accion"]}')
        print(f'  {decision["descripcion"]}')
        print()
        print('  Palancas sugeridas para mejorar:')
        for p in palancas:
            print(f'    - {p}')
        print('=' * 65 + '\n')

    return resultado


if __name__ == '__main__':
    print('Casos de demostracion del Nivel 5 JDL\n')

    evaluar_startup(
        runway=30, burn_ratio=0.6, growth_mom=20, team_exp=9, gross_margin=70,
        nombre='Caso A: Startup saludable (Serie A)',
    )

    evaluar_startup(
        runway=10, burn_ratio=1.7, growth_mom=8, team_exp=5, gross_margin=35,
        nombre='Caso B: Startup en zona gris (revisar)',
    )

    evaluar_startup(
        runway=4, burn_ratio=2.8, growth_mom=2, team_exp=4, gross_margin=15,
        nombre='Caso C: Startup en riesgo critico',
    )

    evaluar_startup(
        runway=15, burn_ratio=2.0, growth_mom=12, team_exp=7, gross_margin=45,
        nombre='Caso D: Negociar terminos (riesgo alto pero salvable)',
    )
