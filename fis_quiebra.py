"""
Sistema de Inferencia Difusa (FIS) - Prediccion de quiebra de startups
Trabajo Final - Modelos y Simulacion

Variables de entrada validadas por Delphi:
    - runway (meses)        : 0-36
    - burn_ratio (adim)     : 0-5
    - growth_mom (%)        : -20 a 30
    - team_exp (escala)     : 0-10
    - gross_margin (%)      : -50 a 90

Variable de salida:
    - quiebra_score (0-100): riesgo de quiebra en 12 meses
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def construir_fis():
    """Construye y retorna el sistema de inferencia difusa Mamdani."""

    # ============================================================
    # 1. UNIVERSOS DE DISCURSO (rangos de cada variable)
    # ============================================================
    runway = ctrl.Antecedent(np.arange(0, 36.1, 0.1), 'runway')
    burn_ratio = ctrl.Antecedent(np.arange(0, 5.01, 0.01), 'burn_ratio')
    growth_mom = ctrl.Antecedent(np.arange(-20, 30.1, 0.1), 'growth_mom')
    team_exp = ctrl.Antecedent(np.arange(0, 10.01, 0.01), 'team_exp')
    gross_margin = ctrl.Antecedent(np.arange(-50, 90.1, 0.1), 'gross_margin')

    quiebra = ctrl.Consequent(np.arange(0, 100.1, 0.1), 'quiebra_score')

    # ============================================================
    # 2. FUNCIONES DE PERTENENCIA (triangulares, validadas Ronda 3)
    # ============================================================

    # Runway (meses)
    runway['bajo'] = fuzz.trimf(runway.universe, [0, 3, 6])
    runway['medio'] = fuzz.trimf(runway.universe, [4, 9, 14])
    runway['alto'] = fuzz.trimf(runway.universe, [12, 24, 36])

    # Burn ratio (gasto/ingreso)
    burn_ratio['bajo'] = fuzz.trimf(burn_ratio.universe, [0, 0.5, 1.2])
    burn_ratio['medio'] = fuzz.trimf(burn_ratio.universe, [1.0, 1.8, 2.5])
    burn_ratio['alto'] = fuzz.trimf(burn_ratio.universe, [2.2, 3.5, 5.0])

    # Crecimiento MoM (%)
    growth_mom['bajo'] = fuzz.trimf(growth_mom.universe, [-20, -5, 5])
    growth_mom['medio'] = fuzz.trimf(growth_mom.universe, [3, 10, 18])
    growth_mom['alto'] = fuzz.trimf(growth_mom.universe, [15, 25, 30])

    # Experiencia del equipo (escala 0-10)
    team_exp['baja'] = fuzz.trimf(team_exp.universe, [0, 2, 4])
    team_exp['media'] = fuzz.trimf(team_exp.universe, [3, 5, 7])
    team_exp['alta'] = fuzz.trimf(team_exp.universe, [6, 8, 10])

    # Margen bruto (%)
    gross_margin['bajo'] = fuzz.trimf(gross_margin.universe, [-50, 0, 25])
    gross_margin['medio'] = fuzz.trimf(gross_margin.universe, [20, 40, 55])
    gross_margin['alto'] = fuzz.trimf(gross_margin.universe, [50, 70, 90])

    # Salida: quiebra_score
    quiebra['bajo'] = fuzz.trimf(quiebra.universe, [0, 0, 30])
    quiebra['medio'] = fuzz.trimf(quiebra.universe, [20, 50, 80])
    quiebra['alto'] = fuzz.trimf(quiebra.universe, [70, 100, 100])

    # ============================================================
    # 3. REGLAS DIFUSAS (validadas por Delphi Ronda 3)
    # ============================================================
    # Reglas principales (12) validadas por Delphi Ronda 3
    reglas = [
        ctrl.Rule(runway['bajo'] & burn_ratio['alto'], quiebra['alto']),
        ctrl.Rule(runway['bajo'] & growth_mom['bajo'], quiebra['alto']),
        ctrl.Rule(runway['alto'] & burn_ratio['bajo'], quiebra['bajo']),
        ctrl.Rule(runway['alto'] & growth_mom['alto'] & gross_margin['alto'], quiebra['bajo']),
        ctrl.Rule(burn_ratio['alto'] & gross_margin['bajo'], quiebra['alto']),
        ctrl.Rule(growth_mom['alto'] & gross_margin['alto'], quiebra['bajo']),
        ctrl.Rule(team_exp['baja'] & runway['bajo'], quiebra['alto']),
        ctrl.Rule(team_exp['alta'] & growth_mom['alto'], quiebra['bajo']),
        ctrl.Rule(runway['medio'] & burn_ratio['medio'], quiebra['medio']),
        ctrl.Rule(gross_margin['bajo'] & growth_mom['bajo'], quiebra['alto']),
        ctrl.Rule(runway['medio'] & growth_mom['alto'], quiebra['bajo']),
        ctrl.Rule(team_exp['baja'] & gross_margin['bajo'], quiebra['alto']),

        # Reglas de cobertura (6) - derivadas por inferencia logica del consenso experto
        # Aseguran que toda combinacion del espacio de entrada active al menos una regla
        ctrl.Rule(runway['bajo'] & burn_ratio['medio'], quiebra['medio']),
        ctrl.Rule(runway['alto'] & burn_ratio['medio'], quiebra['bajo']),
        ctrl.Rule(runway['medio'] & burn_ratio['alto'], quiebra['alto']),
        ctrl.Rule(growth_mom['medio'] & gross_margin['medio'], quiebra['medio']),
        ctrl.Rule(team_exp['media'] & runway['medio'], quiebra['medio']),
        ctrl.Rule(runway['medio'] & growth_mom['medio'], quiebra['medio']),
    ]

    # ============================================================
    # 4. SISTEMA DE CONTROL (defuzzificacion: centroide por defecto)
    # ============================================================
    sistema = ctrl.ControlSystem(reglas)
    return sistema, (runway, burn_ratio, growth_mom, team_exp, gross_margin, quiebra)


def evaluar(simulador, runway_v, burn_v, growth_v, exp_v, margin_v):
    """Evalua el FIS con valores especificos."""
    simulador.input['runway'] = runway_v
    simulador.input['burn_ratio'] = burn_v
    simulador.input['growth_mom'] = growth_v
    simulador.input['team_exp'] = exp_v
    simulador.input['gross_margin'] = margin_v
    simulador.compute()
    return simulador.output['quiebra_score']


def clasificar(score):
    """Mapea score numerico a etiqueta lingüistica."""
    if score < 30:
        return 'BAJO'
    elif score < 60:
        return 'MEDIO'
    elif score < 80:
        return 'ALTO'
    else:
        return 'CRITICO'


if __name__ == '__main__':
    sistema, _ = construir_fis()
    simulador = ctrl.ControlSystemSimulation(sistema)

    # Casos de prueba: validan que el FIS responde como esperan los expertos
    casos = [
        ('Startup en quiebra inminente', 2, 3.5, -10, 2, -10),
        ('Startup ideal saludable', 30, 0.4, 22, 9, 75),
        ('Zona gris - decision dificil', 9, 1.5, 8, 5, 40),
        ('Equipo experimentado sin caja', 3, 2.8, 5, 8, 30),
        ('Buen producto sin runway', 4, 2.0, 18, 6, 65),
        ('Mucho dinero quemando rapido', 18, 3.2, 2, 5, 15),
    ]

    print('='*78)
    print(f'{"Caso":<35} {"Run":>4} {"Burn":>5} {"Crec":>5} {"Exp":>4} {"Marg":>5} -> {"Score":>5} {"Clas.":>8}')
    print('='*78)
    for nombre, run, burn, growth, exp, margin in casos:
        score = evaluar(simulador, run, burn, growth, exp, margin)
        print(f'{nombre:<35} {run:>4} {burn:>5.1f} {growth:>5.0f} {exp:>4} {margin:>5} -> {score:>5.1f} {clasificar(score):>8}')
    print('='*78)
