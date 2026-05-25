"""
API REST - Modelo JDL para prediccion de quiebra de startups
=============================================================
Endpoint GET /evaluar que toma 5 metricas de una startup
y devuelve la evaluacion completa del FIS Mamdani.

Para correr:
    uvicorn api:app --reload
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from skfuzzy import control as ctrl
from fastapi.middleware.cors import CORSMiddleware
import warnings
warnings.filterwarnings('ignore')

from fis_quiebra import construir_fis, clasificar
from evaluador import MATRIZ_DECISION, palancas_mejora

app = FastAPI(
    title="API JDL - Riesgo de quiebra de startups",
    description=(
        "Sistema experto basado en arquitectura JDL que evalua el riesgo de "
        "quiebra de una startup en horizonte de 12 meses. Integra metodo Delphi, "
        "sistema de inferencia difusa Mamdani, simulacion Monte Carlo y "
        "modelos de aprendizaje automatico.\n\n"
        "**Autor:** Brayan Stiven Munoz Quiroz\n\n"
        "**Asignatura:** Modelos y Simulacion - Mayo 2026"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Construir el FIS una sola vez al iniciar la API (mas eficiente)
SISTEMA, _ = construir_fis()


@app.get("/", tags=["Info"])
def root():
    """Endpoint raiz con informacion del servicio."""
    return {
        "servicio": "API JDL - Riesgo de quiebra de startups",
        "version": "1.0.0",
        "documentacion": "/docs",
        "endpoints": {
            "GET /": "Info del servicio",
            "GET /salud": "Health check",
            "GET /evaluar": "Evaluar una startup (parametros query)",
            "GET /ejemplos": "Ver 3 casos demo precargados",
        },
    }


@app.get("/salud", tags=["Info"])
def salud():
    """Health check del servicio."""
    return {"status": "ok", "fis_cargado": True}


@app.get("/evaluar", tags=["Evaluacion"])
def evaluar(
    runway: float = Query(
        ..., ge=0, le=36,
        description="Meses de caja restantes (0-36)",
        example=12,
    ),
    burn_ratio: float = Query(
        ..., ge=0, le=5,
        description="Gasto mensual / ingreso mensual (0-5)",
        example=1.8,
    ),
    growth_mom: float = Query(
        ..., ge=-20, le=30,
        description="Crecimiento mensual de ingresos en %  (-20 a +30)",
        example=11.1,
    ),
    team_exp: float = Query(
        ..., ge=0, le=10,
        description="Experiencia ponderada del equipo en escala 0-10",
        example=8,
    ),
    gross_margin: float = Query(
        ..., ge=-50, le=90,
        description="Margen bruto en %  (-50 a +90)",
        example=35,
    ),
    nombre: str = Query(
        "Startup", description="Nombre de la startup (opcional)",
    ),
):
    """
    Evalua el riesgo de quiebra de una startup.

    Retorna:
    - **score**: numero 0-100 (mayor = mayor riesgo de quiebra)
    - **clasificacion**: BAJO / MEDIO / ALTO / CRITICO
    - **color**: verde / amarillo / naranja / rojo
    - **accion**: INVERTIR / DUE DILIGENCE / NEGOCIAR / RECHAZAR
    - **descripcion**: explicacion de la accion
    - **urgencia**: ventana de tiempo recomendada
    - **palancas**: lista de mejoras accionables
    """
    try:
        sim = ctrl.ControlSystemSimulation(SISTEMA)
        sim.input['runway'] = float(runway)
        sim.input['burn_ratio'] = float(burn_ratio)
        sim.input['growth_mom'] = float(growth_mom)
        sim.input['team_exp'] = float(team_exp)
        sim.input['gross_margin'] = float(gross_margin)
        sim.compute()
        score = float(sim.output['quiebra_score'])
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"El FIS no pudo evaluar esta combinacion de inputs "
                f"(no se activo ninguna regla). Detalle: {str(e)}"
            ),
        )

    clase = clasificar(score)
    decision = MATRIZ_DECISION[clase]
    entradas = {
        'runway': runway, 'burn_ratio': burn_ratio,
        'growth_mom': growth_mom, 'team_exp': team_exp,
        'gross_margin': gross_margin,
    }
    palancas = palancas_mejora(entradas, score)

    return {
        "nombre": nombre,
        "entradas": entradas,
        "resultado": {
            "score": round(score, 2),
            "clasificacion": clase,
            "color": decision['color'],
            "color_hex": decision['hex'],
            "accion": decision['accion'],
            "descripcion": decision['descripcion'],
            "urgencia": decision['urgencia'],
        },
        "palancas_mejora": palancas,
    }


@app.get("/ejemplos", tags=["Ejemplos"])
def ejemplos():
    """
    Devuelve 3 casos demo precargados para probar la API sin necesidad de inputs.

    Cada caso incluye:
    - Descripcion narrativa (lenguaje natural)
    - Inputs traducidos al lenguaje del modelo
    - Evaluacion completa
    """
    casos = [
        {
            "id": "ejemplo_1",
            "nombre": "Healthy SaaS B2B",
            "narrativa": (
                "Startup SaaS B2B con 14 meses operando. Gasto el mes pasado "
                "USD 180k y facturo USD 100k. Crecio de USD 90k a USD 100k entre "
                "marzo y abril. El CEO tiene un PhD y 8 anos en el sector. "
                "Margen bruto del 35%."
            ),
            "traduccion": {
                "runway": 12,
                "burn_ratio": 1.8,
                "growth_mom": 11.1,
                "team_exp": 8,
                "gross_margin": 35,
            },
        },
        {
            "id": "ejemplo_2",
            "nombre": "Burning B2C Marketplace",
            "narrativa": (
                "Marketplace B2C de delivery. Solo le quedan 5 meses de caja. "
                "Gasta USD 250k al mes pero solo factura USD 90k. El crecimiento "
                "se estanco: paso de USD 88k a USD 90k entre meses. El equipo "
                "fundador es de primer emprendimiento (2 anos de experiencia "
                "promedio). Los margenes son magros, apenas 8%."
            ),
            "traduccion": {
                "runway": 5,
                "burn_ratio": 2.8,
                "growth_mom": 2.3,
                "team_exp": 2,
                "gross_margin": 8,
            },
        },
        {
            "id": "ejemplo_3",
            "nombre": "Promising Early Stage",
            "narrativa": (
                "Fintech en seed. Acaba de levantar pre-seed y tiene 24 meses "
                "de caja. Burn ratio bajo (0.6x) porque opera con equipo lean. "
                "Esta creciendo 22% mes a mes desde el lanzamiento. Los "
                "fundadores son ex-Nubank con 9 anos de experiencia. Margen "
                "bruto del 60%."
            ),
            "traduccion": {
                "runway": 24,
                "burn_ratio": 0.6,
                "growth_mom": 22,
                "team_exp": 9,
                "gross_margin": 60,
            },
        },
    ]

    # Evaluar cada uno
    for caso in casos:
        t = caso["traduccion"]
        try:
            sim = ctrl.ControlSystemSimulation(SISTEMA)
            sim.input['runway'] = float(t['runway'])
            sim.input['burn_ratio'] = float(t['burn_ratio'])
            sim.input['growth_mom'] = float(t['growth_mom'])
            sim.input['team_exp'] = float(t['team_exp'])
            sim.input['gross_margin'] = float(t['gross_margin'])
            sim.compute()
            score = float(sim.output['quiebra_score'])
            clase = clasificar(score)
            decision = MATRIZ_DECISION[clase]
            caso["resultado"] = {
                "score": round(score, 2),
                "clasificacion": clase,
                "color": decision['color'],
                "accion": decision['accion'],
            }
        except Exception as e:
            caso["resultado"] = {"error": str(e)}

    return {"casos": casos}
