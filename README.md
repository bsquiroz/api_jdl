# API JDL — Riesgo de quiebra de startups

API REST que expone el modelo JDL para evaluar el riesgo de quiebra de una startup en horizonte de 12 meses. Implementada con **FastAPI**.

---

## Arquitectura

```
GET /evaluar?runway=12&burn_ratio=1.8&...
         ↓
   FastAPI (api.py)
         ↓
   FIS Mamdani (fis_quiebra.py)
         ↓
   Matriz de decisión (evaluador.py)
         ↓
   JSON con score, clasificación, acción y palancas
```

---

## Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Información del servicio |
| GET | `/salud` | Health check |
| GET | `/evaluar` | **Evaluar una startup** (5 parámetros query) |
| GET | `/ejemplos` | 3 casos demo precargados |
| GET | `/docs` | **Documentación Swagger interactiva** |

---

## Estructura del proyecto

```
api-jdl/
├── api.py              # API FastAPI (este archivo)
├── fis_quiebra.py      # Sistema de inferencia difusa (18 reglas)
├── evaluador.py        # Matriz de decisión + palancas
├── requirements.txt    # Dependencias Python
└── README.md           # Este documento
```
