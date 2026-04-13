# CVA — Mini analizador de CVs v2.0

> **💡 Contexto del Proyecto (Proof of Concept)** > Este repositorio es una Prueba de Concepto (PoC) personal diseñada para experimentar con el ecosistema de IA generativa ejecutada en local (Ollama, Qwen). Dado que mi stack principal se centra en arquitecturas .NET y Angular, utilicé IA generativa como "pair programmer" para acelerar el desarrollo del boilerplate en Python y FastAPI. 
> 
> El objetivo principal de este proyecto no es demostrar maestría sintáctica en Python, sino explorar la **arquitectura de integración de LLMs**, el diseño de prompts dinámicos mediante YAML, y la gestión de procesos bloqueantes en APIs asíncronas.

## Cómo levantar

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Variables de entorno:

| Variable          | Default              | Descripción                          |
|-------------------|----------------------|--------------------------------------|
| `LLM_MODEL`       | `qwen2.5:14b`        | Modelo Ollama a usar                 |
| `EMBED_MODEL`     | `all-MiniLM-L6-v2`   | Modelo de sentence-transformers      |
| `DB_PATH`         | `db_candidatos.json` | Ruta del archivo de historial        |
| `MAX_FILE_SIZE_MB`| `15`                 | Límite de tamaño de PDF              |
| `MAX_CV_CHARS`    | `6000`               | Cap de caracteres enviados al LLM    |
| `LOG_LEVEL`       | `INFO`               | Nivel de logging                     |

---

## Endpoints

| Método | Ruta          | Descripción                        |
|--------|---------------|------------------------------------|
| POST   | /evaluar-cv   | Evaluar un CV contra una vacante   |
| GET    | /sectores     | Listar sectores disponibles        |
| GET    | /candidatos   | Listar candidatos evaluados        |
| GET    | /health       | Estado del servicio                |
| GET    | /docs         | Swagger UI (FastAPI automático)    |

### Parámetros de /evaluar-cv

```
modo:        "file" | "b64"           (default: "file")
archivo:     PDF adjunto              (requerido si modo=file)
cv_b64:      string base64 del PDF    (requerido si modo=b64)
puesto:      str                      (requerido)
descripcion: str                      (requerido)
requiere:    str                      (requerido)
sector:      str                      (default: "general")
```

### Sectores disponibles

| Valor           | Descripción                   |
|-----------------|-------------------------------|
| `general`       | Fallback genérico             |
| `ti`            | Tecnologías de la Información |
| `administrativo`| Administrativo y Operaciones  |
| `rrhh`          | Recursos Humanos              |
| `ventas`        | Ventas y Comercial            |
| `finanzas`      | Finanzas y Contabilidad       |
| `arte_disenio`  | Arte, Diseño y Creatividad    |

---

## Agregar un nuevo sector

Crea un archivo `prompts/sectors/mi_sector.yaml` con esta estructura:

```yaml
nombre: "Nombre del Sector"
rol_evaluador: "descripción del rol evaluador que adoptará el LLM"

evaluation_focus: |
  FOCO DE EVALUACIÓN:
  Qué aspectos priorizar, cómo interpretar la experiencia, qué señales buscar...

equivalencias: |
  EQUIVALENCIAS:
  - "Término A / Término B / Término C" → concepto unificado
  ...

penalization_guide: |
  1.0 → Match total: ...
  0.8 → Match parcial: ...
  0.5 → Brecha significativa: ...
  0.2 → Incompatibilidad: ...
```

Eso es todo. El sistema lo detecta automáticamente al siguiente request.

---

## Arquitectura de decisiones

### ¿Por qué una sola llamada LLM?
La v1 hacía dos llamadas: una para extraer datos personales y otra para evaluar.
Fusionarlas reduce la latencia ~50% y el LLM tiene más contexto al extraer los datos.

### ¿Por qué YAMLs y no prompts en Python?
El equipo de RRHH puede ajustar los criterios de evaluación sin tocar código.
El builder los cachea con `@lru_cache`, así que no hay overhead de I/O en producción.

### ¿Por qué `asyncio.to_thread`?
Ollama y sentence-transformers son bloqueantes (síncronos). Dentro de un endpoint
`async def` de FastAPI, bloquearían el event loop completo. `to_thread` los delega
a un thread pool, manteniendo el servidor responsivo.

### ¿Por qué escritura atómica en la DB?
`os.replace(tmp, DB_PATH)` es atómica en Linux. Si el proceso muere en mitad de la
escritura, el archivo original queda intacto. Sin esto, una caída puede dejar el JSON
corrupto y perder todo el historial.
