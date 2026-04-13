import asyncio
import base64
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import fitz
import ollama
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from sentence_transformers import SentenceTransformer, util

from prompts.builder import PromptBuilder

LLM_MODEL      = os.getenv("LLM_MODEL", "qwen2.5:14b")
EMBED_MODEL    = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
DB_PATH        = os.getenv("DB_PATH", "db_candidatos.json")
MAX_MB         = int(os.getenv("MAX_FILE_SIZE_MB", "15"))
MAX_CV_CHARS   = int(os.getenv("MAX_CV_CHARS", "6000"))   

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("cva")

_db_lock        = asyncio.Lock()
_embed_model:   Optional[SentenceTransformer] = None
_prompt_builder: Optional[PromptBuilder]      = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_model, _prompt_builder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Cargando modelo de embeddings en %s...", device.upper())
    _embed_model    = await asyncio.to_thread(SentenceTransformer, EMBED_MODEL, device=device)
    _prompt_builder = PromptBuilder()
    log.info("CVA listo. Sectores disponibles: %s", _prompt_builder.sectores_disponibles())
    yield
    log.info("CVA apagado.")


app = FastAPI(title="CVA — Analizador de CVs", version="2.0.0", lifespan=lifespan)


def _leer_db() -> dict:
    if not os.path.exists(DB_PATH):
        return {}
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("DB ilegible, iniciando vacía: %s", exc)
        return {}


def _escribir_db(db: dict) -> None:
    tmp_path = DB_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, DB_PATH)


async def _persistir(clave: str, datos: dict) -> None:
    async with _db_lock:
        db = await asyncio.to_thread(_leer_db)
        db[clave] = datos
        await asyncio.to_thread(_escribir_db, db)


def _pdf_desde_ruta(ruta: str) -> str:
    with fitz.open(ruta) as doc:
        return " ".join(p.get_text("text").replace("\n", " ") for p in doc).strip()


def _pdf_desde_b64(b64: str) -> str:
    pdf_bytes = base64.b64decode(b64)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return " ".join(p.get_text("text").replace("\n", " ") for p in doc).strip()


async def _extraer_texto_cv(modo: str, archivo: Optional[UploadFile], cv_b64: Optional[str]) -> str:
    if modo == "file":
        if not archivo:
            raise HTTPException(400, "Modo 'file' requiere adjuntar un PDF.")
        if archivo.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(400, "Solo se aceptan archivos PDF.")

        contenido = await archivo.read()
        if len(contenido) > MAX_MB * 1024 * 1024:
            raise HTTPException(413, f"El archivo supera el límite de {MAX_MB} MB.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contenido)
            ruta = tmp.name
        try:
            texto = await asyncio.to_thread(_pdf_desde_ruta, ruta)
        except Exception as exc:
            raise HTTPException(422, f"No se pudo leer el PDF: {exc}") from exc
        finally:
            os.unlink(ruta)
        return texto

    elif modo == "b64":
        if not cv_b64:
            raise HTTPException(400, "Modo 'b64' requiere el campo cv_b64.")
        try:
            return await asyncio.to_thread(_pdf_desde_b64, cv_b64)
        except Exception as exc:
            raise HTTPException(422, f"Base64 inválido o PDF corrupto: {exc}") from exc

    else:
        raise HTTPException(400, f"Modo '{modo}' no soportado. Usa 'file' o 'b64'.")


def _calcular_score_semantico(texto_cv: str, vacante: str) -> float:
    v_cv  = _embed_model.encode(texto_cv, convert_to_tensor=True)
    v_vac = _embed_model.encode(vacante,  convert_to_tensor=True)
    return round(max(0.0, util.cos_sim(v_cv, v_vac).item()) * 100, 1)


def _llamar_llm(system_prompt: str, user_content: str) -> dict:
    respuesta = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        format="json",
    )
    raw = respuesta["message"]["content"]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("LLM devolvió JSON inválido: %s", raw[:300])
        raise ValueError(f"El modelo no devolvió JSON válido: {exc}") from exc


@app.post("/evaluar-cv", summary="Evaluar un CV contra una vacante")
async def evaluar_cv(
    modo:        str          = Form("file",    description="'file' o 'b64'"),
    archivo:     UploadFile   = File(None,      description="PDF del CV (modo file)"),
    cv_b64:      str          = Form(None,      description="PDF en base64 (modo b64)"),
    puesto:      str          = Form(...,       description="Nombre del puesto"),
    descripcion: str          = Form(...,       description="Descripción del puesto"),
    requiere:    str          = Form(...,       description="Requisitos del puesto"),
    sector:      str          = Form("general", description="Sector: ti, administrativo, rrhh, ventas, finanzas, arte_disenio, general"),
):
    texto_cv = await _extraer_texto_cv(modo, archivo, cv_b64)
    if not texto_cv:
        raise HTTPException(422, "El PDF no contiene texto extraíble.")

    vacante = f"Puesto: {puesto}. Descripción: {descripcion}. Requisitos: {requiere}"

    score_semantico = await asyncio.to_thread(
        _calcular_score_semantico, texto_cv, vacante
    )
    system_prompt = _prompt_builder.build(sector)
    user_content  = f"VACANTE:\n{vacante}\n\nCV DEL CANDIDATO:\n{texto_cv[:MAX_CV_CHARS]}"

    try:
        resultado_llm = await asyncio.to_thread(_llamar_llm, system_prompt, user_content)
    except ValueError as exc:
        raise HTTPException(502, f"Error en el modelo LLM: {exc}") from exc

    penalizacion  = float(resultado_llm.get("penalizacion_sugerida", 1.0))
    score_final   = max(0, min(100, round(score_semantico * penalizacion)))

    nombre = resultado_llm.get("nombreCandidato", "No especificado").upper().strip()
    ts     = datetime.now()

    resultado = {
        "nombreCandidato": nombre,
        "email":           resultado_llm.get("email",    "No especificado"),
        "telefono":        resultado_llm.get("telefono", "No especificado"),
        "calificacion":    score_final,
        "sector":          sector,
        "analisis_detallado": {
            "resumen_general":      resultado_llm.get("resumen_ejecutivo", ""),
            "fortalezas":           resultado_llm.get("fortalezas", ""),
            "debilidades_criticas": resultado_llm.get("debilidades", ""),
            "recomendacion_final":  resultado_llm.get("recomendacion_directa", ""),
        },
        "_meta": {
            "score_semantico_raw":   score_semantico,
            "penalizacion_aplicada": penalizacion,
            "modelo_llm":            LLM_MODEL,
            "fecha":                 ts.isoformat(timespec="seconds"),
        },
    }

    clave_db = f"{nombre}_{ts.strftime('%Y%m%d_%H%M%S')}"
    await _persistir(clave_db, resultado)

    log.info("CV evaluado — candidato: %s | score: %d | sector: %s", nombre, score_final, sector)
    return resultado


@app.get("/sectores", summary="Listar sectores disponibles")
def listar_sectores():
    return {"sectores": _prompt_builder.sectores_disponibles()}


@app.get("/candidatos", summary="Listar candidatos evaluados")
async def listar_candidatos():
    async with _db_lock:
        db = await asyncio.to_thread(_leer_db)
    return {"total": len(db), "candidatos": list(db.keys())}


@app.get("/health")
def health():
    return {"status": "ok", "modelo_llm": LLM_MODEL, "modelo_embed": EMBED_MODEL}
