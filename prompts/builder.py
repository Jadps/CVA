import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger("cva.prompts")

SECTORS_DIR = Path(__file__).parent / "sectors"

_BASE_TEMPLATE = """Eres un evaluador de CVs experto. Actúas como {rol_evaluador} para el sector: {sector_nombre}.

{evaluation_focus}

{equivalencias}

Analiza el CV del candidato contra la vacante recibida y devuelve ÚNICAMENTE un JSON válido con la siguiente estructura. Sin markdown, sin texto fuera del JSON:

{{
    "nombreCandidato": "nombre completo del candidato tal como aparece en el CV",
    "email": "email o 'No especificado'",
    "telefono": "teléfono o 'No especificado'",
    "resumen_ejecutivo": "2-3 oraciones sobre la idoneidad del candidato para ESTA vacante específica",
    "fortalezas": "narrativa con 2-4 puntos fuertes directamente relevantes para el puesto",
    "debilidades": "brechas críticas vs los requisitos de la vacante, o 'Sin brechas significativas'",
    "penalizacion_sugerida": <número decimal entre 0.0 y 1.0 según la guía abajo>,
    "recomendacion_directa": "una de estas tres opciones con justificación breve: 'Contratar — [motivo]' | 'Segunda entrevista — [motivo]' | 'Descartar — [motivo]'"
}}

GUÍA DE PENALIZACIÓN (penalizacion_sugerida):
{penalization_guide}

REGLAS ABSOLUTAS:
- Si un dato no aparece en el CV: escribe exactamente "No especificado"
- No inventes experiencia, habilidades ni logros que no estén en el CV
- Evalúa la vacante concreta recibida, no un puesto genérico
- Responde SOLO con el JSON. Sin explicaciones adicionales.
"""


@lru_cache(maxsize=32)
def _cargar_sector(sector: str) -> dict:
    """
    Carga y cachea el YAML del sector.
    Si el sector no existe, cae al 'general' sin lanzar error.
    """
    ruta = SECTORS_DIR / f"{sector}.yaml"
    if not ruta.exists():
        log.warning("Sector '%s' no encontrado, usando 'general'.", sector)
        ruta = SECTORS_DIR / "general.yaml"
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class PromptBuilder:

    def build(self, sector: str) -> str:
        """Construye el system prompt completo para el sector dado."""
        cfg = _cargar_sector(sector.lower().strip())
        return _BASE_TEMPLATE.format(
            sector_nombre      = cfg.get("nombre", "General"),
            rol_evaluador      = cfg.get("rol_evaluador", "experto en selección de personal"),
            evaluation_focus   = cfg.get("evaluation_focus", "").strip(),
            equivalencias      = cfg.get("equivalencias", "").strip(),
            penalization_guide = cfg.get("penalization_guide", "").strip(),
        )

    def sectores_disponibles(self) -> list[str]:
        return sorted(p.stem for p in SECTORS_DIR.glob("*.yaml"))
