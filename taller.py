# app.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# =========================
# Lógica scoring
# =========================

@dataclass(frozen=True)
class CategoryResult:
    j: int
    x: float
    contribution: float
    contribution_pct: float
    delta_from_prev: float
    delta_from_prev_pct: float


def xmin_by_weight(w: float) -> float:
    """
    Asignación de x_min según el mapping del modelo.
    w en proporción (7,5% = 0.075)
    """
    peso_pct = round(w * 100, 1)
    if peso_pct <= 0:
        return 0.0

    mapping = {
        7.5: 0.00,
        5.5: 0.05,
        5.0: 0.10,
        4.5: 0.15,
        4.0: 0.20,
        3.0: 0.25,
        2.5: 0.30,
        2.0: 0.35,
        1.5: 0.40,
        1.0: 0.45,
        0.5: 0.50,
        0.0: 0.0,
    }
    return mapping.get(peso_pct, 0.20)


def generate_scale(
    peso_pct: float,
    k: int,
    xmin: Optional[float] = None,
    xmin_floor: float = 0.01,  #  para que "si existe la variable, cuenta" (por defecto 1%)
) -> Dict:
    """
    Genera escala lógica de k categorías para una variable con peso 'peso_pct' (%).
    - x(j) equiespaciado en [xmin, 1]
    - contribution(j) = w * x(j)
    - delta(j) = contribution(j) - contribution(j-1)
    """
    if k < 2:
        raise ValueError("k debe ser >= 2 (mínimo 2 categorías).")

    w = peso_pct / 100.0

    if xmin is None:
        xmin = xmin_by_weight(w)

    #  Suelo mínimo: evita que la peor categoría aporte 0 si la variable "existe"
    xmin = max(float(xmin_floor), float(xmin))
    xmin = max(0.0, min(1.0, float(xmin)))

    results: List[CategoryResult] = []
    prev_contrib = 0.0

    for j in range(1, k + 1):
        if w == 0:
            x = 0.0
        else:
            x = xmin + (j - 1) * (1.0 - xmin) / (k - 1)

        contrib = w * x
        delta = contrib - prev_contrib if j > 1 else 0.0

        results.append(
            CategoryResult(
                j=j,
                x=round(x, 6),
                contribution=round(contrib, 6),
                contribution_pct=round(contrib * 100.0, 4),
                delta_from_prev=round(delta, 6),
                delta_from_prev_pct=round(delta * 100.0, 4),
            )
        )
        prev_contrib = contrib

    x_min_effective = results[0].x if results else 0.0
    x_max_effective = results[-1].x if results else 0.0
    delta_max = w * (x_max_effective - x_min_effective)

    return {
        "peso_pct": float(peso_pct),
        "w": w,
        "k": int(k),
        "xmin": float(xmin),
        "x_min_effective": x_min_effective,
        "x_max_effective": x_max_effective,
        "delta_max": round(delta_max, 6),
        "delta_max_pct": round(delta_max * 100.0, 4),
        "categories": results,
    }


# =========================
# App (pesos fijos)
# =========================

st.set_page_config(page_title="Taller scoring por tarjetas", layout="wide")
st.title("Taller scoring — Tarjetas por variable (peso fijo)")


def init_model():
    raw = [
        ("Antigüedad 1ª contratación", 7.5),
        ("Vinculación: Nº de Ramos con nosotros", 7.5),
        ("Rentabilidad de la póliza actual", 7.5),
        ("Descuentos o Recargos aplicados sobre tarifa", 5.5),
        ("Morosidad: Históricos sin incidencia en devolución (Anotaciones de póliza)", 5.0),
        ("Engagement comercial / Uso de canales propios (App / Área cliente / Web privada)", 4.5),
        ("Frecuencia uso de coberturas complementarias que no emiten siniestralidad.", 4.5),
        ("Total de asegurados en el Total de sus pólizas - Media de asegurados por póliza", 4.5),
        ("Edad", 4.5),
        ("Rentabilidad de la póliza retrospectivo/histórica (LTV)", 4.5),
        ("Tipo de distribución", 4.5),
        ("Vinculación: Coberturas complementarias opcionales", 4.5),
        ("Contactabilidad: Más de X Campos de datos (Tiene App, Teléfono, Mail, etc.).", 4.0),
        ("Edad del asegurado más mayor", 4.0),
        ("Vinculación familiar", 3.0),
        ("Prescriptor", 3.0),
        ("Exposición a comunicaciones de marca (RRSS, mailing…)", 3.0),
        ("Descendencia", 3.0),
        ("Medio de pago", 2.5),
        ("Frecuencia de pago (Periodicidad)", 2.0),
        ("Probabilidad de desglose", 1.5),
        ("Tipo de producto", 1.5),
        ("NPS", 1.5),
        ("Mascotas", 1.5),
        ("Localización (enfocado a potencial de compra)", 1.5),
        ("Autónomo", 1.0),
        ("Siniestralidad (Salud)", 1.0),
        ("Grado de digitalización de la póliza", 0.5),
        ("Profesión", 0.5),
        ("Nivel educativo", 0.5),
        ("Sexo", 0.0),
    ]

    variables = []
    for idx, (name, peso) in enumerate(raw, start=1):
        preset_labels = ["", "", ""]
        if "Nº de Ramos con nosotros" in name:
            preset_labels = ["0 ramos", "1-2 ramos", "3 o más ramos"]

        variables.append(
            {
                "id": f"var_{idx:02d}",
                "name": name,
                "peso_pct": float(peso),  # fijo
                "k": 3,
                "labels": preset_labels,
                "notes": "",
            }
        )

    return {"variables": variables}


if "model" not in st.session_state:
    st.session_state.model = init_model()


def normalize_labels(var: dict):
    k = int(var["k"])
    labels = list(var.get("labels") or [])
    if len(labels) < k:
        labels += [""] * (k - len(labels))
    if len(labels) > k:
        labels = labels[:k]
    var["labels"] = labels


def scale_to_df(scale: dict, labels: List[str]) -> pd.DataFrame:
    rows = []
    for idx, r in enumerate(scale["categories"]):
        label = labels[idx] if idx < len(labels) else ""
        rows.append(
            {
                "K (j)": r.j,
                "Etiqueta (texto libre)": label,
                "x(j)": r.x,
                "Suma al score total % (w*x)": r.contribution_pct,
                "Δ vs prev %": r.delta_from_prev_pct,
            }
        )
    return pd.DataFrame(rows)


# =========================
# NUEVO: Acciones globales
# =========================

def clear_all(model: dict, default_k: int = 3) -> dict:
    """Borra etiquetas y notas; pone k=default_k en todas (mantiene pesos)."""
    for v in model.get("variables", []):
        v["k"] = int(default_k)
        v["labels"] = [""] * int(default_k)
        v["notes"] = ""
        # reponer preset de ramos si quieres mantenerlo incluso al borrar:
        if "Nº de Ramos con nosotros" in v["name"]:
            v["labels"] = ["0 ramos", "1-2 ramos", "3 o más ramos"]
    return model


def randomize_model(model: dict, k_min: int = 2, k_max: int = 6) -> dict:
    """Rellena el modelo con valores aleatorios (demo)."""
    demo_words = ["Peor", "Bajo", "Medio", "Alto", "Mejor", "Top", "Ok", "Riesgo", "Premium", "Básico"]
    for v in model.get("variables", []):
        k = random.randint(k_min, k_max)
        v["k"] = k
        v["labels"] = [f"{random.choice(demo_words)} {i+1}" for i in range(k)]
        v["notes"] = f"Auto-demo ({k} categorías). Reemplazar en el taller."
        if "Nº de Ramos con nosotros" in v["name"]:
            v["k"] = 3
            v["labels"] = ["0 ramos", "1-2 ramos", "3 o más ramos"]
    return model


# =========================
# Sidebar
# =========================

with st.sidebar:

    # --- LOGO ---
    st.image("LOGOTIPO-AES-02.png", use_container_width=True)

    st.subheader("Controles")

    xmin_floor = st.slider(
        "Suelo mínimo xmin (para que la peor K sume)",
        min_value=0.0,
        max_value=0.30,
        value=0.01,
        step=0.01,
        help="Ej: 0.01 = 1% en escala [0,1]. Evita que K=1 aporte 0 cuando la variable existe.",
    )

    cA, cB = st.columns(2)
    with cA:
        if st.button("Aleatorio (demo)", use_container_width=True):
            st.session_state.model = randomize_model(st.session_state.model)
            st.rerun()

    with cB:
        if st.button("Borrar todo", use_container_width=True):
            st.session_state.model = clear_all(st.session_state.model)
            st.rerun()

    st.divider()

    if st.button("↩️ Reset modelo (pierde cambios)", use_container_width=True):
        st.session_state.model = init_model()
        st.rerun()

    st.divider()
    st.caption("Exporta el estado del taller (k, etiquetas, notas).")
    st.download_button(
        "⬇️ Descargar JSON del taller",
        data=pd.Series(st.session_state.model).to_json(force_ascii=False, indent=2).encode("utf-8"),
        file_name="taller_scoring.json",
        mime="application/json",
        use_container_width=True,
    )


# =========================
# Tarjetas
# =========================

vars_list = st.session_state.model["variables"]

col1, col2 = st.columns(2, gap="large")
cols = [col1, col2]

for i, var in enumerate(vars_list):
    normalize_labels(var)

    with cols[i % 2]:
        with st.container(border=True):
            st.subheader(var["name"])

            w = float(var["peso_pct"]) / 100.0
            xmin_auto = xmin_by_weight(w)
            xmin_effective = max(float(xmin_floor), float(xmin_auto))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Peso (%)", f"{var['peso_pct']:.1f}")
            m2.metric("w", f"{w:.4f}")
            m3.metric("xmin (auto)", f"{xmin_auto:.2f}")
            m4.metric("xmin (efectivo)", f"{xmin_effective:.2f}")

            new_k = st.number_input(
                "k (nº de subcategorías)",
                min_value=2,
                max_value=10,
                value=int(var["k"]),
                step=1,
                key=f"k_{var['id']}",
            )
            var["k"] = int(new_k)
            normalize_labels(var)

            st.markdown("**Etiquetas por categoría (texto libre)**")
            left, right = st.columns(2)
            for j in range(1, int(var["k"]) + 1):
                target = left if j % 2 == 1 else right
                var["labels"][j - 1] = target.text_input(
                    f"K = {j}",
                    value=var["labels"][j - 1],
                    key=f"lbl_{var['id']}_{j}",
                    placeholder="Ej: 1-2 ramos",
                )

            var["notes"] = st.text_area(
                "Notas / criterio (opcional)",
                value=var.get("notes", ""),
                key=f"notes_{var['id']}",
                placeholder="Cómo decidimos la categorización, rangos, etc.",
            )

            scale = generate_scale(
                peso_pct=float(var["peso_pct"]),
                k=int(var["k"]),
                xmin=None,
                xmin_floor=float(xmin_floor),
            )
            df = scale_to_df(scale, var["labels"])

            st.caption(f"Impacto máximo de esta variable (Δ máx): {scale['delta_max_pct']:.2f}% del score total")
            st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Resumen del modelo (solo lo que cambia en el taller)")
summary = []
for v in st.session_state.model["variables"]:
    summary.append(
        {
            "Variable": v["name"],
            "Peso (%)": float(v["peso_pct"]),
            "k": int(v["k"]),
            "Etiquetas (preview)": " | ".join([lab for lab in v["labels"] if lab][:3]) + (" ..." if len(v["labels"]) > 3 else ""),
        }
    )
st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
