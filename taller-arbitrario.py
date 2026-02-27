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


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def generate_scale(
    w: float,
    k: int,
    xmin: Optional[float] = None,
    xmin_floor: float = 0.01,  # para que "si existe la variable, cuenta"
) -> Dict:
    """
    Genera escala lógica de k categorías para una variable con peso w en [0,1].
    - x(j) equiespaciado en [xmin, 1]
    - contribution(j) = w * x(j)
    - delta(j) = contribution(j) - contribution(j-1)
    """
    if k < 2:
        raise ValueError("k debe ser >= 2 (mínimo 2 categorías).")

    w = clamp01(w)

    # xmin por defecto: si no se especifica, usamos xmin_floor como mínimo de escala
    if xmin is None:
        xmin = float(xmin_floor)

    xmin = clamp01(max(float(xmin_floor), float(xmin)))

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
        "w": float(w),
        "peso_pct": float(w * 100.0),
        "k": int(k),
        "xmin": float(xmin),
        "x_min_effective": x_min_effective,
        "x_max_effective": x_max_effective,
        "delta_max": round(delta_max, 6),
        "delta_max_pct": round(delta_max * 100.0, 4),
        "categories": results,
    }


# =========================
# App (taller arbitrario: k + w)
# =========================

st.set_page_config(page_title="Taller scoring por tarjetas", layout="wide")
st.title("Taller scoring — Tarjetas por variable (k + peso arbitrarios)")


def init_model():
    # Partimos de tus pesos originales (%) para inicializar, pero ahora se editan en [0,1]
    raw_pct = [
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
    for idx, (name, peso_pct) in enumerate(raw_pct, start=1):
        preset_labels = ["", "", ""]
        if "Nº de Ramos con nosotros" in name:
            preset_labels = ["0 ramos", "1-2 ramos", "3 o más ramos"]

        variables.append(
            {
                "id": f"var_{idx:02d}",
                "name": name,
                # ahora guardamos w en [0,1]
                "w": float(peso_pct) / 100.0,
                "k": 3,
                "labels": preset_labels,
                "notes": "",
            }
        )

    return {
        "variables": variables,
        "settings": {
            "normalize_weights": False,
        },
    }


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
# Acciones globales
# =========================

def clear_all(model: dict, default_k: int = 3) -> dict:
    """Borra etiquetas/notas y resetea a k=default_k. No toca los pesos."""
    for v in model.get("variables", []):
        var_id = v["id"]

        v["k"] = int(default_k)
        v["labels"] = [""] * int(default_k)
        v["notes"] = ""

        if "Nº de Ramos con nosotros" in v["name"]:
            v["labels"] = ["0 ramos", "1-2 ramos", "3 o más ramos"]

        st.session_state.pop(f"k_{var_id}", None)
        st.session_state.pop(f"notes_{var_id}", None)
        st.session_state.pop(f"w_{var_id}", None)

        for j in range(1, 11):
            st.session_state.pop(f"lbl_{var_id}_{j}", None)

    return model


def randomize_model(model: dict, k_min: int = 2, k_max: int = 6) -> dict:
    """Rellena k/etiquetas/notas con valores aleatorios (demo)."""
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


def get_effective_weights(model: dict) -> Dict[str, float]:
    """
    Devuelve w efectivo por id (aplicando o no normalización).
    NO modifica el modelo.
    """
    vars_ = model.get("variables", [])
    raw = {v["id"]: clamp01(float(v.get("w", 0.0))) for v in vars_}
    do_norm = bool(model.get("settings", {}).get("normalize_weights", False))
    total = sum(raw.values())

    if do_norm and total > 0:
        return {vid: w / total for vid, w in raw.items()}
    return raw


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.image("LOGOTIPO-AES-05.png", use_container_width=True)

    st.subheader("Controles globales")

    xmin_floor = st.slider(
        "Suelo mínimo xmin (para que la peor K sume)",
        min_value=0.0,
        max_value=0.30,
        value=0.01,
        step=0.01,
        help="Ej: 0.01 = 1% en escala [0,1]. Evita que K=1 aporte 0 cuando la variable existe.",
    )

    # Normalización de pesos
    st.session_state.model["settings"]["normalize_weights"] = st.toggle(
        "Normalizar pesos (que sumen 1)",
        value=bool(st.session_state.model["settings"].get("normalize_weights", False)),
        help="Si activas esto, el cálculo usa w_normalizado = w / suma(w). No cambia el valor guardado.",
    )

    # Métricas globales de pesos
    raw_sum = sum(clamp01(float(v.get("w", 0.0))) for v in st.session_state.model["variables"])
    st.metric("Suma pesos (introducidos)", f"{raw_sum:.4f}")

    cA, cB = st.columns(2)
    with cA:
        if st.button("Aleatorio (demo)", use_container_width=True):
            st.session_state.model = randomize_model(st.session_state.model)
            st.rerun()

    with cB:
        if st.button("Borrar todo (k/etiquetas/notas)", use_container_width=True):
            st.session_state.model = clear_all(st.session_state.model)
            st.rerun()

    st.divider()

    st.subheader("Leyenda columnas del scoring")
    st.markdown(
        """
**K (j)**  
Número de subcategoría (orden interno de peor → mejor).

**x(j)**  
Valor normalizado entre 0 y 1 asignado a cada categoría.

**Suma al score total % (w*x)**  
Impacto real que aporta esa categoría al score final del cliente.

**Δ vs prev %**  
Incremento de score respecto a la categoría anterior.
"""
    )

    st.divider()

    if st.button("↩️ Reset modelo (pierde cambios)", use_container_width=True):
        st.session_state.model = init_model()
        st.rerun()

    st.divider()

    st.caption("Exporta el estado del taller (w, k, etiquetas, notas + setting de normalización).")
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
w_effective_by_id = get_effective_weights(st.session_state.model)

col1, col2 = st.columns(2, gap="large")
cols = [col1, col2]

for i, var in enumerate(vars_list):
    normalize_labels(var)

    with cols[i % 2]:
        with st.container(border=True):
            st.subheader(var["name"])

            # Peso editable (raw) y peso efectivo (si hay normalización)
            w_raw = clamp01(float(var.get("w", 0.0)))
            w_eff = clamp01(float(w_effective_by_id.get(var["id"], w_raw)))

            m1, m2, m3 = st.columns(3)
            m1.metric("w (introducido)", f"{w_raw:.4f}")
            m2.metric("w (efectivo)", f"{w_eff:.4f}" + (" (norm.)" if st.session_state.model["settings"]["normalize_weights"] else ""))
            m3.metric("Peso (%) efectivo", f"{(w_eff * 100.0):.2f}%")

            new_w = st.slider(
                "Peso w (0 a 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(w_raw),
                step=0.01,
                key=f"w_{var['id']}",
                help="Peso arbitrario en [0,1]. Si activas normalización global, se reescala internamente.",
            )
            var["w"] = float(new_w)

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

            # Recalcular pesos efectivos después de edición de esta variable (sin rerun)
            # (para que el cálculo de la tarjeta sea consistente si normalizas)
            w_effective_by_id = get_effective_weights(st.session_state.model)
            w_eff = float(w_effective_by_id.get(var["id"], clamp01(float(var.get("w", 0.0)))))

            scale = generate_scale(
                w=float(w_eff),
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
w_effective_by_id = get_effective_weights(st.session_state.model)

for v in st.session_state.model["variables"]:
    w_raw = clamp01(float(v.get("w", 0.0)))
    w_eff = clamp01(float(w_effective_by_id.get(v["id"], w_raw)))
    summary.append(
        {
            "Variable": v["name"],
            "w (raw)": round(w_raw, 4),
            "w (efectivo)": round(w_eff, 4),
            "Peso % (efectivo)": round(w_eff * 100.0, 2),
            "k": int(v["k"]),
            "Etiquetas (preview)": " | ".join([lab for lab in v["labels"] if lab][:3]) + (" ..." if len(v["labels"]) > 3 else ""),
        }
    )

st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
