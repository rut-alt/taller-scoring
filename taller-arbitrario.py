# app.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

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


def generate_scale_manual_x(
    peso_pct: float,   # peso en % (ya reescalado si aplica)
    k: int,
    x_values: List[float],
) -> Dict:
    """
    x(j) manual (nivel) en [0,1]
    contrib_pct(j) = peso_pct * x(j)
    """
    if k < 2:
        raise ValueError("k debe ser >= 2.")
    peso_pct = float(peso_pct)

    xs = list(x_values or [])
    if len(xs) < k:
        xs += [0.0] * (k - len(xs))
    if len(xs) > k:
        xs = xs[:k]
    xs = [clamp01(x) for x in xs]

    results: List[CategoryResult] = []
    prev_contrib_pct = 0.0

    for j in range(1, k + 1):
        x = xs[j - 1]
        contrib_pct = peso_pct * x
        delta_pct = contrib_pct - prev_contrib_pct if j > 1 else 0.0

        results.append(
            CategoryResult(
                j=j,
                x=round(x, 6),
                contribution=round(contrib_pct / 100.0, 6),  # en proporción, por compatibilidad
                contribution_pct=round(contrib_pct, 4),      # EN %
                delta_from_prev=round(delta_pct / 100.0, 6),
                delta_from_prev_pct=round(delta_pct, 4),     # EN %
            )
        )
        prev_contrib_pct = contrib_pct

    x_min_effective = results[0].x if results else 0.0
    x_max_effective = results[-1].x if results else 0.0
    delta_max_pct = peso_pct * (x_max_effective - x_min_effective)

    return {
        "peso_pct": float(peso_pct),
        "k": int(k),
        "x_min_effective": x_min_effective,
        "x_max_effective": x_max_effective,
        "delta_max_pct": round(delta_max_pct, 4),
        "categories": results,
        "x_values": xs,
    }


# =========================
# App
# =========================

st.set_page_config(page_title="Taller scoring por tarjetas", layout="wide")
st.title("Taller scoring — pesos fijos + x(j) manual (máximo 100%)")


def init_model():
    # Pesos fijos del modelo (%)
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
        preset_x = [0.0, 0.5, 1.0]

        if "Nº de Ramos con nosotros" in name:
            preset_labels = ["0 ramos", "1-2 ramos", "3 o más ramos"]
            preset_x = [0.0, 0.6, 1.0]

        variables.append(
            {
                "id": f"var_{idx:02d}",
                "name": name,
                "peso_pct": float(peso_pct),  # FIJO
                "k": 3,
                "labels": preset_labels,
                "x_values": preset_x,
                "notes": "",
            }
        )

    return {
        "variables": variables,
        "settings": {
            # si los pesos no suman 100, esto los reescala proporcionalmente
            "rescale_weights_to_100": True,
            # anclas recomendadas para escala por variable
            "force_best_x1": True,
            "force_worst_x0": False,
            "enforce_monotone_default": True,
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


def normalize_x_values(var: dict):
    k = int(var["k"])
    xs = list(var.get("x_values") or [])
    if len(xs) < k:
        xs += [0.0] * (k - len(xs))
    if len(xs) > k:
        xs = xs[:k]
    var["x_values"] = [clamp01(x) for x in xs]


def get_effective_peso_pct_by_id(model: dict) -> Dict[str, float]:
    """
    Devuelve peso_pct efectivo por variable.
    Si rescale_weights_to_100=True, aplica alpha = 100 / suma_pesos_raw.
    """
    vars_ = model.get("variables", [])
    raw = {v["id"]: float(v.get("peso_pct", 0.0)) for v in vars_}
    total_raw = sum(raw.values())
    do_rescale = bool(model.get("settings", {}).get("rescale_weights_to_100", True))

    if do_rescale and total_raw > 0:
        alpha = 100.0 / total_raw
        return {vid: w * alpha for vid, w in raw.items()}
    return raw


def scale_to_df(scale: dict, labels: List[str]) -> pd.DataFrame:
    rows = []
    for idx, r in enumerate(scale["categories"]):
        label = labels[idx] if idx < len(labels) else ""
        rows.append(
            {
                "K (j)": r.j,
                "Etiqueta (texto libre)": label,
                "x(j)": r.x,
                "Suma al score total % (peso*x)": r.contribution_pct,
                "Δ vs prev %": r.delta_from_prev_pct,
            }
        )
    return pd.DataFrame(rows)


# =========================
# Acciones globales
# =========================

def clear_all(model: dict, default_k: int = 3) -> dict:
    for v in model.get("variables", []):
        var_id = v["id"]
        v["k"] = int(default_k)
        v["labels"] = [""] * int(default_k)
        v["x_values"] = [0.0, 0.5, 1.0] if default_k == 3 else [round(i / (default_k - 1), 6) for i in range(default_k)]
        v["notes"] = ""

        if "Nº de Ramos con nosotros" in v["name"]:
            v["labels"] = ["0 ramos", "1-2 ramos", "3 o más ramos"]
            v["x_values"] = [0.0, 0.6, 1.0]

        st.session_state.pop(f"k_{var_id}", None)
        st.session_state.pop(f"notes_{var_id}", None)
        st.session_state.pop(f"mono_{var_id}", None)
        for j in range(1, 11):
            st.session_state.pop(f"lbl_{var_id}_{j}", None)
            st.session_state.pop(f"x_{var_id}_{j}", None)
    return model


def randomize_model(model: dict, k_min: int = 2, k_max: int = 6) -> dict:
    demo_words = ["Peor", "Bajo", "Medio", "Alto", "Mejor", "Top", "Ok", "Riesgo", "Premium", "Básico"]
    for v in model.get("variables", []):
        k = random.randint(k_min, k_max)
        v["k"] = k
        v["labels"] = [f"{random.choice(demo_words)} {i+1}" for i in range(k)]
        v["notes"] = f"Auto-demo ({k} categorías). Reemplazar en el taller."

        xs = sorted([random.random() for _ in range(k)])
        xs[0] = 0.0
        xs[-1] = 1.0
        v["x_values"] = [round(clamp01(x), 4) for x in xs]

        if "Nº de Ramos con nosotros" in v["name"]:
            v["k"] = 3
            v["labels"] = ["0 ramos", "1-2 ramos", "3 o más ramos"]
            v["x_values"] = [0.0, 0.6, 1.0]
    return model


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.image("LOGOTIPO-AES-05.png", use_container_width=True)

    st.subheader("Controles globales")

    st.session_state.model["settings"]["rescale_weights_to_100"] = st.toggle(
        "Reescalar pesos para que SUMEN 100%",
        value=bool(st.session_state.model["settings"].get("rescale_weights_to_100", True)),
        help="Si la suma de pesos fijos no es 100, aplica un factor proporcional para que el score máximo sea 100%.",
    )

    st.session_state.model["settings"]["force_best_x1"] = st.toggle(
        "Forzar x(mejor) = 1",
        value=bool(st.session_state.model["settings"].get("force_best_x1", True)),
        help="Recomendado para que el mejor cliente pueda alcanzar el peso completo de cada variable.",
    )

    st.session_state.model["settings"]["force_worst_x0"] = st.toggle(
        "Forzar x(peor) = 0",
        value=bool(st.session_state.model["settings"].get("force_worst_x0", False)),
        help="Opcional: fija la peor categoría a 0 para todas las variables.",
    )

    # métricas globales
    raw_sum = sum(float(v.get("peso_pct", 0.0)) for v in st.session_state.model["variables"])
    eff_map = get_effective_peso_pct_by_id(st.session_state.model)
    eff_sum = sum(eff_map.values())

    st.metric("Suma pesos (raw)", f"{raw_sum:.2f}%")
    st.metric("Suma pesos (efectiva)", f"{eff_sum:.2f}%")

    if abs(eff_sum - 100.0) < 1e-6:
        st.success("Score máximo teórico (x=1 en todo): 100% ✅")
    else:
        st.warning(f"Score máximo teórico (x=1 en todo): {eff_sum:.2f}%")

    cA, cB = st.columns(2)
    with cA:
        if st.button("Aleatorio (demo)", use_container_width=True):
            st.session_state.model = randomize_model(st.session_state.model)
            st.rerun()

    with cB:
        if st.button("Borrar todo (k/etiquetas/x/notas)", use_container_width=True):
            st.session_state.model = clear_all(st.session_state.model)
            st.rerun()

    st.divider()

    st.subheader("Leyenda")
    st.markdown(
        """
**Peso fijo (%)**: importancia de la variable en el score.

**x(j)**: nivel manual (0 a 1) por categoría.

**Aporte (%)**: `peso% * x(j)`

Si reescalas pesos a 100%, el cliente perfecto (x=1 en todo) suma 100%.
"""
    )

    st.divider()

    if st.button("↩️ Reset modelo (pierde cambios)", use_container_width=True):
        st.session_state.model = init_model()
        st.rerun()

    st.divider()

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
peso_eff_by_id = get_effective_peso_pct_by_id(st.session_state.model)

col1, col2 = st.columns(2, gap="large")
cols = [col1, col2]

force_best = bool(st.session_state.model["settings"].get("force_best_x1", True))
force_worst = bool(st.session_state.model["settings"].get("force_worst_x0", False))
enforce_default = bool(st.session_state.model["settings"].get("enforce_monotone_default", True))

for i, var in enumerate(vars_list):
    normalize_labels(var)
    normalize_x_values(var)

    with cols[i % 2]:
        with st.container(border=True):
            st.subheader(var["name"])

            peso_raw = float(var.get("peso_pct", 0.0))
            peso_eff = float(peso_eff_by_id.get(var["id"], peso_raw))

            m1, m2 = st.columns(2)
            m1.metric("Peso fijo (raw) %", f"{peso_raw:.2f}")
            m2.metric("Peso efectivo %", f"{peso_eff:.2f}" + (" (reescalado)" if st.session_state.model["settings"]["rescale_weights_to_100"] else ""))

            var["k"] = int(
                st.number_input(
                    "k (nº de subcategorías)",
                    min_value=2,
                    max_value=10,
                    value=int(var["k"]),
                    step=1,
                    key=f"k_{var['id']}",
                )
            )

            normalize_labels(var)
            normalize_x_values(var)

            st.markdown("**Etiquetas por categoría (texto libre)**")
            left, right = st.columns(2)
            for j in range(1, int(var["k"]) + 1):
                target = left if j % 2 == 1 else right
                var["labels"][j - 1] = target.text_input(
                    f"K = {j}",
                    value=var["labels"][j - 1],
                    key=f"lbl_{var['id']}_{j}",
                )

            st.markdown("**x(j) manual (0 a 1)**")
            leftx, rightx = st.columns(2)

            enforce_monotone = st.checkbox(
                "Forzar x(j) no decreciente (opcional)",
                value=enforce_default,
                key=f"mono_{var['id']}",
                help="Si está activo, al escribir un x más bajo que el anterior se ajusta automáticamente.",
            )

            for j in range(1, int(var["k"]) + 1):
                target = leftx if j % 2 == 1 else rightx
                current = float(var["x_values"][j - 1])

                new_x = target.number_input(
                    f"x para K = {j}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current),
                    step=0.01,
                    key=f"x_{var['id']}_{j}",
                )
                new_x = clamp01(new_x)

                if enforce_monotone and j > 1:
                    prev = float(var["x_values"][j - 2])
                    if new_x < prev:
                        new_x = prev

                var["x_values"][j - 1] = new_x

            # anclas globales
            if force_worst:
                var["x_values"][0] = 0.0
            if force_best:
                var["x_values"][-1] = 1.0

            if st.button("↺ Reset x(j) a rampa 0→1", key=f"resetx_{var['id']}"):
                k = int(var["k"])
                var["x_values"] = [round(i / (k - 1), 6) for i in range(k)]
                for j in range(1, k + 1):
                    st.session_state.pop(f"x_{var['id']}_{j}", None)
                st.rerun()

            var["notes"] = st.text_area(
                "Notas / criterio (opcional)",
                value=var.get("notes", ""),
                key=f"notes_{var['id']}",
                placeholder="Cómo decidimos la categorización, rangos, etc.",
            )

            scale = generate_scale_manual_x(
                peso_pct=float(peso_eff),
                k=int(var["k"]),
                x_values=var["x_values"],
            )
            df = scale_to_df(scale, var["labels"])

            st.caption(f"Impacto máximo de esta variable (si x=1): {peso_eff:.2f}% del score total")
            st.dataframe(df, use_container_width=True, hide_index=True)


st.divider()
st.subheader("Resumen del modelo")

peso_eff_by_id = get_effective_peso_pct_by_id(st.session_state.model)
summary = []
for v in st.session_state.model["variables"]:
    peso_raw = float(v.get("peso_pct", 0.0))
    peso_eff = float(peso_eff_by_id.get(v["id"], peso_raw))
    xs = [clamp01(x) for x in (v.get("x_values") or [])]
    summary.append(
        {
            "Variable": v["name"],
            "Peso raw %": round(peso_raw, 2),
            "Peso efectivo %": round(peso_eff, 2),
            "k": int(v["k"]),
            "x (preview)": " | ".join([f"{x:.2f}" for x in xs[:3]]) + (" ..." if len(xs) > 3 else ""),
        }
    )

st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
