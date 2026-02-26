# app.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


# =========================
# Tu lógica (intacta)
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


def generate_scale(peso_pct: float, k: int, xmin: Optional[float] = None) -> Dict:
    if k < 2:
        raise ValueError("k debe ser >= 2 (mínimo 2 categorías).")

    w = peso_pct / 100.0
    if xmin is None:
        xmin = xmin_by_weight(w)

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
        "xmin": xmin,
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
    # 👉 Aquí pones tu catálogo real. El peso NO se edita en la UI.
    return {
        "variables": [
            {
                "id": "antiguedad",
                "name": "Antigüedad 1ª contratación",
                "peso_pct": 7.5,
                "k": 3,
                "labels": ["", "", ""],
                "notes": "",
            },
            {
                "id": "vinculacion_ramos",
                "name": "Vinculación: Nº de Ramos con nosotros",
                "peso_pct": 7.5,
                "k": 3,
                "labels": ["0 ramos", "1-2 ramos", "3 o más ramos"],
                "notes": "",
            },
            {
                "id": "rentabilidad",
                "name": "Rentabilidad de la póliza actual",
                "peso_pct": 7.5,
                "k": 3,
                "labels": ["", "", ""],
                "notes": "",
            },
            {
                "id": "descuentos_recargos",
                "name": "Descuentos o Recargos aplicados sobre tarifa",
                "peso_pct": 5.5,
                "k": 3,
                "labels": ["", "", ""],
                "notes": "",
            },
            {
                "id": "morosidad",
                "name": "Morosidad: Históricos sin incidencia en devolución",
                "peso_pct": 5.0,
                "k": 3,
                "labels": ["", "", ""],
                "notes": "",
            },
            {
                "id": "engagement",
                "name": "Engagement comercial / Uso de canales propios",
                "peso_pct": 4.5,
                "k": 3,
                "labels": ["", "", ""],
                "notes": "",
            },
        ]
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
                "j": r.j,
                "Etiqueta (texto libre)": label,
                "x(j)": r.x,
                "Aporte % (w*x)": r.contribution_pct,
                "Δ vs prev %": r.delta_from_prev_pct,
            }
        )
    return pd.DataFrame(rows)


with st.sidebar:
    st.subheader("Controles")
    if st.button("↩️ Reset modelo (pierde cambios)"):
        st.session_state.model = init_model()
        st.rerun()

    st.divider()
    st.caption("Exporta el estado del taller (k, etiquetas, notas).")
    st.download_button(
        "⬇️ Descargar JSON del taller",
        data=pd.Series(st.session_state.model).to_json(force_ascii=False, indent=2).encode("utf-8"),
        file_name="taller_scoring.json",
        mime="application/json",
    )


vars_list = st.session_state.model["variables"]

# 2 columnas de tarjetas
col1, col2 = st.columns(2, gap="large")
cols = [col1, col2]

for i, var in enumerate(vars_list):
    normalize_labels(var)

    with cols[i % 2]:
        with st.container(border=True):
            st.subheader(var["name"])

            # Peso fijo: se muestra pero no se edita
            w = float(var["peso_pct"]) / 100.0
            xmin = xmin_by_weight(w)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Peso (%)", f"{var['peso_pct']:.1f}")
            m2.metric("w", f"{w:.4f}")
            m3.metric("xmin (auto)", f"{xmin:.2f}")
            # delta_max depende de k, lo sacamos tras generar scale (lo mostramos abajo)

            # Solo editas k
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

            # Etiquetas (texto libre por j)
            st.markdown("**Etiquetas por categoría (texto libre)**")
            left, right = st.columns(2)
            for j in range(1, int(var["k"]) + 1):
                target = left if j % 2 == 1 else right
                var["labels"][j - 1] = target.text_input(
                    f"j = {j}",
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

            # Cálculo
            scale = generate_scale(peso_pct=float(var["peso_pct"]), k=int(var["k"]), xmin=None)
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
