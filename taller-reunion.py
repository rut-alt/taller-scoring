from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class CategoryResult:
    j: int
    x: float
    contribution_pct: float
    delta_from_prev_pct: float


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_list_len(values: List, n: int, fill=0.0) -> List:
    vals = list(values or [])
    if len(vals) < n:
        vals += [fill] * (n - len(vals))
    if len(vals) > n:
        vals = vals[:n]
    return vals


def gaps_to_x(k: int, gaps: List[float], cap_mode: str = "clip") -> Dict:
    """
    Construye x(j) con x(k)=1 y gaps entre categorías.

    - x(mejor) = 1
    - sum(gaps) <= 1 idealmente
    - si sum(gaps) > 1:
        * clip: recorta el último gap
        * scale: reescala todos proporcionalmente
    """
    if k < 2:
        raise ValueError("k debe ser >= 2.")

    gaps = normalize_list_len(gaps, k - 1, fill=0.0)
    gaps = [max(0.0, float(g)) for g in gaps]
    s = sum(gaps)

    if s > 1.0 + 1e-12:
        if cap_mode == "scale":
            gaps = [g / s for g in gaps]
        else:
            excess = s - 1.0
            gaps[-1] = max(0.0, gaps[-1] - excess)

    s_eff = sum(gaps)
    remaining = max(0.0, 1.0 - s_eff)

    xs = [0.0] * k
    xs[-1] = 1.0
    acc = 0.0
    for idx in range(k - 2, -1, -1):
        acc += gaps[idx]
        xs[idx] = clamp01(1.0 - acc)
    xs[-1] = 1.0

    return {
        "x_values": xs,
        "gaps_eff": gaps,
        "sum_gaps": s_eff,
        "remaining": remaining,
    }


def generate_scale_fixed_weight(peso_pct: float, x_values: List[float]) -> Dict:
    k = len(x_values)
    xs = [clamp01(x) for x in x_values]

    results: List[CategoryResult] = []
    prev = 0.0
    for j in range(1, k + 1):
        x = xs[j - 1]
        contrib = float(peso_pct) * x
        delta = contrib - prev if j > 1 else 0.0
        results.append(
            CategoryResult(
                j=j,
                x=round(x, 6),
                contribution_pct=round(contrib, 4),
                delta_from_prev_pct=round(delta, 4),
            )
        )
        prev = contrib

    return {"peso_pct": float(peso_pct), "k": k, "categories": results, "x_values": xs}


def scale_to_df(scale: dict, labels: List[str]) -> pd.DataFrame:
    rows = []
    for idx, r in enumerate(scale["categories"]):
        rows.append(
            {
                "K (j)": r.j,
                "Etiqueta": labels[idx] if idx < len(labels) else "",
                "x(j)": r.x,
                "Suma al score total % (peso*x)": r.contribution_pct,
                "Δ vs prev %": r.delta_from_prev_pct,
            }
        )
    return pd.DataFrame(rows)


def build_fixed_model() -> Dict:
    return {
        "variables": [
            {
                "id": "var_01",
                "name": "Antigüedad 1ª contratación",
                "peso_pct": 7.5,
                "k": 5,
                "labels": ["<= 2 años", "<= 5 años", "<= 7 años", "<= 9 años", ">= 10 años"],
                "gaps": [0.3, 0.52, 0.18],
                "notes": "",
            },
            {
                "id": "var_02",
                "name": "Vinculación: Nº de Ramos con nosotros",
                "peso_pct": 7.5,
                "k": 3,
                "labels": ["1 ramo", "2 ramos", "3 o más ramos"],
                "gaps": [0.4, 0.6],
                "notes": "",
            },
            {
                "id": "var_03",
                "name": "Rentabilidad de la póliza actual",
                "peso_pct": 7.5,
                "k": 3,
                "labels": ["negativo", "bajo", "medio-alto"],
                "gaps": [0.5, 0.0],
                "notes": "",
            },
            {
                "id": "var_04",
                "name": "Descuentos o Recargos aplicados sobre tarifa",
                "peso_pct": 5.5,
                "k": 3,
                "labels": ["con descuento", "sin descuento", "recargo"],
                "gaps": [0.5, 0.5],
                "notes": "",
            },
            {
                "id": "var_05",
                "name": "Morosidad: Históricos sin incidencia en devolución (Anotaciones de póliza)",
                "peso_pct": 5.0,
                "k": 3,
                "labels": ["3 o más impagos", "Hasta 2 impagos", "sin impagos"],
                "gaps": [0.5, 0.5],
                "notes": "En los ýltimos 24 meses",
            },
            {
                "id": "var_06",
                "name": "Engagement comercial / Uso de canales propios (App / Área cliente / Web privada)",
                "peso_pct": 4.5,
                "k": 2,
                "labels": ["No resgistro", "Si registro"],
                "gaps": [0.5],
                "notes": "",
            },
            {
                "id": "var_07",
                "name": "Frecuencia uso de coberturas complementarias que no emiten siniestralidad.",
                "peso_pct": 4.5,
                "k": 2,
                "labels": ["No", "Si"],
                "gaps": [0.5],
                "notes": "Las usa o no las usa",
            },
            {
                "id": "var_08",
                "name": "Total de asegurados en el Total de sus pólizas - Media de asegurados por póliza",
                "peso_pct": 4.5,
                "k": 4,
                "labels": ["1", ">=3", ">=5", ">=8"],
                "gaps": [0.5, 0.5, 0.0],
                "notes": "",
            },
            {
                "id": "var_09",
                "name": "Edad",
                "peso_pct": 4.5,
                "k": 3,
                "labels": ["> 52 años", "> 43 años", "< 43 años"],
                "gaps": [0.5, 0.5],
                "notes": "",
            },
            {
                "id": "var_10",
                "name": "Rentabilidad de la póliza retrospectivo/histórica (LTV)",
                "peso_pct": 4.5,
                "k": 4,
                "labels": ["negativo", "bajo", "medio", "alto"],
                "gaps": [0.5, 0.5, 0.0],
                "notes": "",
            },
            {
                "id": "var_11",
                "name": "Tipo de distribución",
                "peso_pct": 4.5,
                "k": 4,
                "labels": ["Corredores", "Agentes", "PB", "Venta directa"],
                "gaps": [0.55, 0.5, 0.0],
                "notes": "",
            },
            {
                "id": "var_12",
                "name": "Vinculación: Coberturas complementarias opcionales",
                "peso_pct": 4.5,
                "k": 3,
                "labels": ["Sin", "Con", "+ 2 (o Solo Accidentes)"],
                "gaps": [0.5, 0.5],
                "notes": "",
            },
            {
                "id": "var_13",
                "name": "Contactabilidad: Más de X Campos de datos (Tiene App, Teléfono, Mail, etc.).",
                "peso_pct": 4.0,
                "k": 4,
                "labels": ["Nada", "Teléfono", "Mail", "Mail + Teléfono"],
                "gaps": [0.5, 0.5, 0.0],
                "notes": "",
            },
            {
                "id": "var_14",
                "name": "Edad del asegurado más mayor",
                "peso_pct": 4.0,
                "k": 3,
                "labels": [">= 70 años", ">= 55 años", "<=54"],
                "gaps": [0.5, 0.5],
                "notes": "",
            },
            {
                "id": "var_15",
                "name": "Vinculación familiar",
                "peso_pct": 3.0,
                "k": 2,
                "labels": ["No", "Si"],
                "gaps": [0.5],
                "notes": "",
            },
            {
                "id": "var_16",
                "name": "Prescriptor",
                "peso_pct": 3.0,
                "k": 3,
                "labels": ["No", "Si", "Si + 2 | Apostol"],
                "gaps": [0.5, 0.0],
                "notes": "",
            },
            {
                "id": "var_17",
                "name": "Exposición a comunicaciones de marca (RRSS, mailing…)",
                "peso_pct": 3.0,
                "k": 2,
                "labels": ["No", "Si"],
                "gaps": [0.5],
                "notes": "",
            },
            {
                "id": "var_18",
                "name": "Descendencia",
                "peso_pct": 3.0,
                "k": 2,
                "labels": ["No", "Si"],
                "gaps": [0.5],
                "notes": "",
            },
            {
                "id": "var_19",
                "name": "Medio de pago",
                "peso_pct": 2.5,
                "k": 3,
                "labels": ["Efectivo", "Banco (cuidado EFEB)", "Tarjeta"],
                "gaps": [0.5, 0.5],
                "notes": "",
            },
            {
                "id": "var_20",
                "name": "Frecuencia de pago (Periodicidad)",
                "peso_pct": 2.0,
                "k": 4,
                "labels": ["Mensual o Bimestral", "Trimestral", "Semestral", "Anual o Única o Senior 3"],
                "gaps": [0.5, 0.5, 0.0],
                "notes": "",
            },
            {
                "id": "var_21",
                "name": "Probabilidad de desglose",
                "peso_pct": 1.5,
                "k": 2,
                "labels": ["No", "Si"],
                "gaps": [0.5],
                "notes": "",
            },
            {
                "id": "var_22",
                "name": "Tipo de producto",
                "peso_pct": 1.5,
                "k": 3,
                "labels": ["Bajo: Resto", "Medio: Zero+ o Plus ConCo", "Alto: TAR 75 o Nivelado o Plus SinCo"],
                "gaps": [0.5, 0.5],
                "notes": "",
            },
            {
                "id": "var_23",
                "name": "NPS",
                "peso_pct": 1.5,
                "k": 3,
                "labels": ["0-5", "6-8", "9-10"],
                "gaps": [0.5, 0.5],
                "notes": "",
            },
            {
                "id": "var_24",
                "name": "Mascotas",
                "peso_pct": 1.5,
                "k": 2,
                "labels": ["No", "Si"],
                "gaps": [0.5],
                "notes": "",
            },
            {
                "id": "var_25",
                "name": "Localización (enfocado a potencial de compra)",
                "peso_pct": 1.5,
                "k": 2,
                "labels": ["Resto", "Grandes ciudades"],
                "gaps": [0.5],
                "notes": "",
            },
            {
                "id": "var_26",
                "name": "Autónomo",
                "peso_pct": 1.0,
                "k": 2,
                "labels": ["No", "Si"],
                "gaps": [0.5],
                "notes": "",
            },
            {
                "id": "var_27",
                "name": "Siniestralidad (Salud)",
                "peso_pct": 1.0,
                "k": 4,
                "labels": ["No tiene póliza", "> 90%", "> 80%", "<=80%"],
                "gaps": [0.5, 0.5, 0.0],
                "notes": "",
            },
            {
                "id": "var_28",
                "name": "Grado de digitalización de la póliza",
                "peso_pct": 0.5,
                "k": 2,
                "labels": ["No Firma Digital", "Firma Digital"],
                "gaps": [0.5],
                "notes": "",
            },
            {
                "id": "var_29",
                "name": "Profesión",
                "peso_pct": 0.5,
                "k": 4,
                "labels": ["Paro", "Con Riesgo", "Sin Riesgo", "Jubilado"],
                "gaps": [0.5, 0.5, 0.0],
                "notes": "",
            },
            {
                "id": "var_30",
                "name": "Nivel educativo",
                "peso_pct": 0.5,
                "k": 3,
                "labels": ["Sin estudios", "Medios", "Superiores"],
                "gaps": [0.5, 0.5],
                "notes": "",
            },
            {
                "id": "var_31",
                "name": "Sexo",
                "peso_pct": 0.0,
                "k": 3,
                "labels": ["", "", ""],
                "gaps": [0.5, 0.5],
                "notes": "",
            },
        ],
        "settings": {"cap_mode": "clip"},
    }


def normalize_gaps(var: dict):
    k = int(var["k"])
    gaps = normalize_list_len(var.get("gaps") or [], k - 1, fill=0.0)
    var["gaps"] = [max(0.0, float(g)) for g in gaps]


def init_session_state():
    if "model" not in st.session_state:
        st.session_state.model = build_fixed_model()

    if "cap_mode" not in st.session_state:
        st.session_state.cap_mode = st.session_state.model["settings"].get("cap_mode", "clip")

    for var in st.session_state.model["variables"]:
        normalize_gaps(var)
        for t in range(1, int(var["k"])):
            key = f"gap_{var['id']}_{t}"
            if key not in st.session_state:
                st.session_state[key] = float(var["gaps"][t - 1])


def export_current_model() -> Dict:
    export_model = {
        "variables": [],
        "settings": {"cap_mode": st.session_state.cap_mode},
    }

    for var in st.session_state.model["variables"]:
        normalize_gaps(var)
        current_gaps = []
        for t in range(1, int(var["k"])):
            current_gaps.append(float(st.session_state[f"gap_{var['id']}_{t}"]))

        export_model["variables"].append(
            {
                "id": var["id"],
                "name": var["name"],
                "peso_pct": float(var["peso_pct"]),
                "k": int(var["k"]),
                "labels": list(var["labels"]),
                "gaps": current_gaps,
                "notes": var.get("notes", ""),
            }
        )

    return export_model


# =========================
# Streamlit App
# =========================

st.set_page_config(page_title="Taller reunión", layout="wide")
st.title("Taller reunión — variables fijas y solo gaps editables")

init_session_state()

with st.sidebar:
    st.image("LOGOTIPO-AES-05.png", use_container_width=True)
    st.subheader("Controles globales")

    st.session_state.cap_mode = st.selectbox(
        "Si te pasas del saldo (Σ gaps > 1):",
        options=["clip", "scale"],
        index=0 if st.session_state.cap_mode == "clip" else 1,
        help=(
            "clip = recorta el último gap para que Σ=1.\n"
            "scale = reescala todos los gaps proporcionalmente para que Σ=1."
        ),
    )

    total_pesos = sum(float(v.get("peso_pct", 0.0)) for v in st.session_state.model["variables"])
    st.metric("Suma pesos (%)", f"{total_pesos:.2f}%")
    st.caption("Peso, K y etiquetas fijos. Solo se pueden modificar los gaps.")

    st.divider()
    export_payload = export_current_model()
    st.download_button(
        "⬇️ Descargar JSON del taller",
        data=json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="taller-reunion.json",
        mime="application/json",
        use_container_width=True,
    )


vars_list = st.session_state.model["variables"]
col1, col2 = st.columns(2, gap="large")
cols = [col1, col2]

for i, var in enumerate(vars_list):
    with cols[i % 2]:
        with st.container(border=True):
            st.subheader(var["name"])
            st.caption("Peso, K y etiquetas fijos. Solo gaps editables.")

            st.number_input(
                "Peso (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(var["peso_pct"]),
                step=0.5,
                disabled=True,
                key=f"peso_view_{var['id']}",
            )

            st.number_input(
                "k (nº de subcategorías)",
                min_value=2,
                max_value=10,
                value=int(var["k"]),
                step=1,
                disabled=True,
                key=f"k_view_{var['id']}",
            )

            st.markdown("**Etiquetas por categoría**")
            left, right = st.columns(2)
            for j in range(1, int(var["k"]) + 1):
                target = left if j % 2 == 1 else right
                target.text_input(
                    f"K = {j}",
                    value=var["labels"][j - 1],
                    disabled=True,
                    key=f"lbl_view_{var['id']}_{j}",
                )

            st.markdown("**Reparto del saldo (gaps) — x(mejor)=1 fijo**")
            st.caption("Los gaps son las caídas entre categorías al bajar de nivel. Σ gaps ≤ 1.")

            raw_gaps = []
            lg, rg = st.columns(2)
            for t in range(1, int(var["k"])):
                target = lg if t % 2 == 1 else rg
                gap_key = f"gap_{var['id']}_{t}"
                value = target.number_input(
                    f"gap {t} (caída entre K={t} y K={t+1})",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    key=gap_key,
                    disabled=False,
                )
                raw_gaps.append(float(value))

            sum_raw = sum(raw_gaps)
            if sum_raw > 1.0 + 1e-12:
                st.warning(
                    f"⚠️ Te has pasado de saldo: Σ gaps = {sum_raw:.3f} (> 1). "
                    f"Se aplicará el modo '{st.session_state.cap_mode}' para ajustarlo."
                )

            conv = gaps_to_x(
                k=int(var["k"]),
                gaps=raw_gaps,
                cap_mode=st.session_state.cap_mode,
            )
            xs = conv["x_values"]

            eps = 1e-6
            x_min = min(xs)
            discr_range = 1.0 - x_min

            if discr_range >= 0.80:
                discr_label = "muy discriminatoria"
            elif discr_range >= 0.50:
                discr_label = "medianamente discriminatoria"
            else:
                discr_label = "poco discriminatoria"

            if conv["sum_gaps"] < 1.0 - eps:
                st.warning(
                    "🟡 **Variable menos discriminatoria (Σ gaps < 1)**\n\n"
                    f"- **Σ gaps usado** = {conv['sum_gaps']:.3f} → **saldo sin usar** = {conv['remaining']:.3f}\n"
                    f"- **Rango real de discriminación** Δx = x_max − x_min = 1 − {x_min:.3f} = **{discr_range:.3f}**\n"
                    f"- Interpretación: la variable es **{discr_label}**.\n\n"
                    "📐 **Nota:** en este esquema Δx = Σ gaps, porque x(mejor)=1 y x(peor)=1−Σ gaps.\n\n"
                    "**No recomendado** si buscas máxima separación: lo habitual es ajustar a Σ gaps = 1 "
                    "para que la peor categoría sea x=0 y la variable use toda la escala 0–1."
                )

            st.info(
                f"Saldo usado: {conv['sum_gaps']:.3f} | "
                f"Saldo restante: {conv['remaining']:.3f} | "
                f"x(mejor)=1"
            )

            scale = generate_scale_fixed_weight(
                peso_pct=float(var["peso_pct"]),
                x_values=xs,
            )
            df = scale_to_df(scale, var["labels"])

            st.caption(f"Máximo de la variable (mejor): {float(var['peso_pct']):.2f}% (porque x=1)")
            st.dataframe(df, use_container_width=True, hide_index=True)


st.divider()
st.subheader("Resumen del modelo")

summary = []
for var in st.session_state.model["variables"]:
    current_gaps = [float(st.session_state[f"gap_{var['id']}_{t}"]) for t in range(1, int(var["k"]))]
    conv_s = gaps_to_x(
        k=int(var["k"]),
        gaps=current_gaps,
        cap_mode=st.session_state.cap_mode,
    )
    xs_s = conv_s["x_values"]

    summary.append(
        {
            "Variable": var["name"],
            "Peso %": round(float(var["peso_pct"]), 2),
            "k": int(var["k"]),
            "Etiquetas": " | ".join([str(lbl) for lbl in var["labels"]]),
            "Σ gaps": round(conv_s["sum_gaps"], 3),
            "Saldo restante": round(conv_s["remaining"], 3),
            "x (preview)": " | ".join([f"{x:.2f}" for x in xs_s]),
        }
    )

st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
