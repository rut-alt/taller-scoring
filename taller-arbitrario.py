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
    x: float  # aquí lo usamos como p(j) normalizado (share)
    contribution: float
    contribution_pct: float
    delta_from_prev: float
    delta_from_prev_pct: float


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_to_sum1(values: List[float]) -> List[float]:
    """Normaliza lista no-negativa para que sume 1. Si suma 0, reparte uniforme."""
    vals = [max(0.0, float(v)) for v in (values or [])]
    s = sum(vals)
    if s <= 0:
        n = len(vals)
        if n == 0:
            return []
        return [1.0 / n] * n
    return [v / s for v in vals]


def generate_scale_manual_p_sum1(
    w: float,
    k: int,
    p_values: List[float],
    auto_normalize: bool = True,
) -> Dict:
    """
    Escala con p(j) manual (shares) que SUMAN 1 por variable.
    - p_values: lista longitud k con valores en [0,1]
    - si auto_normalize=True, se normaliza internamente para que sumen 1
    - contribution(j) = w * p(j)
    - delta(j) = contribution(j) - contribution(j-1)
    """
    if k < 2:
        raise ValueError("k debe ser >= 2.")
    w = clamp01(w)

    ps = list(p_values or [])
    if len(ps) < k:
        ps += [0.0] * (k - len(ps))
    if len(ps) > k:
        ps = ps[:k]
    ps = [clamp01(p) for p in ps]

    sum_raw = sum(ps)
    ps_eff = normalize_to_sum1(ps) if auto_normalize else ps
    sum_eff = sum(ps_eff)

    results: List[CategoryResult] = []
    prev_contrib = 0.0

    for j in range(1, k + 1):
        p = ps_eff[j - 1]
        contrib = w * p
        delta = contrib - prev_contrib if j > 1 else 0.0

        results.append(
            CategoryResult(
                j=j,
                x=round(p, 6),
                contribution=round(contrib, 6),
                contribution_pct=round(contrib * 100.0, 4),
                delta_from_prev=round(delta, 6),
                delta_from_prev_pct=round(delta * 100.0, 4),
            )
        )
        prev_contrib = contrib

    # aquí delta_max no tiene mucho sentido como antes (porque no es una escala 0..1),
    # pero podemos reportar el máximo aporte de una categoría
    max_contrib = max((r.contribution for r in results), default=0.0)

    return {
        "w": float(w),
        "peso_pct": float(w * 100.0),
        "k": int(k),
        "sum_p_raw": float(sum_raw),
        "sum_p_eff": float(sum_eff),
        "auto_normalize_p": bool(auto_normalize),
        "max_contribution": float(max_contrib),
        "max_contribution_pct": round(max_contrib * 100.0, 4),
        "categories": results,
        "p_values_raw": ps,
        "p_values_eff": ps_eff,
    }


# =========================
# App
# =========================

st.set_page_config(page_title="Taller scoring por tarjetas", layout="wide")
st.title("Taller scoring — Tarjetas por variable (k + peso + p(j) que suma 1)")


def init_model():
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
        # p(j) por defecto uniforme (suma 1)
        preset_p = [1 / 3, 1 / 3, 1 / 3]

        if "Nº de Ramos con nosotros" in name:
            preset_labels = ["0 ramos", "1-2 ramos", "3 o más ramos"]
            preset_p = [0.2, 0.3, 0.5]  # ejemplo, suma 1

        variables.append(
            {
                "id": f"var_{idx:02d}",
                "name": name,
                "w": float(peso_pct) / 100.0,
                "k": 3,
                "labels": preset_labels,
                "p_values": preset_p,  # <-- ahora usamos p_values (shares)
                "notes": "",
            }
        )

    return {
        "variables": variables,
        "settings": {
            "normalize_weights": False,
            "auto_normalize_p": True,  # normaliza p para que sume 1
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


def normalize_p_values(var: dict):
    """Asegura longitud k en p_values; por defecto uniforme si faltan."""
    k = int(var["k"])
    ps = list(var.get("p_values") or [])
    if len(ps) < k:
        ps += [0.0] * (k - len(ps))
    if len(ps) > k:
        ps = ps[:k]
    # clamp
    ps = [clamp01(p) for p in ps]
    # si está auto-normalize, lo dejamos en estado "raw" pero con longitud correcta;
    # la normalización real se aplica en el cálculo.
    var["p_values"] = ps


def scale_to_df(scale: dict, labels: List[str]) -> pd.DataFrame:
    rows = []
    for idx, r in enumerate(scale["categories"]):
        label = labels[idx] if idx < len(labels) else ""
        rows.append(
            {
                "K (j)": r.j,
                "Etiqueta (texto libre)": label,
                "p(j) (suma 1)": r.x,
                "Suma al score total % (w*p)": r.contribution_pct,
                "Δ vs prev %": r.delta_from_prev_pct,
            }
        )
    return pd.DataFrame(rows)


def clear_all(model: dict, default_k: int = 3) -> dict:
    """Borra etiquetas/notas y resetea k y p-values (uniforme). No toca los pesos."""
    for v in model.get("variables", []):
        var_id = v["id"]
        v["k"] = int(default_k)
        v["labels"] = [""] * int(default_k)
        v["p_values"] = [1.0 / default_k] * default_k
        v["notes"] = ""

        if "Nº de Ramos con nosotros" in v["name"]:
            v["labels"] = ["0 ramos", "1-2 ramos", "3 o más ramos"]
            v["p_values"] = [0.2, 0.3, 0.5]

        st.session_state.pop(f"k_{var_id}", None)
        st.session_state.pop(f"notes_{var_id}", None)
        st.session_state.pop(f"w_{var_id}", None)
        st.session_state.pop(f"pauto_{var_id}", None)
        for j in range(1, 11):
            st.session_state.pop(f"lbl_{var_id}_{j}", None)
            st.session_state.pop(f"p_{var_id}_{j}", None)

    return model


def randomize_model(model: dict, k_min: int = 2, k_max: int = 6) -> dict:
    demo_words = ["Peor", "Bajo", "Medio", "Alto", "Mejor", "Top", "Ok", "Riesgo", "Premium", "Básico"]
    for v in model.get("variables", []):
        k = random.randint(k_min, k_max)
        v["k"] = k
        v["labels"] = [f"{random.choice(demo_words)} {i+1}" for i in range(k)]
        v["notes"] = f"Auto-demo ({k} categorías). Reemplazar en el taller."

        ps = [random.random() for _ in range(k)]
        ps = normalize_to_sum1(ps)
        v["p_values"] = [round(p, 4) for p in ps]

        if "Nº de Ramos con nosotros" in v["name"]:
            v["k"] = 3
            v["labels"] = ["0 ramos", "1-2 ramos", "3 o más ramos"]
            v["p_values"] = [0.2, 0.3, 0.5]

    return model


def get_effective_weights(model: dict) -> Dict[str, float]:
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

    st.session_state.model["settings"]["normalize_weights"] = st.toggle(
        "Normalizar pesos (que sumen 1)",
        value=bool(st.session_state.model["settings"].get("normalize_weights", False)),
        help="Si activas esto, el cálculo usa w_normalizado = w / suma(w). No cambia el valor guardado.",
    )

    st.session_state.model["settings"]["auto_normalize_p"] = st.toggle(
        "Auto-normalizar p(j) para que sume 1 (por variable)",
        value=bool(st.session_state.model["settings"].get("auto_normalize_p", True)),
        help="Si está activo, aunque introduzcas p que no suma 1, se reescala internamente.",
    )

    raw_sum = sum(clamp01(float(v.get("w", 0.0))) for v in st.session_state.model["variables"])
    st.metric("Suma pesos (introducidos)", f"{raw_sum:.4f}")

    cA, cB = st.columns(2)
    with cA:
        if st.button("Aleatorio (demo)", use_container_width=True):
            st.session_state.model = randomize_model(st.session_state.model)
            st.rerun()

    with cB:
        if st.button("Borrar todo (k/etiquetas/p/notas)", use_container_width=True):
            st.session_state.model = clear_all(st.session_state.model)
            st.rerun()

    st.divider()

    st.subheader("Leyenda")
    st.markdown(
        """
**p(j)**  
Share interno por categoría **que suma 1** dentro de cada variable.

**Suma al score total % (w*p)**  
Aporte de esa categoría al score total. (Si w está normalizado, el total máximo queda acotado.)
"""
    )

    st.divider()

    if st.button("↩️ Reset modelo (pierde cambios)", use_container_width=True):
        st.session_state.model = init_model()
        st.rerun()

    st.divider()

    st.caption("Exporta el estado del taller (w, k, p_values, etiquetas, notas + normalización).")
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
auto_norm_p_global = bool(st.session_state.model["settings"].get("auto_normalize_p", True))

col1, col2 = st.columns(2, gap="large")
cols = [col1, col2]

for i, var in enumerate(vars_list):
    normalize_labels(var)
    normalize_p_values(var)

    with cols[i % 2]:
        with st.container(border=True):
            st.subheader(var["name"])

            w_raw = clamp01(float(var.get("w", 0.0)))
            w_eff = clamp01(float(w_effective_by_id.get(var["id"], w_raw)))

            m1, m2, m3 = st.columns(3)
            m1.metric("w (introducido)", f"{w_raw:.4f}")
            m2.metric(
                "w (efectivo)",
                f"{w_eff:.4f}" + (" (norm.)" if st.session_state.model["settings"]["normalize_weights"] else ""),
            )
            m3.metric("Peso (%) efectivo", f"{(w_eff * 100.0):.2f}%")

            var["w"] = float(
                st.slider(
                    "Peso w (0 a 1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(w_raw),
                    step=0.01,
                    key=f"w_{var['id']}",
                )
            )

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
            normalize_p_values(var)

            st.markdown("**Etiquetas por categoría (texto libre)**")
            left, right = st.columns(2)
            for j in range(1, int(var["k"]) + 1):
                target = left if j % 2 == 1 else right
                var["labels"][j - 1] = target.text_input(
                    f"K = {j}",
                    value=var["labels"][j - 1],
                    key=f"lbl_{var['id']}_{j}",
                )

            st.markdown("**p(j) manual (shares que suman 1)**")
            leftp, rightp = st.columns(2)

            # toggle local (hereda el global)
            auto_norm_local = st.checkbox(
                "Auto-normalizar p(j) (en esta variable)",
                value=auto_norm_p_global,
                key=f"pauto_{var['id']}",
                help="Si está activo, se reescala para que sume 1 aunque lo introducido no sume 1.",
            )

            for j in range(1, int(var["k"]) + 1):
                target = leftp if j % 2 == 1 else rightp
                current = float(var["p_values"][j - 1])

                new_p = target.number_input(
                    f"p para K = {j}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(current),
                    step=0.01,
                    key=f"p_{var['id']}_{j}",
                )
                var["p_values"][j - 1] = clamp01(new_p)

            if st.button("↺ Reset p(j) uniforme", key=f"resetp_{var['id']}"):
                k = int(var["k"])
                var["p_values"] = [1.0 / k] * k
                for j in range(1, k + 1):
                    st.session_state.pop(f"p_{var['id']}_{j}", None)
                st.rerun()

            var["notes"] = st.text_area(
                "Notas / criterio (opcional)",
                value=var.get("notes", ""),
                key=f"notes_{var['id']}",
            )

            w_effective_by_id = get_effective_weights(st.session_state.model)
            w_eff = float(w_effective_by_id.get(var["id"], clamp01(float(var.get("w", 0.0)))))

            scale = generate_scale_manual_p_sum1(
                w=float(w_eff),
                k=int(var["k"]),
                p_values=var["p_values"],
                auto_normalize=bool(auto_norm_local),
            )
            df = scale_to_df(scale, var["labels"])

            # Feedback de suma p
            sum_raw = scale["sum_p_raw"]
            sum_eff = scale["sum_p_eff"]
            if bool(auto_norm_local):
                st.caption(f"Suma p(j) introducida: {sum_raw:.4f} → (normalizada) {sum_eff:.4f}")
            else:
                if abs(sum_raw - 1.0) > 1e-6:
                    st.warning(f"⚠️ p(j) NO suma 1 (suma={sum_raw:.4f}). Activa auto-normalizar o ajusta valores.")
                else:
                    st.caption("p(j) suma 1 ✅")

            st.caption(f"Aporte máximo de una categoría en esta variable: {scale['max_contribution_pct']:.2f}% del score total")
            st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Resumen del modelo (solo lo que cambia en el taller)")

summary = []
w_effective_by_id = get_effective_weights(st.session_state.model)

for v in st.session_state.model["variables"]:
    w_raw = clamp01(float(v.get("w", 0.0)))
    w_eff = clamp01(float(w_effective_by_id.get(v["id"], w_raw)))
    ps = [clamp01(p) for p in (v.get("p_values") or [])]
    summary.append(
        {
            "Variable": v["name"],
            "w (raw)": round(w_raw, 4),
            "w (efectivo)": round(w_eff, 4),
            "Peso % (efectivo)": round(w_eff * 100.0, 2),
            "k": int(v["k"]),
            "p (preview)": " | ".join([f"{p:.2f}" for p in ps[:3]]) + (" ..." if len(ps) > 3 else ""),
            "Etiquetas (preview)": " | ".join([lab for lab in v["labels"] if lab][:3]) + (" ..." if len(v["labels"]) > 3 else ""),
        }
    )

st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
