import numpy as np
import pandas as pd
import joblib
import streamlit as st
from scipy.optimize import differential_evolution

# ------------------------------------------------------------
# 0. 페이지 설정 (여백 줄이기: wide)
# ------------------------------------------------------------
st.set_page_config(
    page_title="ALD RECIPE",
    layout="wide"
)

# 전체 컨테이너 여백/폰트 살짝 조정 (선택)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 1. 물리 상수 (Colab과 동일)
# ============================================================
Q_ELECTRON = 1.602176634e-19
EPSILON_0  = 8.8541878128e-12
K_BOLTZMANN_EV = 8.617333262e-5
K_BOLTZMANN_J  = 1.380649e-23

V_BIAS   = 1.0
T_DEVICE = 300.0

P0_KW = 1.0
K_POWER = 0.005
T0_C = 200.0
EF_ELEC = 0.4541 / 3600000.0

GAMMA_GAS = 2e-7
F_PURGE_SCCM = 500.0
C_BASE = 0.5

N_CYCLE_MIN = 50
N_CYCLE_MAX = 500


# ============================================================
# 2. 막 별 파라미터 (Colab 최신 버전과 동일, G0 보정 포함)
# ============================================================
FILM_PARAMS = {
    "HfO2": {
        "G0": 0.2973,
        "E_act": 0.07,
        "alpha_A": 3.0,
        "alpha_B": 10.0,
        "k_cvd": 0.03,
        "beta_purge": 3.0,
        "t_B_const": 0.20,

        "epsilon_r": 22, "phi_t": 1.0, "phi_b": 1.58,
        "C_PF": 5e-7, "C_FN": 5e-8, "B_FN": 2.27e9,
        "gamma": 4.0, "tc": 3.5,

        "gamma_prec": 1.0e-3,

        "t_min_film": 2.0, "t_max_film": 8.0,

        "T_min_proc": 200.0, "T_max_proc": 350.0,
        "t_pulse_min": 0.30, "t_pulse_max": 1.00,
        "t_purge_min": 3.0,  "t_purge_max": 8.0,
        "dose_min": 0.5, "dose_max": 2.0,
    },

    "Al2O3": {
        "G0": 2.7546,
        "E_act": 0.06,
        "alpha_A": 4.0,
        "alpha_B": 15.0,
        "k_cvd": 0.02,
        "beta_purge": 3.5,
        "t_B_const": 0.05,

        "epsilon_r": 9.0, "phi_t": 1.3, "phi_b": 2.0,
        "C_PF": 3e-7, "C_FN": 4e-8, "B_FN": 2.0e9,
        "gamma": 4.0, "tc": 3.0,

        "gamma_prec": 8e-4,

        "t_min_film": 3.0, "t_max_film": 15.0,

        "T_min_proc": 150.0, "T_max_proc": 300.0,
        "t_pulse_min": 0.05, "t_pulse_max": 0.20,
        "t_purge_min": 1.0,  "t_purge_max": 5.0,
        "dose_min": 0.5, "dose_max": 2.0,
    },

    "TiO2": {
        "G0": 0.6464,
        "E_act": 0.08,
        "alpha_A": 2.5,
        "alpha_B": 10.0,
        "k_cvd": 0.04,
        "beta_purge": 2.5,
        "t_B_const": 0.20,

        "epsilon_r": 40.0, "phi_t": 0.9, "phi_b": 1.4,
        "C_PF": 6e-7, "C_FN": 6e-8, "B_FN": 2.0e9,
        "gamma": 4.0, "tc": 3.0,

        "gamma_prec": 1e-3,

        "t_min_film": 5.0, "t_max_film": 20.0,

        "T_min_proc": 200.0, "T_max_proc": 350.0,
        "t_pulse_min": 0.10, "t_pulse_max": 0.60,
        "t_purge_min": 2.0,  "t_purge_max": 8.0,
        "dose_min": 0.5, "dose_max": 2.0,
    },

    "SiO2": {
        "G0": 0.5666,
        "E_act": 0.07,
        "alpha_A": 3.5,
        "alpha_B": 15.0,
        "k_cvd": 0.05,
        "beta_purge": 2.5,
        "t_B_const": 0.20,

        "epsilon_r": 3.9, "phi_t": 1.6, "phi_b": 3.1,
        "C_PF": 2e-7, "C_FN": 3e-8, "B_FN": 2.2e9,
        "gamma": 4.0, "tc": 2.5,

        "gamma_prec": 1.5e-3,

        "t_min_film": 3.0,  "t_max_film": 15.0,

        "T_min_proc": 200.0, "T_max_proc": 350.0,
        "t_pulse_min": 0.30, "t_pulse_max": 1.00,
        "t_purge_min": 3.0,  "t_purge_max": 10.0,
        "dose_min": 0.5, "dose_max": 2.0,
    }
}


# ============================================================
# 3. 모델 로드
# ============================================================
@st.cache_resource
def load_model(film):
    fname = f"{film.lower()}_ald_rf_model.joblib"
    return joblib.load(fname)


# ============================================================
# 4. 최적화 목적 함수 (logJ 사용 반영)
# ============================================================
def objective(x, model, target_t, params):
    T, tA, tP, Nf, dose = x
    N = int(round(Nf))

    if N < N_CYCLE_MIN or N > N_CYCLE_MAX:
        return 1e12

    X = pd.DataFrame([[T, tA, tP, N, dose]],
                     columns=model.feature_names_in_)

    try:
        t_pred, J_log_pred, C_pred = model.predict(X)[0]
    except Exception:
        return 1e12

    if t_pred <= 0 or C_pred <= 0 or np.isnan(t_pred) or np.isnan(J_log_pred) or np.isnan(C_pred):
        return 1e12

    J_uA = 10**J_log_pred - 1.0
    if J_uA <= 0 or np.isnan(J_uA):
        return 1e12

    # 1) 두께 제약
    rel_err = abs(t_pred - target_t) / target_t
    tol = 0.05

    if rel_err > tol:
        return 1e8 * (rel_err - tol)**2 + 1e6 * rel_err

    # 2) 허용 오차 내에서 J, CO2 최적화
    w_j = 10.0
    w_c = 1.0

    loss = 0.0
    loss += w_j * J_log_pred
    loss += w_c * C_pred

    J_max = 10.0
    if J_uA > J_max:
        over = (J_uA - J_max) / J_max
        loss += 1e4 * over**2

    return loss


# ============================================================
# 5. 최적화 실행
# ============================================================
def run_opt(model, target_t, params):

    bounds = [
        (params["T_min_proc"], params["T_max_proc"]),
        (params["t_pulse_min"], params["t_pulse_max"]),
        (params["t_purge_min"], params["t_purge_max"]),
        (N_CYCLE_MIN, N_CYCLE_MAX),
        (params["dose_min"], params["dose_max"]),
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(model, target_t, params),
        maxiter=60,
        popsize=14,
        polish=True,
    )

    T, tA, tP, Nf, dose = result.x
    N = int(round(Nf))

    X = pd.DataFrame([[T, tA, tP, N, dose]],
                     columns=model.feature_names_in_)
    t_pred, J_log_pred, C_pred = model.predict(X)[0]
    J_uA = 10**J_log_pred - 1.0

    return {
        "T_proc_C": T,
        "t_A_pulse_s": tA,
        "t_B_pulse_s": params["t_B_const"],
        "t_purge_s": tP,
        "N_cycle": N,
        "dose_scc": dose,
        "t_film_nm": t_pred,
        "J_leak_uA_cm2": J_uA,
        "J_leak_log10": J_log_pred,
        "CO2eq_kg": C_pred,
        "cost": result.fun,
    }


# ============================================================
# 6. UI
# ============================================================
def main():
    st.title("ALD RECIPE")

    film = st.sidebar.selectbox(
        "막 선택",
        ["HfO2", "Al2O3", "TiO2", "SiO2"]
    )
    params = FILM_PARAMS[film]

    target_t = st.sidebar.slider(
        "목표 두께 (nm)",
        float(params["t_min_film"]),
        float(params["t_max_film"]),
        float((params["t_min_film"] + params["t_max_film"]) / 2),
        0.1,
    )

    model = load_model(film)

    if st.button("최적화 실행"):
        with st.spinner("Searching optimal recipe..."):
            res = run_opt(model, target_t, params)

        # 성공 메세지 + 효과
        st.success(f"{film} 최적화 완료!")
        st.balloons()
        st.markdown(
            "<div style='text-align:center; font-size:64px; margin-top:-30px;'>⚡⚡⚡</div>",
            unsafe_allow_html=True,
        )

        st.metric("목표 두께", f"{target_t:.3f} nm")
        st.metric("예측 두께", f"{res['t_film_nm']:.3f} nm")
        st.metric("누설 전류 (µA/cm²)", f"{res['J_leak_uA_cm2']:.3e}")
        st.metric("CO2eq (kg)", f"{res['CO2eq_kg']:.3f}")

        st.write("### 최적 공정 조건")

        # 예쁜 표용 데이터프레임 구성
        df = pd.DataFrame([res]).copy()

        # 표시할 컬럼만 선택 (index / logJ / cost 제거)
        show_cols = [
            "T_proc_C",
            "t_A_pulse_s",
            "t_B_pulse_s",
            "t_purge_s",
            "N_cycle",
            "dose_scc",
            "t_film_nm",
            "J_leak_uA_cm2",
            "CO2eq_kg",
        ]
        df = df[show_cols]

        # 헤더 한글+단위로 변경
        df.columns = [
            "공정 온도 (°C)",
            "A 펄스 (s)",
            "B 펄스 (s)",
            "퍼지 시간 (s)",
            "사이클 수",
            "Dose (scc)",
            "막 두께 (nm)",
            "누설 전류 (µA/cm²)",
            "CO2eq (kg)",
        ]

        # 인덱스 제거하고 HTML 테이블로 렌더링
        html_table = df.to_html(
            index=False,
            float_format=lambda x: f"{x:0.4g}",
            border=0,
            classes="dataframe",
        )
        st.markdown(html_table, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
