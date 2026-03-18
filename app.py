import streamlit as st
import pandas as pd
import joblib
import lightgbm  # 必须导入以确保joblib能正确加载LGBM模型
import shap
import matplotlib.pyplot as plt

# --- 页面基础配置 ---
st.set_page_config(
    page_title="VTE风险预测与解释系统(Alfafa-sepsis-vte)",
    page_icon="🩸",
    layout="wide"
)

# --- 模型加载 ---
@st.cache_resource  # 使用缓存，避免每次都重新加载模型
def load_model(path):
    """加载 .joblib 格式的模型"""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"错误：模型文件 '{path}' 未找到。")
        st.error(f"请确保 '{path}' 文件与您的Streamlit应用在同一个目录下。")
        return None
    except Exception as e:
        st.error(f"加载模型时发生未知错误: {e}")
        return None

# 加载您训练好的LightGBM模型
lgbm_model = load_model('LightGBM.joblib')

# --- 特征定义 ---
# 定义模型训练时使用的完整特征列
FEATURE_COLUMNS = [
    "vte_history", "cancer", "respiratory_failure", "heart_failure", "albumin_max",
    "creatinine_max", "inr_min", "pt_min", "alt_max", "fresh_frozen_plasma_input",
    "platelets_input", "rbw_input", "vasopressin", "sedative", "cvc"
]

# 将特征分为数值型和二元（是/否）型
NUMERIC_FEATURES = [
    "albumin_max", "creatinine_max", "inr_min", "pt_min", "alt_max",
    "fresh_frozen_plasma_input", "platelets_input", "rbw_input"
]
BINARY_FEATURES = [
    "vte_history", "cancer", "respiratory_failure", "heart_failure",
    "vasopressin", "sedative", "cvc"
]

# --- 【新增】为数值特征设置默认值 ---
# 您可以根据实际情况（如特征的平均值、中位数或临床正常值）修改这些默认值
DEFAULT_VALUES = {
    "albumin_max": 2.4,
    "creatinine_max": 1.6,
    "inr_min": 1.1,
    "pt_min": 12.7,
    "alt_max": 46.0,
    "fresh_frozen_plasma_input": 0.0,
    "platelets_input": 0.0,
    "rbw_input": 0.0
}


# --- 页面标题 ---
st.title("🩸 基于LightGBM的VTE事件风险预测系统(Alfafa-sepsis-vte)")
st.markdown("---")

# --- 用户输入界面 ---
if lgbm_model:
    with st.expander("点击此处输入/修改患者指标", expanded=True):
        input_data = {}

        with st.form("vte_input_form"):
            st.subheader("数值指标")
            num_cols = st.columns(4)
            for i, feature in enumerate(NUMERIC_FEATURES):
                with num_cols[i % 4]:
                    # 【修改】使用 st.number_input 的 value 参数设置默认值
                    input_data[feature] = st.number_input(
                        label=feature,
                        step=1.0,
                        format="%.2f",
                        value=DEFAULT_VALUES.get(feature, 0.0) # 从字典获取默认值
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("二元指标 (是/否)")
            bin_cols = st.columns(4)
            for i, feature in enumerate(BINARY_FEATURES):
                with bin_cols[i % 4]:
                     # 【修改】为radio设置默认值'否' (index=0)
                    value = st.radio(
                        label=feature,
                        options=['否', '是'],
                        key=f"radio_{feature}",
                        horizontal=True,
                        index=0
                    )
                    input_data[feature] = 1 if value == '是' else 0

            submitted = st.form_submit_button("执行VTE风险预测")

    # --- 预测和结果展示 ---
    if submitted:
        st.header("📊 预测结果与个体化解释")

        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLUMNS]

        prediction_proba = lgbm_model.predict_proba(input_df)[:, 1][0]

        risk_level, risk_color = "", ""
        if prediction_proba <= 0.0078:
            risk_level, risk_color = "低风险", "green"
        elif 0.0078 < prediction_proba <= 0.0294:
            risk_level, risk_color = "中风险", "orange"
        else:
            risk_level, risk_color = "高风险", "red"

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="VTE事件预测概率", value=f"{prediction_proba:.4%}")
        with col2:
            st.markdown(f"### 风险等级: <font color='{risk_color}'>**{risk_level}**</font>", unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("个体化预测归因 (SHAP Waterfall)")
        st.markdown(
            "下图解释了每个特征如何将预测概率从基线值（`base value`）推向最终的输出值。"
            "**红色**的特征是增加风险的因素，**蓝色**的特征是降低风险的因素。"
        )
        # 创建SHAP解释器并计算SHAP值
        explainer = shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(input_df)

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            ),
            show=False
        )

        st.pyplot(plt.gcf())

        with st.expander("查看本次输入的详细信息"):
            st.dataframe(input_df.style.highlight_max(axis=1))

else:
    st.warning("模型未能加载，应用无法运行。请检查 'LightGBM.joblib' 文件是否存在。")

