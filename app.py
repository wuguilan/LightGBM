import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import lightgbm as lgb  # 导入 lightgbm
import numpy as np

# --- 1. 页面基础设置 ---
# set_page_config() 必须是第一个 Streamlit 命令
st.set_page_config(layout="wide")
st.title("🏥 脓毒症患者多重风险分层工具")
st.markdown("""
**使用LightGBM模型对使用肝素类药物的脓毒症患者进行 血栓、出血及死亡 的风险分层**  
*请在下方输入患者的临床参数，然后点击“评估风险等级”按钮。*
""")


# --- 2. 加载模型和特征 ---
# 使用 st.cache_resource 来缓存加载的模型，提高效率
@st.cache_resource
def load_model():
    # 加载最终的LightGBM多分类模型
    pipeline = joblib.load("final_risk_stratification_model.joblib")
    # 从Pipeline中提取模型本身，用于SHAP分析
    model = pipeline.named_steps['model']
    # 定义新模型的特征名称
    feature_names = ['septic_shock', 'acutephysiologyscore', 'bleed_history', 'heart_failure', 'respiratory_failure',
                     'acs', 'hypertension', 'albumin_max', 'bun_max', 'bilirubin_max', 'creatinine_max']
    return pipeline, model, feature_names


# 执行加载
pipeline, model, feature_names = load_model()

# --- 3. 初始化SHAP解释器 ---
# SHAP需要模型本身，而不是整个pipeline
explainer = shap.TreeExplainer(model)


# --- 4. 定义用户输入界面 ---
def user_input_features():
    st.header("患者临床参数输入")
    col1, col2, col3 = st.columns(3)

    # 重新组织输入项以匹配新模型的11个特征
    with col1:
        septic_shock = st.selectbox("是否为感染性休克 (Septic Shock)", ["否", "是"], index=0)
        acutephysiologyscore = st.number_input("急性生理学评分 (APACHE II/SOFA)", min_value=0, max_value=100, value=15)
        bleed_history = st.selectbox("是否有出血史 (Bleed History)", ["否", "是"], index=0)
        heart_failure = st.selectbox("是否有心力衰竭 (Heart Failure)", ["否", "是"], index=0)

    with col2:
        respiratory_failure = st.selectbox("是否有呼吸衰竭 (Respiratory Failure)", ["否", "是"], index=0)
        acs = st.selectbox("是否有急性冠脉综合征 (ACS)", ["否", "是"], index=0)
        hypertension = st.selectbox("是否有高血压 (Hypertension)", ["否", "是"], index=0)
        albumin_max = st.number_input("最大白蛋白 (g/L)", min_value=10.0, max_value=60.0, value=35.0, step=0.1)

    with col3:
        bun_max = st.number_input("最大尿素氮 (BUN, mg/dL)", min_value=5, max_value=150, value=20)
        bilirubin_max = st.number_input("最大胆红素 (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
        creatinine_max = st.number_input("最大肌酐 (mg/dL)", min_value=0.3, max_value=15.0, value=1.2, step=0.1)

    # 将用户输入转换为模型需要的DataFrame格式
    data = {
        'septic_shock': 1 if septic_shock == "是" else 0,
        'acutephysiologyscore': acutephysiologyscore,
        'bleed_history': 1 if bleed_history == "是" else 0,
        'heart_failure': 1 if heart_failure == "是" else 0,
        'respiratory_failure': 1 if respiratory_failure == "是" else 0,
        'acs': 1 if acs == "是" else 0,
        'hypertension': 1 if hypertension == "是" else 0,
        'albumin_max': albumin_max,
        'bun_max': bun_max,
        'bilirubin_max': bilirubin_max,
        'creatinine_max': creatinine_max
    }

    # 确保列的顺序与训练时一致
    return pd.DataFrame([data], columns=feature_names)


# --- 5. 主函数：运行整个应用 ---
def main():
    # 获取用户输入
    input_df = user_input_features()

    if st.button("评估风险等级"):
        try:
            # --- 核心预测逻辑 ---
            # 使用完整的pipeline进行预测，它会自动处理标准化
            prediction_class = pipeline.predict(input_df)[0]
            prediction_proba = pipeline.predict_proba(input_df)[0]

            # 定义风险等级的名称和颜色
            risk_labels = {0: "低风险", 1: "中风险", 2: "高风险"}
            risk_colors = {0: "green", 1: "orange", 2: "red"}

            # --- 结果展示 ---
            st.success("风险评估完成！")

            # 使用st.metric来突出显示结果
            st.metric(
                label="综合风险等级",
                value=risk_labels[prediction_class]
            )
            st.write(
                f"模型判定该患者属于 **<span style='color:{risk_colors[prediction_class]};'>{risk_labels[prediction_class]}</span>**。",
                unsafe_allow_html=True)

            # 显示每个类别的具体概率
            st.subheader("各风险等级概率")
            probabilities_df = pd.DataFrame({
                '风险等级': [risk_labels[i] for i in range(len(prediction_proba))],
                '概率': [f"{p * 100:.1f}%" for p in prediction_proba]
            })
            st.table(probabilities_df)

            # --- SHAP 可解释性分析 ---
            # --- SHAP 可解释性分析 ---
            scaled_input = pipeline.named_steps['scaler'].transform(input_df)

            scaled_input_df = pd.DataFrame(
                scaled_input,
                columns=feature_names
            )

            shap_values = explainer(scaled_input_df)

            st.subheader("个体化风险归因分析 (SHAP)")
            st.markdown("""
            下图展示了哪些因素对当前病人的风险等级判断贡献最大：
            - **红色**：推高风险
            - **蓝色**：降低风险
            """)

            st.write(f"**对预测结果 “{risk_labels[prediction_class]}” 的归因分析:**")

            fig, ax = plt.subplots()
            shap.plots.waterfall(
                shap_values[0, :, prediction_class],
                max_display=10,
                show=False
            )
            st.pyplot(fig)


        except Exception as e:
            st.error(f"在预测过程中发生错误: {str(e)}")


# --- 6. 侧边栏信息 ---
with st.sidebar:
    st.header("关于此工具")
    st.markdown("""
    - **模型类型**: LightGBM 多分类器
    - **基础模型**: 由两个复杂的Stacking集成模型蒸馏而来
    - **预测目标**: 脓毒症患者的综合风险等级（低、中、高）
    - **训练数据**: 来自多中心的ICU脓毒症患者数据
    """)

    st.header("使用说明")
    st.markdown("""
    1. 在主界面输入患者的11项临床指标。
    2. 点击“评估风险等级”按钮。
    3. 查看模型给出的风险等级、概率及SHAP归因分析。
    """)

    st.warning("""
    **临床决策声明**  
    本工具的预测结果仅供参考，不能替代执业医师的临床判断。所有医疗决策都应基于对患者具体情况的全面评估。
    """)

# --- 运行主程序 ---
if __name__ == '__main__':
    main()

