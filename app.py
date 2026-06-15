import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

st.set_page_config(
    page_title="Triagem Fuzzy Hospitalar",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #0056b3;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #4a5568;
            margin-bottom: 1.5rem;
        }
        .triage-card {
            padding: 1.5rem;
            border-radius: 10px;
            color: #ffffff;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .report-title {
            font-size: 1.3rem;
            font-weight: bold;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            color: #002855;
        }
        </style>
    """, unsafe_allow_html=True)
except TypeError as erro_original:
    raise TypeError(f"Falha de renderização no Streamlit: O parâmetro fornecido para permitir HTML é inválido. Verifique possíveis erros de digitação no nome do argumento. Erro base: {erro_original}")

st.markdown('<p class="main-title">🩺 Sistema de Triagem Hospitalar Inteligente</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Motor de Inferência Fuzzy para Classificação de Urgência Baseado em Sinais Vitais</p>', unsafe_allow_html=True)

if 'val_temp' not in st.session_state:
    st.session_state['val_temp'] = 36.5
if 'val_oxim' not in st.session_state:
    st.session_state['val_oxim'] = 98
if 'val_fc' not in st.session_state:
    st.session_state['val_fc'] = 80
if 'val_pas' not in st.session_state:
    st.session_state['val_pas'] = 120
if 'val_dor' not in st.session_state:
    st.session_state['val_dor'] = 2


st.sidebar.markdown("## 📋 Dados do Paciente & Calibração")
st.sidebar.markdown("### 🧬 Casos Clínicos (Presets)")
preset = st.sidebar.selectbox("Carregar caso de teste:", [
    "Personalizado (Ajuste Manual)",
    "Caso 1: Paciente Saudável (Sintomas Leves)",
    "Caso 2: Febre Alta & Taquicardia (Sepse/Infecção)",
    "Caso 3: Choque Hipovolêmico (Pressão Baixa & FC Alta)",
    "Caso 4: Parada Respiratória Iminente (SpO2 Crítica)",
    "Caso 5: Crise Hipertensiva com Dor Torácica Severa",
    "Caso 6: Hipotermia Grave"
], key='preset_selector')

if 'last_preset' not in st.session_state:
    st.session_state['last_preset'] = "Personalizado (Ajuste Manual)"

if preset != st.session_state['last_preset']:
    st.session_state['last_preset'] = preset
    if preset == "Caso 1: Paciente Saudável (Sintomas Leves)":
        st.session_state['val_temp'] = 36.5
        st.session_state['val_oxim'] = 98
        st.session_state['val_fc'] = 75
        st.session_state['val_pas'] = 120
        st.session_state['val_dor'] = 1
    elif preset == "Caso 2: Febre Alta & Taquicardia (Sepse/Infecção)":
        st.session_state['val_temp'] = 39.5
        st.session_state['val_oxim'] = 96
        st.session_state['val_fc'] = 115
        st.session_state['val_pas'] = 115
        st.session_state['val_dor'] = 4
    elif preset == "Caso 3: Choque Hipovolêmico (Pressão Baixa & FC Alta)":
        st.session_state['val_temp'] = 35.8
        st.session_state['val_oxim'] = 94
        st.session_state['val_fc'] = 130
        st.session_state['val_pas'] = 80
        st.session_state['val_dor'] = 6
    elif preset == "Caso 4: Parada Respiratória Iminente (SpO2 Crítica)":
        st.session_state['val_temp'] = 36.2
        st.session_state['val_oxim'] = 84
        st.session_state['val_fc'] = 105
        st.session_state['val_pas'] = 110
        st.session_state['val_dor'] = 3
    elif preset == "Caso 5: Crise Hipertensiva com Dor Torácica Severa":
        st.session_state['val_temp'] = 36.8
        st.session_state['val_oxim'] = 97
        st.session_state['val_fc'] = 95
        st.session_state['val_pas'] = 185
        st.session_state['val_dor'] = 9
    elif preset == "Caso 6: Hipotermia Grave":
        st.session_state['val_temp'] = 34.2
        st.session_state['val_oxim'] = 95
        st.session_state['val_fc'] = 50
        st.session_state['val_pas'] = 95
        st.session_state['val_dor'] = 2

st.sidebar.markdown("### 🩸 Sinais Vitais do Paciente")
input_temp = st.sidebar.number_input("Temperatura (°C)", min_value=34.0, max_value=43.0, step=0.1, key='val_temp')
input_oxim = st.sidebar.number_input("Oximetria (SpO2 %)", min_value=50, max_value=100, step=1, key='val_oxim')
input_fc = st.sidebar.number_input("Frequência Cardíaca (bpm)", min_value=30, max_value=200, step=1, key='val_fc')
input_pas = st.sidebar.number_input("Pressão Arterial Sistólica (mmHg)", min_value=50, max_value=200, step=1, key='val_pas')
input_dor = st.sidebar.number_input("Nível de Dor (0 a 10)", min_value=0, max_value=10, step=1, key='val_dor')

# Calibração dos Limites Fuzzy
with st.sidebar.expander("⚙️ Calibração dos Limites Fuzzy", expanded=False):
    st.write("Ajuste os pontos de transição das variáveis linguísticas em tempo real:")
    
    st.markdown("**Temperatura (°C)**")
    lim_t_hipo = st.slider("Hipotermia até:", 34.5, 36.5, 35.5, step=0.1)
    lim_t_febre = st.slider("Febre a partir de:", 37.0, 39.0, 37.8, step=0.1)
    
    st.markdown("**Oximetria (SpO2 %)**")
    lim_o_crit = st.slider("Crítica até:", 80, 94, 90, step=1)
    lim_o_norm = st.slider("Normal a partir de:", 91, 98, 95, step=1)
    
    st.markdown("**Frequência Cardíaca (bpm)**")
    lim_fc_bradi = st.slider("Bradicardia até:", 45, 75, 60, step=1)
    lim_fc_taqui = st.slider("Taquicardia a partir de:", 85, 120, 100, step=1)
    
    st.markdown("**Pressão Arterial Sistólica (mmHg)**")
    lim_pas_hipo = st.slider("Hipotensão até:", 80, 100, 90, step=1)
    lim_pas_hiper = st.slider("Hipertensão a partir de:", 120, 160, 140, step=1)
    
    st.markdown("**Nível de Dor (0-10)**")
    lim_dor_leve = st.slider("Dor Leve até:", 1, 5, 3, step=1)
    lim_dor_seve = st.slider("Dor Severa a partir de:", 6, 9, 7, step=1)

# FIS
x_temp = np.arange(34.0, 43.1, 0.1)
x_oxim = np.arange(50, 101, 1)
x_fc = np.arange(30, 201, 1)
x_pas = np.arange(50, 201, 1)
x_dor = np.arange(0, 11, 1)
x_urg = np.arange(0, 101, 1)

# Variáveis fuzzy
temp = ctrl.Antecedent(x_temp, 'temperatura')
oxim = ctrl.Antecedent(x_oxim, 'oximetria')
fc = ctrl.Antecedent(x_fc, 'frequencia_cardiaca')
pas = ctrl.Antecedent(x_pas, 'pressao_sistolica')
dor = ctrl.Antecedent(x_dor, 'dor')
urgencia = ctrl.Consequent(x_urg, 'urgencia')

# Funções
# Temperatura
temp['hipotermia'] = fuzz.trapmf(temp.universe, [34.0, 34.0, lim_t_hipo - 0.5, lim_t_hipo])
temp['normal'] = fuzz.trimf(temp.universe, [lim_t_hipo - 0.5, (lim_t_hipo + lim_t_febre)/2, lim_t_febre + 0.5])
temp['febre'] = fuzz.trapmf(temp.universe, [lim_t_febre, lim_t_febre + 0.5, 43.0, 43.0])

# Oximetria
oxim['critica'] = fuzz.trapmf(oxim.universe, [50, 50, lim_o_crit - 2, lim_o_crit])
oxim['limitrofe'] = fuzz.trimf(oxim.universe, [lim_o_crit - 2, (lim_o_crit + lim_o_norm)/2, lim_o_norm + 1])
oxim['normal'] = fuzz.trapmf(oxim.universe, [lim_o_norm, lim_o_norm + 1, 100, 100])

# Frequência Cardíaca
fc['bradicardia'] = fuzz.trapmf(fc.universe, [30, 30, lim_fc_bradi - 10, lim_fc_bradi])
fc['normal'] = fuzz.trimf(fc.universe, [lim_fc_bradi - 5, (lim_fc_bradi + lim_fc_taqui)/2, lim_fc_taqui + 10])
fc['taquicardia'] = fuzz.trapmf(fc.universe, [lim_fc_taqui, lim_fc_taqui + 10, 200, 200])

# Pressão Arterial
pas['hipotensao'] = fuzz.trapmf(pas.universe, [50, 50, lim_pas_hipo - 10, lim_pas_hipo])
pas['normal'] = fuzz.trimf(pas.universe, [lim_pas_hipo - 5, (lim_pas_hipo + lim_pas_hiper)/2, lim_pas_hiper + 15])
pas['hipertensao'] = fuzz.trapmf(pas.universe, [lim_pas_hiper, lim_pas_hiper + 10, 200, 200])

# Dor
dor['leve'] = fuzz.trapmf(dor.universe, [0, 0, lim_dor_leve, lim_dor_leve + 1])
dor['moderada'] = fuzz.trimf(dor.universe, [lim_dor_leve, (lim_dor_leve + lim_dor_seve)/2, lim_dor_seve + 1])
dor['severa'] = fuzz.trapmf(dor.universe, [lim_dor_seve, lim_dor_seve + 1, 10, 10])

# Urgencia
urgencia['nao_urgente'] = fuzz.trimf(urgencia.universe, [0, 15, 35])
urgencia['pouco_urgente'] = fuzz.trimf(urgencia.universe, [25, 45, 65])
urgencia['muito_urgente'] = fuzz.trimf(urgencia.universe, [55, 75, 85])
urgencia['emergencia'] = fuzz.trapmf(urgencia.universe, [75, 90, 100, 100])

# Regra Base
rules = [
    # Emergência
    ctrl.Rule(oxim['critica'], urgencia['emergencia'], label='R1: Oximetria Crítica -> Emergência'),
    ctrl.Rule(temp['febre'] & fc['taquicardia'], urgencia['emergencia'], label='R2: Febre & Taquicardia -> Emergência'),
    ctrl.Rule(pas['hipotensao'] & fc['taquicardia'], urgencia['emergencia'], label='R3: Choque (Hipotensão & Taquicardia) -> Emergência'),
    ctrl.Rule(pas['hipotensao'] & temp['hipotermia'], urgencia['emergencia'], label='R4: Hipotensão & Hipotermia -> Emergência'),
    
    # Muito Urgente
    ctrl.Rule(oxim['limitrofe'], urgencia['muito_urgente'], label='R5: Oximetria Limítrofe -> Muito Urgente'),
    ctrl.Rule(temp['hipotermia'], urgencia['muito_urgente'], label='R6: Hipotermia isolada -> Muito Urgente'),
    ctrl.Rule(fc['bradicardia'], urgencia['muito_urgente'], label='R7: Bradicardia grave -> Muito Urgente'),
    ctrl.Rule(dor['severa'] & temp['febre'], urgencia['muito_urgente'], label='R8: Dor Severa & Febre -> Muito Urgente'),
    ctrl.Rule(dor['severa'] & pas['hipertensao'], urgencia['muito_urgente'], label='R9: Dor Severa & Hipertensão -> Muito Urgente'),
    
    # Pouco Urgente
    ctrl.Rule(temp['febre'] & fc['normal'], urgencia['pouco_urgente'], label='R10: Febre isolada com FC Normal -> Pouco Urgente'),
    ctrl.Rule(pas['hipertensao'] & fc['normal'], urgencia['pouco_urgente'], label='R11: Hipertensão isolada com FC Normal -> Pouco Urgente'),
    ctrl.Rule(dor['moderada'], urgencia['pouco_urgente'], label='R12: Dor Moderada -> Pouco Urgente'),
    ctrl.Rule(dor['severa'] & temp['normal'] & oxim['normal'], urgencia['pouco_urgente'], label='R13: Dor Severa com sinais vitais Normais -> Pouco Urgente'),
    
    # Não Urgente
    ctrl.Rule(temp['normal'] & fc['normal'] & pas['normal'] & oxim['normal'] & dor['leve'], urgencia['nao_urgente'], label='R14: Todos os parâmetros Normais & Dor Leve -> Não Urgente'),
    ctrl.Rule(temp['normal'] & fc['normal'] & oxim['normal'] & dor['leve'], urgencia['nao_urgente'], label='R15: Sinais Vitais Normais & Dor Leve -> Não Urgente')
]

hospital_system = ctrl.ControlSystem(rules)
triage_sim = ctrl.ControlSystemSimulation(hospital_system)

triage_sim.input['temperatura'] = input_temp
triage_sim.input['oximetria'] = input_oxim
triage_sim.input['frequencia_cardiaca'] = input_fc
triage_sim.input['pressao_sistolica'] = input_pas
triage_sim.input['dor'] = input_dor

# Simulação Fuzzy
try:
    triage_sim.compute()
    result_urgencia = triage_sim.output['urgencia']
    sim_error = None
except Exception as e:
    result_urgencia = 50.0 # fallback
    sim_error = str(e)

# Calculate individual membership grades for mathematical report
grades = {
    'Temperatura': {
        'Hipotermia': fuzz.interp_membership(x_temp, temp['hipotermia'].mf, input_temp),
        'Normal': fuzz.interp_membership(x_temp, temp['normal'].mf, input_temp),
        'Febre': fuzz.interp_membership(x_temp, temp['febre'].mf, input_temp)
    },
    'Oximetria': {
        'Crítica': fuzz.interp_membership(x_oxim, oxim['critica'].mf, input_oxim),
        'Limítrofe': fuzz.interp_membership(x_oxim, oxim['limitrofe'].mf, input_oxim),
        'Normal': fuzz.interp_membership(x_oxim, oxim['normal'].mf, input_oxim)
    },
    'Frequência Cardíaca': {
        'Bradicardia': fuzz.interp_membership(x_fc, fc['bradicardia'].mf, input_fc),
        'Normal': fuzz.interp_membership(x_fc, fc['normal'].mf, input_fc),
        'Taquicardia': fuzz.interp_membership(x_fc, fc['taquicardia'].mf, input_fc)
    },
    'Pressão Arterial': {
        'Hipotensão': fuzz.interp_membership(x_pas, pas['hipotensao'].mf, input_pas),
        'Normal': fuzz.interp_membership(x_pas, pas['normal'].mf, input_pas),
        'Hipertensão': fuzz.interp_membership(x_pas, pas['hipertensao'].mf, input_pas)
    },
    'Nível de Dor': {
        'Leve': fuzz.interp_membership(x_dor, dor['leve'].mf, input_dor),
        'Moderada': fuzz.interp_membership(x_dor, dor['moderada'].mf, input_dor),
        'Severa': fuzz.interp_membership(x_dor, dor['severa'].mf, input_dor)
    }
}

# Output de urgência
if result_urgencia < 25:
    triage_name = "NÃO URGENTE"
    triage_color = "#2ecc71"
    triage_text_color = "#ffffff"
    wait_time = "Atendimento em até 240 minutos"
    triage_desc = "Paciente está estável e pode aguardar o atendimento clínico de rotina de forma segura."
elif result_urgencia < 50:
    triage_name = "POUCO URGENTE"
    triage_color = "#f1c40f"
    triage_text_color = "#000000"
    wait_time = "Atendimento em até 120 minutos"
    triage_desc = "Paciente apresenta sintomas de gravidade moderada, necessitando de reavaliação periódica."
elif result_urgencia < 75:
    triage_name = "MUITO URGENTE"
    triage_color = "#e67e22"
    triage_text_color = "#ffffff"
    wait_time = "Atendimento em até 10 minutos"
    triage_desc = "Paciente apresenta sinais clínicos graves com risco moderado de evolução para óbito."
else:
    triage_name = "EMERGÊNCIA / CRÍTICO"
    triage_color = "#e74c3c"
    triage_text_color = "#ffffff"
    wait_time = "Atendimento IMEDIATO"
    triage_desc = "Estado de altíssima gravidade com risco iminente de morte. Canalizar todos os recursos para intervenção imediata."

tab_res, tab_plots, tab_rules = st.tabs([
    "📊 Resultado da Triagem",
    "📈 Gráficos de Sinais Vitais",
    "📜 Regras & Calibração"
])

# TAB 1: RESULTADO DA TRIAGEM
with tab_res:
    if sim_error:
        st.error(f"⚠️ Erro ao simular o sistema fuzzy: {sim_error}. Certifique-se de que os limites estão configurados de forma que permitam o acionamento de pelo menos uma regra.")
    
    col_metric, col_desc = st.columns([1, 2])
    
    with col_metric:
        st.markdown(f"""
            <div class="triage-card" style="background-color: {triage_color}; color: {triage_text_color}; text-align: center;">
                <p style="margin:0; font-size: 1.1rem; font-weight: bold; text-transform: uppercase;">Classificação de Urgência</p>
                <h1 style="margin:0; font-size: 2.8rem; font-weight: 900;">{result_urgencia:.1f}%</h1>
                <h2 style="margin:0; font-size: 1.5rem; font-weight: bold;">{triage_name}</h2>
                <hr style="border-top: 1px solid rgba(255,255,255,0.3); margin: 0.8rem 0;">
                <p style="margin:0; font-size: 1.0rem; font-weight: bold;">⏱️ {wait_time}</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.metric(label="Pontuação de Urgência", value=f"{result_urgencia:.1f}/100")
        
    with col_desc:
        st.markdown('<p class="report-title">📝 Parecer Clínico & Justificativa</p>', unsafe_allow_html=True)
        st.write(triage_desc)
        
        st.markdown("##### 📌 Resumo dos Sinais Vitais Detectados")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Temp.", f"{input_temp}°C")
        metric_cols[1].metric("Oximetria (SpO2)", f"{input_oxim}%")
        metric_cols[2].metric("Freq. Cardíaca", f"{input_fc} bpm")
        metric_cols[3].metric("Pressão (PAS)", f"{input_pas} mmHg")
        metric_cols[4].metric("Nível de Dor", f"{input_dor}/10")

    st.markdown("---")
    
    st.markdown('<p class="report-title">🧮 Detalhamento Matemático (Graus de Pertinência)</p>', unsafe_allow_html=True)
    st.write(r"Abaixo está detalhado o grau de pertinência $\mu(x)$ de cada sinal vital coletado em relação aos conjuntos nebulosos definidos. Um valor de `1.00` representa pertinência total e `0.00` representa nenhuma pertinência.")
    
    col_per_1, col_per_2 = st.columns(2)
    
    with col_per_1:
        st.markdown("##### 🩺 Sinais Vitais Físicos")
        for var_name in ['Temperatura', 'Oximetria', 'Frequência Cardíaca', 'Pressão Arterial']:
            st.markdown(f"**{var_name}**")
            for term, val in grades[var_name].items():
                st.write(f"- {term}: `{val:.2f}`")
                st.progress(float(val))
                
    with col_per_2:
        st.markdown("##### 🧠 Sintomas Subjetivos & Regras de Ativação")
        st.markdown("**Nível de Dor**")
        for term, val in grades['Nível de Dor'].items():
            st.write(f"- {term}: `{val:.2f}`")
            st.progress(float(val))
            
        # Explicação das Regras
        rule_activations = [
            ("R1: Oximetria Crítica", grades['Oximetria']['Crítica'], "Emergência"),
            ("R2: Febre & Taquicardia", min(grades['Temperatura']['Febre'], grades['Frequência Cardíaca']['Taquicardia']), "Emergência"),
            ("R3: Choque (Hipotensão & Taquicardia)", min(grades['Pressão Arterial']['Hipotensão'], grades['Frequência Cardíaca']['Taquicardia']), "Emergência"),
            ("R4: Hipotensão & Hipotermia", min(grades['Pressão Arterial']['Hipotensão'], grades['Temperatura']['Hipotermia']), "Emergência"),
            ("R5: Oximetria Limítrofe", grades['Oximetria']['Limítrofe'], "Muito Urgente"),
            ("R6: Hipotermia Isolada", grades['Temperatura']['Hipotermia'], "Muito Urgente"),
            ("R7: Bradicardia Grave", grades['Frequência Cardíaca']['Bradicardia'], "Muito Urgente"),
            ("R8: Dor Severa & Febre", min(grades['Nível de Dor']['Severa'], grades['Temperatura']['Febre']), "Muito Urgente"),
            ("R9: Dor Severa & Hipertensão", min(grades['Nível de Dor']['Severa'], grades['Pressão Arterial']['Hipertensão']), "Muito Urgente"),
            ("R10: Febre com FC Normal", min(grades['Temperatura']['Febre'], grades['Frequência Cardíaca']['Normal']), "Pouco Urgente"),
            ("R11: Hipertensão com FC Normal", min(grades['Pressão Arterial']['Hipertensão'], grades['Frequência Cardíaca']['Normal']), "Pouco Urgente"),
            ("R12: Dor Moderada", grades['Nível de Dor']['Moderada'], "Pouco Urgente"),
            ("R13: Dor Severa com Sinais Normais", min(grades['Nível de Dor']['Severa'], grades['Temperatura']['Normal'], grades['Oximetria']['Normal']), "Pouco Urgente"),
            ("R14: Todos Parâmetros Normais & Dor Leve", min(grades['Temperatura']['Normal'], grades['Frequência Cardíaca']['Normal'], grades['Pressão Arterial']['Normal'], grades['Oximetria']['Normal'], grades['Nível de Dor']['Leve']), "Não Urgente"),
            ("R15: Sinais Normais & Dor Leve", min(grades['Temperatura']['Normal'], grades['Frequência Cardíaca']['Normal'], grades['Oximetria']['Normal'], grades['Nível de Dor']['Leve']), "Não Urgente")
        ]
        
        active_rules = [(name, strength, cons) for name, strength, cons in rule_activations if strength > 0]
        active_rules.sort(key=lambda x: x[1], reverse=True)
        
        st.markdown("**⚡ Regras Fuzzy Ativadas**")
        if active_rules:
            for r_name, r_strength, r_cons in active_rules:
                st.write(f"- **{r_name}**: Ativação = `{r_strength:.2f}` ➔ Conseqüente: *{r_cons}*")
        else:
            st.warning("Nenhuma regra de inferência foi ativada. O sistema está aplicando uma defuzzificação padrão.")

    st.markdown("---")
    
    st.markdown('<p class="report-title">📐 Gráfico de Defuzzificação (Método do Centroide)</p>', unsafe_allow_html=True)
    st.write("O gráfico abaixo mostra as funções de pertinência do Nível de Urgência de saída. A área sombreada em cinza representa a agregação das regras ativadas cortadas no nível de seus respectivos coeficientes. A linha roxa marca a coordenada do **Centróide** resultante, que representa a nota numérica final de urgência:")
    
    # Gráfico defuzzification
    plt.figure()
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = {'nao_urgente': '#2ecc71', 'pouco_urgente': '#f1c40f', 'muito_urgente': '#e67e22', 'emergencia': '#e74c3c'}
    for term_name, term_obj in urgencia.terms.items():
        ax.plot(urgencia.universe, term_obj.mf, label=term_name.replace('_', ' ').title(), color=colors.get(term_name, '#34495e'), linewidth=1.5, linestyle=':')
    
    try:
        urgencia.view(sim=triage_sim)
        fig_urg = plt.gcf()
        fig_urg.set_size_inches(10, 4)
        st.pyplot(fig_urg)
        plt.close(fig_urg)
    except Exception as e:
        ax.axvline(x=result_urgencia, color='#8e44ad', linestyle='-', linewidth=3, label=f'Centróide = {result_urgencia:.1f}')
        ax.set_title("Nível de Urgência")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

# TAB 2: GRÁFICOS DE SINAIS VITAIS
with tab_plots:
    st.markdown('<p class="report-title">📈 Visualização das Funções de Pertinência e Estados do Paciente</p>', unsafe_allow_html=True)
    st.write("Os gráficos abaixo ilustram a modelagem matemática das curvas nebulosas e indicam a posição exata do paciente em cada sinal vital (linha vertical tracejada):")
    
    def make_custom_plot(universe, fuzzy_var, current_val, title, unit, color_scheme):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        for (term_name, term_obj), col in zip(fuzzy_var.terms.items(), color_scheme):
            ax.plot(universe, term_obj.mf, label=term_name.capitalize(), color=col, linewidth=2)
            ax.fill_between(universe, 0, term_obj.mf, color=col, alpha=0.1)
            
        ax.axvline(x=current_val, color='#2c3e50', linestyle='--', linewidth=2, label=f'Paciente ({current_val}{unit})')
        ax.set_title(title, fontsize=10, fontweight='bold', color="#2c3e50")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(universe[0], universe[-1])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        return fig

    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        fig1 = make_custom_plot(
            x_temp, temp, input_temp, 
            "Temperatura Corporal", "°C", 
            ['#3498db', '#2ecc71', '#e74c3c'] 
        )
        st.pyplot(fig1)
        plt.close(fig1)
        
    with row1_col2:
        fig2 = make_custom_plot(
            x_oxim, oxim, input_oxim, 
            "Saturação de Oxigênio (Oximetria SpO2)", "%", 
            ['#e74c3c', '#f1c40f', '#2ecc71'] 
        )
        st.pyplot(fig2)
        plt.close(fig2)
        
    # Row 2 of plots
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        fig3 = make_custom_plot(
            x_fc, fc, input_fc, 
            "Frequência Cardíaca (Pulso)", " bpm", 
            ['#3498db', '#2ecc71', '#e74c3c'] 
        )
        st.pyplot(fig3)
        plt.close(fig3)
        
    with row2_col2:
        fig4 = make_custom_plot(
            x_pas, pas, input_pas, 
            "Pressão Arterial Sistólica (Sistólica)", " mmHg", 
            ['#e74c3c', '#2ecc71', '#9b59b6'] #
        )
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown("---")
    row3_col1, row3_col2 = st.columns([1.5, 1])
    
    with row3_col1:
        fig5 = make_custom_plot(
            x_dor, dor, input_dor, 
            "Escala Visual Analógica de Dor", "/10", 
            ['#2ecc71', '#f39c12', '#e74c3c']
        )
        st.pyplot(fig5)
        plt.close(fig5)
        
    with row3_col2:
        st.markdown("##### ℹ️ Entendendo a Resposta Fuzzy")
        st.write(r"""
            Ao olhar para as curvas, veja onde a linha vertical do paciente intercepta cada cor.
            Por exemplo:
            - Se a linha da **Temperatura** cruzar a curva vermelha (Febre) na altura de `0.8` e a curva verde (Normal) na altura de `0.2`, isso significa matematicamente que o paciente é considerado 'Febril' com 80% de certeza e 'Normal' com 20% de certeza.
            - O sistema combina essas frações em múltiplos antecedentes usando regras booleanas modificadas ($\min$ para conjunções 'E' e $\max$ para disjunções 'OU') para gerar a urgência ponderada final.
        """)

# TAB 3: REGRAS & CALIBRAÇÃO
with tab_rules:
    st.markdown('<p class="report-title">📜 Base de Regras de Inferência Nebulosa</p>', unsafe_allow_html=True)
    st.write("A base de regras mapeia os sinais vitais linguísticos em graus clínicos de urgência, baseando-se em diretrizes médicas de triagem (como o protocolo de Manchester):")
    
    rule_data = []
    for r in rules:
        rule_data.append({
            "Regra": r.label,
            "Antecedente (Condição)": str(r.antecedent),
            "Conseqüente (Resultado)": str(r.consequent)
        })
        
    st.table(rule_data)

