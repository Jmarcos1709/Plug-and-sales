import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Dashboard Plug & Sales",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS customizados
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Função para carregar e processar dados
@st.cache_data
def carregar_dados(arquivo):
    try:
        df = pd.read_csv(arquivo, sep=';', encoding='utf-8', skiprows=3)
        
        # Limpar dados
        df = df[df['Unnamed: 0'].notna()]
        df = df[~df['Unnamed: 0'].astype(str).str.contains('NaN|TOTAL', na=False)]
        
        # Converter data
        df['Data'] = pd.to_datetime(df['Unnamed: 0'], format='%d/%m/%Y', errors='coerce')
        df = df[df['Data'].notna()]
        
        # Função para limpar valores
        def limpar_valor(valor):
            if pd.isna(valor):
                return 0
            if isinstance(valor, str):
                valor = valor.replace('R$', '').replace('.', '').replace(',', '.').strip()
                try:
                    return float(valor)
                except:
                    return 0
            return float(valor)
        
        # Limpar colunas numéricas
        colunas_numericas = ['CTWA', 'PRÉ SELL', 'TOTAL', 'META', 'WHATSAPP', 'ADS', 
                            'PAGE', 'PG', 'QUALIFICADO', 'DESQUALIFICADO', 'PARADO NO FUNIL',
                            'WHATSAPP.2', 'LIGAÇÃO', 'TOTAL.1']
        
        for col in colunas_numericas:
            if col in df.columns:
                df[col] = df[col].apply(limpar_valor)
        
        # Adicionar colunas calculadas
        df['MesAno'] = df['Data'].dt.strftime('%m/%Y')
        df['Mes'] = df['Data'].dt.month
        df['Ano'] = df['Data'].dt.year
        df['CPL'] = df['TOTAL'] / df['META']
        df['CPL'] = df['CPL'].replace([np.inf, -np.inf], np.nan)
        df['Taxa_Conversao'] = (df['WHATSAPP.2'] / df['META']) * 100
        df['Taxa_Qualificacao'] = (df['QUALIFICADO'] / df['META']) * 100
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

# Header
st.title("📊 Dashboard Plug & Sales")
st.markdown("---")

# Upload do arquivo
uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV", type=['csv'])

if uploaded_file is None:
    # Tentar carregar o arquivo existente
    try:
        df = carregar_dados('Plug-And-Sales-CSV.csv')
    except:
        st.warning("⚠ Faça upload do arquivo CSV para visualizar o dashboard")
        st.stop()
else:
    df = carregar_dados(uploaded_file)

if df is None or len(df) == 0:
    st.error("Não foi possível processar os dados do arquivo")
    st.stop()

# Sidebar - Filtros
st.sidebar.header("🔍 Filtros")

# Filtro de data
data_min = df['Data'].min()
data_max = df['Data'].max()

col1, col2 = st.sidebar.columns(2)
with col1:
    data_inicio = st.date_input("Data Início", data_min, min_value=data_min, max_value=data_max)
with col2:
    data_fim = st.date_input("Data Fim", data_max, min_value=data_min, max_value=data_max)

# Aplicar filtros
df_filtrado = df[(df['Data'] >= pd.to_datetime(data_inicio)) & (df['Data'] <= pd.to_datetime(data_fim))]

# Filtro de canal
canais_disponiveis = ['Todos', 'CTWA', 'Pré Sell']
canal_selecionado = st.sidebar.selectbox("Canal", canais_disponiveis)

# KPIs Principais
st.header("📈 Métricas Principais")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_gasto = df_filtrado['TOTAL'].sum()
    st.metric("💰 Gasto Total", f"R$ {total_gasto:,.2f}")

with col2:
    total_leads = df_filtrado['META'].sum()
    st.metric("👥 Total de Leads", f"{total_leads:,.0f}")

with col3:
    cpl_medio = total_gasto / total_leads if total_leads > 0 else 0
    st.metric("📊 CPL Médio", f"R$ {cpl_medio:.2f}")

with col4:
    total_qualificados = df_filtrado['QUALIFICADO'].sum()
    st.metric("✅ Qualificados", f"{total_qualificados:,.0f}")

with col5:
    taxa_qualificacao = (total_qualificados / total_leads * 100) if total_leads > 0 else 0
    st.metric("📈 Taxa Qualificação", f"{taxa_qualificacao:.1f}%")

st.markdown("---")

# Row 1: Gráficos de Evolução
col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Evolução do Gasto Total")
    gasto_diario = df_filtrado.groupby('Data')['TOTAL'].sum().reset_index()
    fig_gasto = px.line(gasto_diario, x='Data', y='TOTAL', 
                        markers=True,
                        labels={'TOTAL': 'Gasto (R$)', 'Data': 'Data'})
    fig_gasto.update_traces(line_color='#3498db', line_width=3)
    fig_gasto.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_gasto, use_container_width=True)

with col2:
    st.subheader("💼 Gasto por Canal")
    gasto_ctwa = df_filtrado['CTWA'].sum()
    gasto_presell = df_filtrado['PRÉ SELL'].sum()
    
    fig_canais = go.Figure(data=[
        go.Bar(name='CTWA', x=['CTWA'], y=[gasto_ctwa], marker_color='#3498db'),
        go.Bar(name='Pré Sell', x=['Pré Sell'], y=[gasto_presell], marker_color='#e74c3c')
    ])
    fig_canais.update_layout(
        height=400,
        showlegend=False,
        yaxis_title="Gasto (R$)"
    )
    st.plotly_chart(fig_canais, use_container_width=True)

# Row 2: Leads
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Distribuição de Leads por Fonte")
    lead_whatsapp = df_filtrado['WHATSAPP'].sum()
    lead_ads = df_filtrado['ADS'].sum()
    lead_page = df_filtrado['PAGE'].sum()
    
    fig_leads = go.Figure(data=[go.Pie(
        labels=['WhatsApp', 'Ads', 'Page'],
        values=[lead_whatsapp, lead_ads, lead_page],
        hole=.3,
        marker_colors=['#2ecc71', '#3498db', '#9b59b6']
    )])
    fig_leads.update_layout(height=400)
    st.plotly_chart(fig_leads, use_container_width=True)

with col2:
    st.subheader("📊 Leads Mensais por Fonte")
    leads_mensais = df_filtrado.groupby('MesAno')[['WHATSAPP', 'ADS', 'PAGE']].sum().reset_index()
    
    fig_leads_mensais = go.Figure()
    fig_leads_mensais.add_trace(go.Bar(name='WhatsApp', x=leads_mensais['MesAno'], 
                                       y=leads_mensais['WHATSAPP'], marker_color='#2ecc71'))
    fig_leads_mensais.add_trace(go.Bar(name='Ads', x=leads_mensais['MesAno'], 
                                       y=leads_mensais['ADS'], marker_color='#3498db'))
    fig_leads_mensais.add_trace(go.Bar(name='Page', x=leads_mensais['MesAno'], 
                                       y=leads_mensais['PAGE'], marker_color='#9b59b6'))
    
    fig_leads_mensais.update_layout(
        barmode='group',
        height=400,
        xaxis_title="Mês/Ano",
        yaxis_title="Quantidade de Leads"
    )
    st.plotly_chart(fig_leads_mensais, use_container_width=True)

# Row 3: Qualificação e Conversão
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔄 Funil de Qualificação")
    total_qualificado = df_filtrado['QUALIFICADO'].sum()
    total_desqualificado = df_filtrado['DESQUALIFICADO'].sum()
    total_parado = df_filtrado['PARADO NO FUNIL'].sum()
    
    fig_qualificacao = go.Figure(data=[go.Bar(
        y=['Qualificado', 'Desqualificado', 'Parado'],
        x=[total_qualificado, total_desqualificado, total_parado],
        orientation='h',
        marker_color=['#2ecc71', '#e74c3c', '#f39c12'],
        text=[f'{total_qualificado:,.0f}', f'{total_desqualificado:,.0f}', f'{total_parado:,.0f}'],
        textposition='auto'
    )])
    fig_qualificacao.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Quantidade"
    )
    st.plotly_chart(fig_qualificacao, use_container_width=True)

with col2:
    st.subheader("💹 Taxa de Conversão Mensal")
    conversao_mensal = df_filtrado.groupby('MesAno')['Taxa_Conversao'].mean().reset_index()
    
    fig_conversao = px.line(conversao_mensal, x='MesAno', y='Taxa_Conversao',
                            markers=True,
                            labels={'Taxa_Conversao': 'Taxa (%)', 'MesAno': 'Mês/Ano'})
    fig_conversao.update_traces(line_color='#9b59b6', line_width=3)
    fig_conversao.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_conversao, use_container_width=True)

# Row 4: CPL e Top Dias
col1, col2 = st.columns(2)

with col1:
    st.subheader("💵 CPL Médio ao Longo do Tempo")
    cpl_diario = df_filtrado.groupby('Data')['CPL'].mean().reset_index()
    
    fig_cpl = px.line(cpl_diario, x='Data', y='CPL',
                      markers=True,
                      labels={'CPL': 'CPL (R$)', 'Data': 'Data'})
    fig_cpl.update_traces(line_color='#e67e22', line_width=3)
    fig_cpl.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_cpl, use_container_width=True)

with col2:
    st.subheader("🔝 Top 10 Dias com Maior Gasto")
    top_gastos = df_filtrado.nlargest(10, 'TOTAL')[['Data', 'TOTAL']].sort_values('TOTAL', ascending=True)
    top_gastos['Data_Formatada'] = top_gastos['Data'].dt.strftime('%d/%m/%Y')
    
    fig_top = go.Figure(data=[go.Bar(
        x=top_gastos['TOTAL'],
        y=top_gastos['Data_Formatada'],
        orientation='h',
        marker_color='#c0392b',
        text=top_gastos['TOTAL'].apply(lambda x: f'R$ {x:,.2f}'),
        textposition='auto'
    )])
    fig_top.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Gasto (R$)"
    )
    st.plotly_chart(fig_top, use_container_width=True)

# Tabela de dados
st.markdown("---")
st.header("📋 Dados Detalhados")

# Resumo mensal
resumo_mensal = df_filtrado.groupby('MesAno').agg({
    'TOTAL': 'sum',
    'META': 'sum',
    'WHATSAPP': 'sum',
    'ADS': 'sum',
    'QUALIFICADO': 'sum',
    'DESQUALIFICADO': 'sum',
    'PARADO NO FUNIL': 'sum'
}).reset_index()

resumo_mensal['CPL'] = (resumo_mensal['TOTAL'] / resumo_mensal['META']).round(2)
resumo_mensal['Taxa_Qualificacao_%'] = ((resumo_mensal['QUALIFICADO'] / resumo_mensal['META']) * 100).round(2)

# Renomear colunas para exibição
resumo_mensal.columns = ['Mês/Ano', 'Gasto Total (R$)', 'Total Leads', 'Leads WhatsApp', 
                          'Leads Ads', 'Qualificados', 'Desqualificados', 'Parados', 
                          'CPL (R$)', 'Taxa Qualificação (%)']

st.dataframe(resumo_mensal, use_container_width=True, height=400)

# Download dos dados
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col2:
    csv = resumo_mensal.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Relatório CSV",
        data=csv,
        file_name=f'relatorio_plug_sales_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p>Dashboard Plug & Sales | Desenvolvido com Streamlit 📊</p>
    </div>
    """, unsafe_allow_html=True)