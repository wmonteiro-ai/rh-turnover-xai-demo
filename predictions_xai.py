import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

##########
#Imports
##########
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import plotly.express as px
import plotly.graph_objects as go

from evalml.model_understanding.prediction_explanations import explain_predictions

##########
#Streamlit setup
##########
st.set_page_config(page_title='Inteledge - Predição de Turnover - RH', page_icon="🔭", layout="centered", initial_sidebar_state="auto", menu_items=None)

##########
#Functions for the predictions and for the page layout
##########
@st.cache(allow_output_mutation=True)
def get_pickles():
	df_sample_true, df_sample_false, df_min_max, df_cat_vars, logical_types = pickle.load(open('sample_pt.pkl', 'rb'))
	best_pipeline, expected_value = pickle.load(open('model_pt.pkl', 'rb'))
	
	df_sample_true = df_sample_true.sort_values(by='% certeza', ascending=False).round(2)
	df_sample_false = df_sample_false.sort_values(by='% certeza', ascending=False).round(2)
	
	cols = df_sample_true.drop('Previsão', axis=1).columns.tolist()
	df_sample_true['Nome'] = ['Mariana', 'Juan', 'Carlos', 'Patricia', 'Alexandra']
	df_sample_true['Sobrenome'] = ['Luz', 'Garcia', 'Perez', 'Zanella', 'Neves']
	df_sample_true = df_sample_true.head(5)[['Nome', 'Sobrenome'] + [cols[-1:] + cols[:-1]]]

	df_sample_false['Nome'] = ['Pedro', 'Georgina', 'Natalia', 'Thiago', 'Denise']
	df_sample_false['Sobrenome'] = ['Neto', 'Lima', 'Diaz', 'Batista', 'Garcez']
	df_sample_false = df_sample_false.head(5)[['Nome', 'Sobrenome'] + [cols[-1:] + cols[:-1]]]
	
	return best_pipeline, expected_value, df_sample_true, df_sample_false, df_min_max, df_cat_vars, logical_types

def plot_importances(best_pipeline, df):
	# predictions
	pred = best_pipeline.predict(df).values[0]
	pred = 'Pedirá demissão' if pred == 1 else 'Não pedirá demissão'
	pred_proba = best_pipeline.predict_proba(df).values[0]
	starting_value = expected_value*100

	df_plot = explain_predictions(pipeline=best_pipeline, input_features=df.reset_index(drop=True),
								y=None, top_k_features=len(df.columns), indices_to_explain=[0],
								include_explainer_values=True, output_format='dataframe')

	if np.argmax(pred_proba) == 1:
		df_plot['quantitative_explanation'] = df_plot['quantitative_explanation']*100
	else:
		starting_value = 100-starting_value
		df_plot['quantitative_explanation'] = df_plot['quantitative_explanation']*-100

	df_plot = df_plot.sort_values(by='quantitative_explanation')
	df_plot['Soma'] = starting_value+df_plot['quantitative_explanation'].cumsum()
	df_plot = df_plot.sort_values(by='quantitative_explanation')
	df_plot['Influencia para este resultado?'] = df_plot['quantitative_explanation']<0
	df_plot = df_plot.round(2)

	cols_to_rename = {'Salário em relação a outros com o mesmo cargo na equipe': 'Salário x outros na equipe',
					'Salário em relação a outros com o mesmo cargo no mercado': 'Salário x outros no mercado',
					'Quantidade de feedbacks nos últimos 12 meses': 'Qtd. feedbacks últ. 12 meses',
					'Experiência do líder neste cargo (em anos)': 'Experiência do líder (anos)',
					'% salarial em relação à média de mercado': '% salário x média de mercado',
					'Meses desde o último aumento ou prêmio': 'Meses desde últ. aumento/prêmio',
					'Turnover da equipe nos últimos 12 meses': 'Turnover da equipe (12 meses)'}
	vals_to_rename = {'Entre em linha e acima do esperado': 'Em linha/acima do esperado'}
	df_plot = df_plot.replace(cols_to_rename)
	df = df.rename(columns=cols_to_rename).replace(vals_to_rename)

	col_names = []
	for col in df_plot['feature_names'].values:
		col_names.append(f'{col}<br><em>({df[col].values[0]})</em>')
        
	fig_xai = go.Figure(go.Waterfall(
		name='Projeção',
		base=0,
		orientation="h",
		y=['Projeção inicial'] + col_names + ['Final'],
		x=[starting_value] + df_plot['quantitative_explanation'].values.tolist() + [0],
		measure=['absolute'] + ['relative']*len(df_plot) + ['total'],
		text=[None] + [f'{x:.1f}%' for x in df_plot['Soma'].values] + [None],
		#textposition = "outside",
		connector = {"line":{"color":"rgb(63, 63, 63)"}},
	))

	fig_xai.update_xaxes(range=[max(0, df_plot['Soma'].min()-18),
								min(100, df_plot['Soma'].max()+18)])

	fig_xai.update_layout(
	title=f'Principais influenciadores para esta predição:<br>(<b>{pred}</b>, com {round(100*pred_proba.max(),1)}% de certeza)',
	showlegend = False,
	width=430,
	height=1400
	)

	return fig_xai, pred

##########
#Preparing the simulator
##########
# loading the predictive models
best_pipeline, expected_value, df_sample_true, df_sample_false, df_min_max, df_cat_vars, logical_types = get_pickles()

##########
#Section 1 - History
##########
col1, _, _ = st.columns(3)
with col1:
	st.image('inteledge.png')

st.title('Predição de Turnover - TI')
st.markdown('Aqui na inteledge desenvolvemos Inteligência Artificial de alta performance para resolver problemas em RH. Possuímos conhecimento técnico e científico comprovado com uma experiência forjada no ambiente corporativo durante vários anos para entregar aquilo que é eficaz e que funciona com um forte embasamento teórico. Aqui, demonstramos para você um exemplo de algoritmo que prevê o risco de saída de um profissional a partir da sua base histórica. Interessante, não é? Ficou interessado? Entre em contato conosco no @inteledge.lab no [Instagram](https://instagram.com/inteledge.lab) ou no [LinkedIn](https://www.linkedin.com/company/inteledge/)!')

st.write('Primeiramente, veja uma amostra de algumas previsões do algoritmo para uma base histórica. É este o tipo de resultado que você terá acesso.')
st.header('Últimas previsões do algoritmo')

st.write('Amostra de 5 pessoas com maior probabilidade de pedir demissão:')
st.dataframe(df_sample_true)

st.write('Amostra de 5 pessoas com maior probabilidade de **não** pedir demissão:')
st.dataframe(df_sample_false)

##########
#Section 2 - Simulator
##########
st.header('Simulador de novos casos')
st.markdown('Já pensou como seria simular novos cenários **em tempo real**? Isto é possível aqui na inteledge. Fique à vontade para testar abaixo novas possibilidades.')
st.write('De um ponto de vista de Ciência de Dados, nenhum algoritmo será 100% correto todas as vezes. Por outro lado, sabendo como ele funciona e como ele chegou a uma determinada previsão nos dá uma confiança e transparência essenciais. Entendemos que toda lógica de uma IA deve ser transparente desde o começo: não pode existir uma caixa-preta para quem utiliza destes algoritmos para tomar decisões.')
col1, col2 = st.columns(2)

with col1:
	# variables 
	
	years_experience = st.slider('Anos de Experiência',
		int(df_min_max['Anos de Experiência']['min']), int(df_min_max['Anos de Experiência']['max']), int(df_sample_true['Anos de Experiência'].iloc[0]))
	
	performance_review = st.selectbox('Avaliação de performance', df_cat_vars['Avaliação de performance'])
		
	distance_work = st.slider('Distância do trabalho (km)',
		int(df_min_max['Distância do trabalho (km)']['min']), int(df_min_max['Distância do trabalho (km)']['max']),
		int(df_sample_true['Distância do trabalho (km)'].iloc[0]))
	
	job_level = st.selectbox('Cargo', df_cat_vars['Cargo'])
	
	overtime = st.selectbox('Está fazendo muitas horas extras?', df_cat_vars['Está fazendo muitas horas extras?'])
	
	marital_status = st.selectbox('Estado civil', df_cat_vars['Estado civil'])
	
	leader_experience = st.slider('Experiência do líder neste cargo (em anos)',
		int(df_min_max['Experiência do líder neste cargo (em anos)']['min']), int(df_min_max['Experiência do líder neste cargo (em anos)']['max']), int(df_sample_true['Experiência do líder neste cargo (em anos)'].iloc[0]))
	
	education = st.selectbox('Formação', df_cat_vars['Formação'])
	
	gender = st.selectbox('Gênero', df_cat_vars['Gênero'])
		
	age = st.slider('Idade',
		int(df_min_max['Idade']['min']), int(df_min_max['Idade']['max']), int(df_sample_true['Idade'].iloc[0]))
	
	months_last_adjustment = st.slider('Meses desde o último aumento ou prêmio',
		int(df_min_max['Meses desde o último aumento ou prêmio']['min']), int(df_min_max['Meses desde o último aumento ou prêmio']['max']), int(df_sample_true['Meses desde o último aumento ou prêmio'].iloc[0]))
		
	pay_alimony = st.selectbox('Paga pensão?', df_cat_vars['Paga pensão?'])	
	
	leader_performance = st.selectbox('Performance do líder', df_cat_vars['Performance do líder'])
	
	previous_position = st.selectbox('Posição na última empresa', df_cat_vars['Posição na última empresa'])
	
	feedbacks_last_12_months = st.selectbox('Quantidade de feedbacks nos últimos 12 meses', df_cat_vars['Quantidade de feedbacks nos últimos 12 meses'])
	
	salary_against_internal = st.selectbox('Salário em relação a outros com o mesmo cargo na equipe', df_cat_vars['Salário em relação a outros com o mesmo cargo na equipe'])
	
	salary_against_external = st.selectbox('Salário em relação a outros com o mesmo cargo no mercado', df_cat_vars['Salário em relação a outros com o mesmo cargo no mercado'])
	
	years_in_company = st.slider('Tempo de empresa',
		int(df_min_max['Tempo de empresa']['min']), int(df_min_max['Tempo de empresa']['max']), int(df_sample_true['Tempo de empresa'].iloc[0]))
	
	years_in_position = st.slider('Tempo no cargo',
		int(df_min_max['Tempo no cargo']['min']), int(df_min_max['Tempo no cargo']['max']), int(df_sample_true['Tempo no cargo'].iloc[0]))
	
	team_turnover = st.selectbox('Turnover da equipe nos últimos 12 meses', df_cat_vars['Turnover da equipe nos últimos 12 meses'])
		
	last_salary_adjustment_against_market = st.slider('% salarial em relação à média de mercado',
		int(df_min_max['% salarial em relação à média de mercado']['min']), int(df_min_max['% salarial em relação à média de mercado']['max']),
		int(df_sample_true['% salarial em relação à média de mercado'].iloc[0]))
	
	last_salary_adjustment_percentage = st.slider('% do último reajuste salarial',
		int(df_min_max['% do último reajuste salarial']['min']), int(df_min_max['% do último reajuste salarial']['max']),
		int(df_sample_true['% do último reajuste salarial'].iloc[0]))

with col2:
	# inference
	df_inference = pd.DataFrame([[age, pay_alimony, distance_work,
		previous_position, education, performance_review,
		gender, overtime,
		salary_against_internal, job_level, leader_performance, marital_status, 
		last_salary_adjustment_against_market, team_turnover,
		last_salary_adjustment_percentage, salary_against_external,
		years_experience, feedbacks_last_12_months, years_in_company, years_in_position,
		months_last_adjustment, leader_experience]],
		columns=['Idade', 'Paga pensão?', 'Distância do trabalho (km)',
	   'Posição na última empresa', 'Formação', 'Avaliação de performance',
	   'Gênero', 'Está fazendo muitas horas extras?',
	   'Salário em relação a outros com o mesmo cargo na equipe', 'Cargo',
	   'Performance do líder', 'Estado civil',
	   '% salarial em relação à média de mercado',
	   'Turnover da equipe nos últimos 12 meses',
	   '% do último reajuste salarial',
	   'Salário em relação a outros com o mesmo cargo no mercado',
	   'Anos de Experiência', 'Quantidade de feedbacks nos últimos 12 meses',
	   'Tempo de empresa', 'Tempo no cargo',
	   'Meses desde o último aumento ou prêmio',
	   'Experiência do líder neste cargo (em anos)'])

	df_inference = df_inference[logical_types.keys()]
	df_inference.ww.init()
	df_inference.ww.set_types(logical_types=logical_types)

	fig_xai, predicao = plot_importances(best_pipeline, df_inference)
	st.plotly_chart(fig_xai)
    
st.markdown('Siga-nos no [Instagram](https://instagram.com/inteledge.lab) e no [LinkedIn](https://www.linkedin.com/company/inteledge/)!')
