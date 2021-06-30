# Core Pkgs
import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns 
import altair as alt
from datetime import datetime

# Online ML Pkgs
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords,TFIDF
from river.compose import Pipeline

# Training Data
#Features are:
			# department,region,education,
			# gender,recruitment_channel,
			# no_of_trainings,age,previous_year_rating,
			# length_of_service,KPIs_met >80%,awards_won?,
			# avg_training_score,
#
data = [("Sales & Marketing,region_7,Master's & above,f,sourcing,1,35,5,8,0,49","DEP_1"),
("Operations,region_22,Bachelor's,m,other,1,30,5,4,0,60","DEP_2"),
("Sales & Marketing,region_19,Bachelor's,m,sourcing,1,34,3,7,0,50","DEP_3"),
("Sales & Marketing,region_23,Bachelor's,m,other,2,39,1,10,0,50","DEP_2"),
("Technology,region_26,Bachelor's,m,other,1,45,3,2,0,73","DEP_1"),
("Analytics,region_2,Bachelor's,m,sourcing,2,31,3,7,0,85","DEP_3"),
("Operations,region_20,Bachelor's,f,other,1,31,3,5,0,59","DEP_2"),
("Operations,region_34,Master's & above,m,sourcing,1,33,3,6,0,63","DEP_3"),
("Analytics,region_20,Bachelor's,m,other,1,28,4,5,0,83","DEP_2"),
("Sales & Marketing,region_1,Master's & above,m,sourcing,1,32,5,5,0,54","DEP_1"),
("Technology,region_23,,m,sourcing,1,30,,1,0,77","DEP_3"),
("Sales & Marketing,region_7,Bachelor's,f,sourcing,1,35,5,3,0,50","DEP_3"),
("Sales & Marketing,region_4,Bachelor's,m,sourcing,1,49,5,5,0,49","DEP_1"),
("Technology,region_29,Master's & above,m,other,2,39,3,16,0,80","DEP_2"),
("R&D,region_2,Master's & above,m,sourcing,1,37,3,7,0,84","DEP_1"),
("Operations,region_7,Bachelor's,m,other,1,37,1,10,0,60,","DEP_2"),
("Technology,region_2,Bachelor's,m,other,1,38,3,5,0,77","DEP_3"),
("Sales & Marketing,region_31,Bachelor's,m,other,1,34,1,4,0,51","DEP_4"),
("Sales & Marketing,region_31,Bachelor's,m,other,1,34,5,8,0,46","DEP_4"),
("Operations,region_15,Bachelor's,m,other,1,37,3,9,0,59","DEP_2")]
	
	



# Model Building
model = Pipeline(('vectorizer',BagOfWords(lowercase=True)),('nv',MultinomialNB()))
for x,y in data:
	model = model.learn_one(x,y)

# Storage in A Database
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()
# department,region,education,gender,recruitment_channel,no_of_trainings,age,previous_year_rating,length_of_service,KPIs_met >80%,awards_won?,avg_training_score,is_promoted
# SQL STUFF
def create_table():
	c.execute('CREATE TABLE IF NOT EXISTS predictionTable(message TEXT,prediction TEXT,probability NUMBER,DEP_1 NUMBER,DEP_2 NUMBER,DEP_3 NUMBER,DEP_4 NUMBER,postdate DATE)')


def add_data(message,prediction,probability,DEP_1,DEP_2,DEP_3,DEP_4,postdate):
    c.execute('INSERT INTO predictionTable(message,prediction,probability,DEP_1,DEP_2,DEP_3,DEP_4,postdate) VALUES (?,?,?,?,?,?,?,?)',(message,prediction,probability,DEP_1,DEP_2,DEP_3,DEP_4,postdate))
    conn.commit()

def view_all_data():
	c.execute("SELECT * FROM predictionTable")
	data = c.fetchall()
	return data


# main function
value="Sales & Marketing,region_7,Master's & above,f,sourcing,1,35,5,8,0,49"
def main():
	menu = ["Home","Manage"]
	create_table()
	
	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.subheader("Home")
		with st.form(key='mlform'):
			col1,col2 = st.beta_columns([2,1])
			with col1:
				with st.beta_container():
					st.write("Features:- department,region,education,gender,recruitment_channel,")
					st.write("no_of_trainings,age,previous_year_rating,")
					st.write("length_of_service,awards_won?,avg_training_score")
				message = st.text_area("message",value=value)
				submit_message = st.form_submit_button(label='Predict')

			with col2:
				st.write("Using River")
				st.write("Predict as DEP_1, DEP_2, DEP_3, DEP_4")

		if submit_message:
			prediction = model.predict_one(message)
			prediction_proba = model.predict_proba_one(message)	
			probability = max(prediction_proba.values())
			postdate = datetime.now()
			# Add Data To Database
			add_data(message,prediction,probability,prediction_proba['DEP_1'],prediction_proba['DEP_2']
			,prediction_proba['DEP_3'],prediction_proba['DEP_4'],postdate)
			st.success("Data Submitted")


			res_col1 ,res_col2 = st.beta_columns(2)
			with res_col1:
				st.info("Original Text")
				st.write(message)

				st.success("Prediction")
				st.write(prediction)

			with res_col2:
				st.info("Probability")
				st.write(prediction_proba)

				# Plot of Probability
				df_proba = pd.DataFrame({'label':prediction_proba.keys(),'probability':prediction_proba.values()})
				# st.dataframe(df_proba)
				# visualization
				fig = alt.Chart(df_proba).mark_bar().encode(x='label',y='probability')
				st.altair_chart(fig,use_container_width=True)	





	elif choice == "Manage":
		st.subheader("Manage")
		stored_data =  view_all_data() 
		new_df = pd.DataFrame(stored_data,columns=['message','prediction','probability','DEP_1','DEP_2','DEP_3','DEP_4','postdate'])
		st.dataframe(new_df)
		new_df['postdate'] = pd.to_datetime(new_df['postdate'])

		
		c = alt.Chart(new_df).mark_line().encode(x='postdate',y='probability')
		st.altair_chart(c)

		c_DEP_1 = alt.Chart(new_df['DEP_1'].reset_index()).mark_line().encode(x='DEP_1',y='index')
		c_DEP_2 = alt.Chart(new_df['DEP_2'].reset_index()).mark_line().encode(x='DEP_2',y='index')
		c_DEP_3 = alt.Chart(new_df['DEP_3'].reset_index()).mark_line().encode(x='DEP_3',y='index')
		c_DEP_4 = alt.Chart(new_df['DEP_4'].reset_index()).mark_line().encode(x='DEP_4',y='index')
		
		c1,c2,c3,c4 = st.beta_columns(4)
		with c1:
			with st.beta_expander("DEP_1 Probability"):
				st.altair_chart(c_DEP_1,use_container_width=True)

		with c2:
			with st.beta_expander("DEP_2 Probability"):
				st.altair_chart(c_DEP_2,use_container_width=True)
		with c3:
			with st.beta_expander("DEP_3 Probability"):
				st.altair_chart(c_DEP_3,use_container_width=True)
		with c4:
			with st.beta_expander("DEP_4 Probability"):
				st.altair_chart(c_DEP_4,use_container_width=True)

if __name__ == '__main__':
	main()


