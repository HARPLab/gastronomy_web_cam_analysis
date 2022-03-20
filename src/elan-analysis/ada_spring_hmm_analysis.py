import copy
import econml
import dowhy
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import cm
from dowhy import CausalModel

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import sys
sys.path.append("..")
# import ada_spring_2022 as main_code

	
def create_heatmap_activity_to_groupstate(df, activitydict_display):
	labels = list(activitydict_display)

	fig = plt.figure(constrained_layout=True)
	fig.set_size_inches(20, 10)

	df_m = pd.melt(df, id_vars=['Meal ID', 'timestamp', 'table-state'], value_vars=['person-A', 'person-B'])

	# print(df[df['person-A'] == 'takeout'].shape)
	# print(df[df['person-B'] == 'takeout'].shape)
	# print(df_m[df_m['value'] == 'takeout'].shape)

	# num_a = df[df['person-A'] == 'takeout'].shape[0]
	# num_b = df[df['person-B'] == 'takeout'].shape[0]
	# num_all = df_m[df_m['value'] == 'takeout'].shape[0]

	# print(num_a / len(df))
	# print(num_b / len(df))
	# print(num_all / len(df))
	
	y_label = "Individual Activity"
	x_label = "Group State"
	df_histo = df_m.groupby(['value', 'table-state']).size().unstack(fill_value=0)

	cm_sum = np.sum(df_histo, axis=0)
	np.seterr(invalid='ignore')
	df_histo = (df_histo / cm_sum.astype(float)) * 100.0
	df_histo = df_histo.T

	coloring = sns.color_palette("light:b", as_cmap=True)
	ax = sns.heatmap(df_histo, annot=True, cmap=coloring, fmt = '.1f', square=True, vmin=-0, vmax=100.0, annot_kws={"size": 14, "weight":'bold'}, cbar=False)
	ax = sns.heatmap(df_histo, mask=df_histo != 0, cmap='Greys', square=True, annot=False, vmin=-0, vmax=100.0, annot_kws={"size": 14, "weight":'bold'}, cbar=False, ax=ax)
	for t in ax.texts: t.set_text(t.get_text() + "%")

	ax.set_xlabel(x_label, fontsize=20, weight='bold')
	ax.set_ylabel(y_label, fontsize=20, weight='bold')	
	ax.set_title("Distribution of Individual Activities within Group States", fontsize=22, weight='bold')

	plt.tight_layout()
	plt.savefig("outputs-2022/relationship_overview.png")
	plt.clf()

	print("Relationship Overview")



def hmm_analyze(df_in, activitydict_display):
	df = copy.copy(df_in)
	codebook_list = activitydict_display
	codebook_to_code	= dict(zip(codebook_list, range(len(codebook_list))))
	codebook_from_code  = dict(zip(range(len(codebook_list)), codebook_list))

	output_set = df['table-state'].unique()

	# print("CODEBOOK")
	# print(codebook)
	n_symbols = len(list(output_set))

	df['code-A'] = df['person-A'].replace(codebook_to_code)
	df['code-B'] = df['person-B'].replace(codebook_to_code)

	filenames_all = list(df['Meal ID'].unique())

	for meal in filenames_all:
		print(meal)
		# hold one out and train on the rest
		bool_test   = (df['Meal ID'] == meal)
		bool_train  = (df['Meal ID'] != meal)
		df_test	= df[bool_test]
		df_train   = df[bool_train]
		# print(len(df_test))
		# print(len(df_train))

		train_A = df_train['code-A'].to_numpy()
		train_B = df_train['code-B'].to_numpy()
		train_X = df_train[['code-A', 'code-B']].to_numpy()

		test_A = df_test['code-A'].to_numpy()
		test_B = df_test['code-B'].to_numpy()
		test_X = df_test[['code-A', 'code-B']].to_numpy()


		# for each hidden state, p emit each obs
		# rows state, cols emission


		from causalgraphicalmodels import CausalGraphicalModel

		model_top_down = CausalGraphicalModel(
			nodes=["group-state", "individual-a", "individual-b"],
			edges=[
				("group-state", "individual-a"), 
				("group-state", "individual-b"), 
			]
		)
		model_top_down.draw()
		
		model_top_down = CausalGraphicalModel(
			nodes=["group-state", "individual-a", "individual-b"],
			edges=[
				("individual-a", "group-state"), 
				("individual-b", "group-state"), 
			]
		)
		model_top_down.draw()

		# # verify dimensions
		# print(test_A.shape)
		# print(test_B.shape)
		# print(test_X.shape)
		# print(train_A.shape)
		# print(train_B.shape)
		# print(train_X.shape)

		# Define causal model
		# model = CausalModel(
		# 	data=df,
		# 	common_causes=["individual-a", "individual-b"],
		# 	outcome=["group-state"])
		# model.view_model(layout="dot")

		identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
		print(identified_estimand)

		# calculate the emission probabilities

		# calculate the transmission probabilities


		# Given individual A and B, predict the group state
		hmm_all = hmm.MultinomialHMM(n_components=8)
		model_all = hmm_all.fit(train_X)
		prediction_all = hmm_all.predict(test_X)

		# Given individual A, predict the group state
		hmm_a = hmm.MultinomialHMM(n_components=8)
		model_a = hmm_a.fit(train_A)
		prediction_a = hmm_a.predict(test_A)

		# Given individual B, predict the group state
		hmm_b = hmm.MultinomialHMM(n_components=8)
		model_b = hmm_b.fit(train_B)
		prediction_b = hmm_b.predict(test_B)





	# for pred_a, pred_b, pred_all, decode_a in zip(prediction_a, prediction_b, prediction_all):
	#	 both_same = (pred_a == pred_b)
	#	 customers_same.append(both_same)

	#	 overall_graph.append(abs(pred_a - pred_b))

	#	 pred_all = (pred_a == pred_b == pred_all)
	#	 all_same.append(pred_all)

	#	 decode_predict.append(decode_a == pred_a)

