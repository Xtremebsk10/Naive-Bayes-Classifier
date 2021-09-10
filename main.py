import pandas as pd
import numpy as np
import time
import random



def read_file(filename):
	df = pd.read_csv(filename, na_values='', sep=';')
	df = df.dropna(how="any")
	return df

def holdout(config):

	print("Reading data and splitting in training/test sets...")

	df = read_file(config['filename'])
	percentage = config['train_percentage'] / 100
	print("Using", round(len(df)*percentage), "tweets to train.")

	train_data = df.sample(frac=percentage)
	test_data = df.drop(train_data.index)

	train_data = train_data.to_numpy()
	test_data = test_data.to_numpy()
	del df
	print("Sets done!")

	voc = create_vocabulary_dict(train_data, config)
	prob_dict, prbs_vj, probs_new_values = naive_bayes(train_data, voc, config)
	predict(test_data, prob_dict, prbs_vj, probs_new_values)


def create_vocabulary_dict(train_data, config):
	print("Creating the vocabulary dictionary...")
	#voc that contains for every word a tuple (count_class_0, count_class_1)
	voc = {}
	#for every entry
	for entry in range(len(train_data)):

		#for every word in the entry tweet
		for word in train_data[entry][1].split():

			#if word not exist we initiate it in voc depending on its value
			if word not in voc.keys():
				
				#if fixed size is active and we reach the desired dict length, we return voc
				if config['vocab_full'] == False and len(voc) == config['vocab_size']:
					return voc
				
				#if class is 1, we initiate the count as class0 = 0, class1 = 1 and inverse in the other case
				if train_data[entry][3] == 1:
					voc[word] = (0,1)
				else:
					voc[word] = (1,0)

			#if word exist we add 1 to the counter of its value
			else:

				if train_data[entry][3] == 1:
					voc[word] = (voc[word][0], voc[word][1] + 1)
				else:
					voc[word] = (voc[word][0] + 1, voc[word][1])
	print("Vocabulary dictionary done!")
	return voc

def naive_bayes(train_data, voc, config):
	print("Processing probabilities of every word...")
	prob_dict = {}
	prbs_vj = [0,0]
	probs_new_values = [0,0]
	#Naive Bayes algorithm based on (same names) the pdf example 
	for value_vj in range(2):

		#subset of examples for which value is vj
		docs = train_data[np.where(train_data[:,3] == value_vj)]

		#P(vj)
		p_vj = len(docs)/len(train_data)
		prbs_vj[value_vj] = p_vj

		#number of words of docs (len of text_j)
		#n = sum(map(len, docs.tweetText.str.split()))
		n = 0
		for tweet in docs[:,1]:
			n += len(tweet.split())
		#n = sum(map(len, docs[:,1][:].split()))

		probs_new_values[value_vj] = (config['laplace'] / (n + (len(voc)*config['laplace'])))

		#for word wk in vocabulary
		for word_wk in voc.keys():

			#nk number of times word occurs in text_j (or word have class value_vj)
			nk = voc[word_wk][value_vj]

			#P(wk|vj)
			p = (nk + config['laplace']) / (n + (len(voc)*config['laplace']))
			#add probability to our dictionary of probabilities
			if word_wk not in prob_dict.keys():
				prob_dict[word_wk] = [0,0]

			prob_dict[word_wk][value_vj] = p

	print("Probabilities processed!")
	return prob_dict, prbs_vj, probs_new_values

def predict(test_data, prob_dict, prbs_vj, probs_new_values):
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for ind in range(len(test_data)):
		prbs = [1,1]

		#for every word in the current tweet
		for w in test_data[ind][1].split():
			#if word exists in our data
			#if w in prob_dict.keys():
			
			if prob_dict.get(w) != None:
				#for every index (0,1) we calculate the chain multiplication of its probabilities
				for val in range(2):
					prbs[val] *= prob_dict[w][val]
			
			else:
				#if not existant we give values depending on laplace smoothing, if 0 then [0,0] if 1 then [xvalue,yvalue]
				for val in range(2):
					prbs[val] *= probs_new_values[val]
		
			

		#finally we add P(vj) to every position
		prbs[0] *= prbs_vj[0]
		prbs[1] *= prbs_vj[1]

		#predict
		if prbs[0] > prbs[1]:

			prediction = 0
		else:

			prediction = 1

		#True Positive
		if prediction == test_data[ind][3] and prediction == 1:
			TP += 1
		#True Negative
		elif prediction == test_data[ind][3] and prediction == 0:
			TN += 1
		#False Positive
		elif prediction != test_data[ind][3] and prediction == 1:
			FP += 1
		#False Negative
		elif prediction != test_data[ind][3] and prediction == 0:
			FN += 1
	print("---------------- RESULTS ----------------")
	print("Accuracy:", (TP+TN)/(TP+TN+FP+FN))
	print("Precision:", TP/(TP+FP))
	print("Recall:", TP/(TP+FN))
	print("-----------------------------------------")



#starts timer
t0 = time.time()

#dictionary with configs like if we want laplace
#laplace = 1 if we want to apply laplace smoothing, 0 if not
#train_percentage is the % of data we want to train and 100-train_percentage for test
#vocab_size if 0 we use the complete dict. if any other number (minor than maximum of words) sets size of vocabulary dict
#filename is file name of our dataset
#vocab_full is a boolean we use to control if we want the full dictionary (true) or use the value given in vocab_size
config = {'laplace':1, 'train_percentage':70, 'vocab_size':0,'vocab_full':True, 'filename':"FinalStemmedSentimentAnalysisDataset.csv"}
holdout(config)

t1 = time.time()
print(t1-t0, " seconds.")
