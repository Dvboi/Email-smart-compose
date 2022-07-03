import streamlit as st
import nltk
import re
import gpt_2_simple as gpt2 
from keras import backend as K

# load the model globally
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='run1')

@st.cache()


def remove_extensions(text):
	'''
	We removed attachments while extracting body but not the name of these attachments
	removing attachment_names based on what i encountered in subject and body
	'''
	ext_patterns = ["\S+\.doc","\S+\.jpeg","\S+\.jpg","\S+\.gif","\S+\.csv","\S+\.ppt","\S+\.dat","\S+\.xml","\S+\.xls","\S+\.sql","\S+\.nsf","\S+\.jar","\S+\.bin","\S+\.txt"]
	pattern = '|'.join(ext_patterns)
	text = re.sub(pattern,'',text)
	return text

def remove_personal_name(text):
	'''
	Helper function to Filter out names using NER
	'''
	s = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
	for ele in s:
		if isinstance(ele, nltk.Tree):
			if ele.label()=='PERSON':
				for word,pos_tag in ele:
					try:     # words containing a special character will raise an error so handling it, these words weren't a name so we can safely skip it
						val = re.sub(word,'',text)
						text = val
					except:
						continue
	return text

def decontracted(phrase):
	"""
	Returns decontracted phrases
	"""
	# specific
	phrase = re.sub(r"won't", "will not", phrase)
	phrase = re.sub(r"can\'t", "can not", phrase)
	phrase = re.sub(r"ain\'t", "am not", phrase)
	phrase = re.sub(r"let\'s", "let us", phrase)
	# general
	phrase = re.sub(r"n\'t", " not", phrase)
	phrase = re.sub(r"\'re", " are", phrase)
	phrase = re.sub(r"\'s", " is", phrase)
	phrase = re.sub(r"\'d", " would", phrase)
	phrase = re.sub(r"\'ll", " will", phrase)
	phrase = re.sub(r"\'t", " not", phrase)
	phrase = re.sub(r"\'ve", " have", phrase)
	phrase = re.sub(r"\'m", " am", phrase)
	return phrase

def remove_timestamps(text):
	'''
	Remove all types of 'text' data from timestamps
	'''
	text = text.replace('AM','')
	text = text.replace('PM','')
	text = text.replace('A.M.','')
	text = text.replace('P.M.','')
	text = text.replace('a.m.','')
	text = text.replace('p.m.','')
	text = re.sub(r"\bam\b",'',text)
	text = re.sub(r"\bpm\b",'',text)
	return text

def final_transform(text):
	'''
	We clean the full text/body using regex and other cleaning functions
	'''
	# remove URL's
	remove_url = r'(www|http)\S+'     # https://stackoverflow.com/a/40823105
	remove_phone = '(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'   # ONLY US numbers for now --> https://stackoverflow.com/a/16699507

	#remove ANY emails
	remove_email = r'\S+@\S+'  # https://stackoverflow.com/a/64036475


	pattern_list_1 = [remove_url,remove_phone,remove_email]

	for pattern in pattern_list_1:
		text = re.sub(pattern,'',text)

	# remove attachment_names
	text = remove_extensions(text)

	# remove any word with digit
	text = re.sub(r'\w*\d\w*', '', text)

	# remove any digit
	text = re.sub('\d','',text)

	# remove text between <>,()
	remove_tags = r'<.*>'
	remove_brackets = r'\(.*\)'
	remove_special_1 = r'\\|-'  # remove raw backslash or '-'
	remove_colon = r'\b[\w]+:' # removes 'something:'

	pattern_list_2 = [remove_tags,remove_brackets,remove_special_1,remove_colon]
	for pattern in pattern_list_2:
		text = re.sub(pattern,'',text)

	# remove anything which is not a character,apostrophy ; remember to give a space on replacing with this
	remove_nonchars = r'[^A-Za-z\']'
	text = re.sub(remove_nonchars,' ',text)

	# remove AM/PM as we have a lot of timestamps in emails
	text = remove_timestamps(text)

	# remove personal names using named entity recognition
	text = remove_personal_name(text)

	# takes care of \t & \n ; remember to give a space on replacing with this
	remove_space = r'\s+'
	text = re.sub(remove_space,' ',text)

	# take care of apostrophies
	text = decontracted(text)

	# remove other junk
	text = text.replace("IMAGE",'')
	text = re.sub(r"\bth\b",'',text)

	return text.strip()

def predict(sent):   
	# check length of sentence
	MAX_LEN = 30
	sent = ' '.join(sent.strip().split()[:MAX_LEN])
	# PREPROCESS
	sent = final_transform(sent)
	# inference
	prefix="<|startoftext|> "+sent
	p = gpt2.generate(sess,
				prefix=prefix,
				truncate="<|endoftext|>",
				length=MAX_LEN,
				run_name='run1',
				temperature=0.7,
				include_prefix=True,    
				return_as_list=True
				)[0]
				
	p = p[len(prefix):]
	return p.strip()


# main frontend code

if __name__ == '__main__':

	st.title("Email Smart-Compose")

	st.text_input("Enter the email prefix", key="email_prefix")
	if st.session_state.email_prefix:
		sent = final_transform(st.session_state.email_prefix)
		
		st.subheader("Predicted auto-complete:")
		with st.spinner('Wait for it...'):
		    output = predict(sent)
		st.success(output)

		st.subheader("Full Sentence:")
		st.markdown(f'<h1 style="color:#33ff33;font-size:18px;">{str(st.session_state.email_prefix) + " " + str(output)}</h1>', unsafe_allow_html=True)
	