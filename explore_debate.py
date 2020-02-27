import pandas as pd
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
# from nltk.stem.porter import *
import gensim
plt.style.use('ggplot')

### STEP 1: Get the Data
#read the data from the csv
df = pd.read_csv("debate_transcripts.csv",encoding='cp1252')

### STEP 2: Explore the Data and get basic metrics/visuals
#create a list of democratic candidates
cur_dem_candidates = ['Joe Biden', 'Bernie Sanders', 'Amy Klobuchar',
       'Tom Steyer', 'Andrew Yang', 'Elizabeth Warren', 'Pete Buttigieg',
       'John Delaney','Michael Bennet', 'Bennett', 'Tulsi Gabbard']
        
df_candidates = df[df['speaker'].isin(cur_dem_candidates)]
#lets get a basic info on the data that we have about our candidates

basic_aggrigations = {'debate_name':'nunique', 'speaking_time_seconds':'sum', 'debate_section':'count'}
basic_columns = {'debate_name': 'Total Debates', 'speaking_time_seconds': 'Total Speaking Time (s)', \
              'debate_section': 'Speaking Opportunities'}

candid_df = df_candidates.groupby('speaker').agg(basic_aggrigations)\
                     .rename(columns=basic_columns)\
                     .sort_values(by=['Total Speaking Time (s)'], ascending=True)\
                     # .reset_index()
#Graph the aggrigations
# candid_df['Total Speaking Time (s)'].sort_values().plot.barh(title='Total Speaking Time (s)', color='blue')
# candid_df['Speaking Opportunities'].sort_values().plot.barh(title='Speaking Opportunities Across Debates', color='blue')
# candid_df['Total Debates'].sort_values().plot.barh(title='Debates Attended', color='blue')

### Step 3: Data Preprocessing

'''
Write a function to perform the pre processing steps on the entire dataset
'''
stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    tokenizer = RegexpTokenizer(r'\w+')
    additional_stop_words = ['I', 'We', 'people', 'Senator', 'And', 'Thank']
    stop_words=set(stopwords.words("english")+additional_stop_words)
    #.simple_preprocess converts a document to a list of tokens, lower cases, tokenizes, de-accents
    for token in tokenizer.tokenize(text.lower()) :
        if token not in stop_words and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = []
for doc in df_candidates.speech:
       processed_docs.append(preprocess(doc))
       
dictionary = gensim.corpora.Dictionary(processed_docs)
'''
OPTIONAL STEP
Remove very rare and very common words:

- words appearing less than 15 times
- words appearing in more than 10% of all documents
'''
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
'''
Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
words and how many times those words appear. Save this to 'bow_corpus'
'''
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

'''
Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
'''
# TODO
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 5, 
                                   id2word = dictionary,                                    
                                   passes = 2,
                                   workers = 2)


'''
For each topic, we will explore the words occuring in that topic and its relative weight
'''
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

# all_text = ' '.join(df['speech'])
# tokenized_words = word_tokenize(all_text)
# tokenizer = RegexpTokenizer(r'\w+')
# tokenized_words = tokenizer.tokenize(all_text.lower())
# tokenized_words = [w.lower() for w in tokenized_words]
# additional_stop_words = ['I', 'We', 'people', 'Senator', 'And', 'Thank']
# stop_words=set(stopwords.words("english")+additional_stop_words)
# filtered_tokenized_words = [word for word in tokenized_words if word not in stop_words]
# fdist = FreqDist(filtered_tokenized_words)
# fdist.plot(30,cumulative=False)
# plt.show()





### Uncomment this if you want to bring in all non-current candidates
# all_people = ['Joe Biden', 'Bernie Sanders', 'Amy Klobuchar',
#        'Tom Steyer', 'Andrew Yang', 'Elizabeth Warren', 'Pete Buttigieg',
#        'Linsey Davis', 'David Muir', 'Monica Hernandez',
#        'Adam Sexton', 'Devin Dwyer', 'Rachel Scott',
#        'Wolf Blitzer', 'Abby Phillips', 'B. Pfannenstiel', 
#        'Brianne P.', 'Judy Woodruff', 'Amy Walter',
#        'Stephanie Sy', 'Tim Alberta', 'Amna Nawaz',
#        'Yamiche A.', 'Rachel Maddow', 'Andrea Mitchell', 'Kamala Harris',
#        'Cory Booker', 'Kristen Welker', 'Ashley Parker', 'Tulsi Gabbard',
#        'Anderson Cooper', 'Erin Burnett', 'Marc Lacey',
#        'Julian Castro', 'Beto O’Rourke', 'A. Cooper', 'Jake Tapper',
#        'Voiceover', 'Jorge Ramos', 'Sec. Castro', 
#        'Dana Bash', 'Bill de Blasio', 'Michael Bennet', 'Jay Inslee',
#        'Kirsten Gillibrand', 'Don Lemon', 'Crowd', 'Kirseten Gillibrand',
#        'Moderator', 'Diana', 'Steve Bullock',
#        'Marianne Williamson', 'John Delaney', 'Tim Ryan', 'John H.',
#        'Female', 'Male', 'John Hickenloop', 
#        'J. Hickenlooper', 'John King', 'N. Henderson', 'Savanagh G.',
#        'Bennett', 'Jose D.B.', 'Eric Stalwell', 'Eric Swalwell',
#        'Lester Holt', 'Savannah G.', 'Chuck Todd', 'Steve Kornacki']