import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import gc
import math
import os
if(not os.path.exists(os.path.dirname(__file__)+"\ProcessedRatings_110.csv")):
    df = pd.read_csv("https://raw.githubusercontent.com/CoolSmit007/Product-Recommender/main/ProcessedRatings_110.csv", error_bad_lines=False)
    df.to_csv("ProcessedRatings_110.csv",index=False)
    del(df)
    gc.collect()
if(not os.path.exists(os.path.dirname(__file__)+"\Processed_data.csv")):
    df = pd.read_csv(csv_url, error_bad_lines=False)
    df.to_csv("Processed_data.csv",index=False)
    del(df)
    gc.collect()
if(not os.path.exists(os.path.dirname(__file__)+"\Correcollabfilter.csv")):
    df21=pd.read_csv("ProcessedRatings_110.csv")
    df21=df21.pivot_table(index=['Reviewid'],columns=['Productid'],values='Rating')
    df21=df21.dropna(thresh=20,axis=1).fillna(0)
    df21=df21.corr(method='pearson')
    df21.to_csv("Correcollabfilter.csv")
    del(df21)
    gc.collect()

@st.cache_resource
def Model():
    df = pd.read_csv('processed_data.csv')
    df=df[['asin','title']]
    gc.collect()
    df21=pd.read_csv("Correcollabfilter.csv",index_col=0)
    vectorizer = TfidfVectorizer(stop_words='english',token_pattern=r'\w{1,}',strip_accents='unicode',analyzer='word')
    X1 = vectorizer.fit_transform(df["title"])
    return df,df21,vectorizer,X1
df,df21,vectorizer,X1 = Model()
def giverec2(item):
  X2=vectorizer.transform(item[0])
  sig=sigmoid_kernel(X2,X1)
  sig1=sig.sum(axis=0)
  sig_scores=list(enumerate(sig1))
  sig_scores=sorted(sig_scores,key=lambda x: x[1],reverse=True)
  sig_scores=sig_scores
  p_ind=[i[0] for i in sig_scores]
  del(X2)
  del(sig)
  del(sig1)
  y=pd.DataFrame()
  y['asin']=df['asin'].iloc[p_ind]
  y['title']=df['title'].iloc[p_ind]
  y.set_index('asin',inplace=True)
  y['contentsig']=[i[1] for i in sig_scores]
  y['contentsig']=y['contentsig'].apply(sigmoid)
  del(sig_scores)
  gc.collect()
  return y
def similaritem(item):
  X2=vectorizer.transform([item])
  sig=sigmoid_kernel(X2,X1)
  sig1=sig.sum(axis=0)
  sig_scores=list(enumerate(sig1))
  sig_scores=sorted(sig_scores,key=lambda x: x[1],reverse=True)
  p_ind=[i[0] for i in sig_scores]
  del(X2)
  del(sig)
  del(sig1)
  del(sig_scores)
  gc.collect()
  return df['asin'].iloc[p_ind]
def givereccb(item,rating):
  for z in similaritem(item):
    if(z in df21.columns):
      x=df21[z]*(rating-2.5)
      x=x.sort_values(ascending=False)
      break
  return x
def givereccbs(items):
  x=pd.DataFrame()
  for ind in items.index:
    x=x.append(givereccb(items[0][ind],items[1][ind]),ignore_index=True)
  y=x.sum().sort_values(ascending=False)
  del(x)
  gc.collect()
  y=y.to_frame()
  y[0]=y[0].apply(sigmoid)
  y.rename(columns={'Productid':'asin',0:'collabsig'},inplace=True)
  return y
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def invsigmoid(x):
  return math.log(x/(1-x))
st.header('Enter User Purchase data')
c1,c2=st.columns([4,1])
with c1:
    p1=st.text_input("Product")
with c2:
    r1=st.number_input("Rating",min_value=0,max_value=5,value=3)
if('p' not in st.session_state):
    st.session_state['p']=[]
if st.button("Add Product"):
   st.session_state['p'].append((p1,r1))
   
if(len(st.session_state['p'])>0):
    pd.set_option('display.max_columns', None)
    st.write(pd.DataFrame(st.session_state["p"],columns=['Product','Rating']))
else:
    st.write("No products entered")
if st.button("Give Reccomendations"):
    userhistory=pd.DataFrame(st.session_state["p"])
    colbrativefilteringresult=givereccbs(userhistory)
    contentfilteringresult=giverec2(userhistory)
    result=pd.concat([contentfilteringresult, colbrativefilteringresult], axis=1)
    result['collabsig'].fillna(result.contentsig,inplace=True)
    result['avg']=result[["contentsig","collabsig"]].mean(axis=1)
    resultf=result.sort_values(by=['avg'],ascending=False)
    resultf['Products(In decreasing order of Recommended)']=resultf['title']
    st.dataframe(resultf['Products(In decreasing order of Recommended)'],hide_index=True,use_container_width=True)
    st.write(f"Relevance = {((1-resultf['avg'].iloc[0]+resultf['avg'].iloc[100])*100)}%")