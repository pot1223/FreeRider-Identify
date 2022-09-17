# ----------------------------------- 사용한 라이브러리 ------------------------------------------ # 

!pip install datatable
!pip install soynlp
!pip install sklearn
!pip install scipy

from soynlp.word import WordExtractor
import datatable as dt 
import pandas as pd 
import re
from soynlp.tokenizer import LTokenizer
from soynlp.word import WordExtractor
from soynlp.utils import DoublespaceLineCorpus
from collections import Counter 
from wordcloud import WordCloud
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import numpy as np
from datetime import datetime
from copy import copy
import matplotlib.pyplot as plt
import seaborn as sns 

# ------------------------- 한글 사용 코드 ---------------------------- # 
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

# ------------------------- 워드 클라우드 할 때 사용 --------------------------- #

#%config InlineBackend.figure_format = 'retina'
 
#!apt -qq -y install fonts-nanum
 
#import matplotlib.font_manager as fm
#fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
#font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
#mpl.font_manager._rebuild()


# --------------------- 카카오톡 txt 데이터를 받아서 Series 형태로 변환하는 함수 ---------------- #

def get_text(file) :
  datatable_df = dt.fread(file , sep ='\t' , encoding ='utf-8')
  df = datatable_df.to_pandas()
  rawText = df.iloc[: , 0]
  return rawText

# --------------------------- Series 형태를 데이터 프레임 형태로 변환하는 함수 ---------------- #

def get_dataFrame(rawText) : 
  content_all = []
  for text in rawText :
    content = text.split(',')
    try :
      content2 = content[1].split(':')
      content_all.append(content[0] + ',' + content2[0] + ',' + content2[1])
    except :
      continue

  t = []  
  n = []
  w = []

  for i in content_all :
    ii = i.split(',' , 2)
    t.append(ii[0])
    n.append(ii[1])
    w.append(ii[2])

  rawdf = pd.DataFrame(data = t ,columns = ['날짜'])
  rawdf['이름'] = n
  rawdf['내용'] = w
  return rawdf

# ---------------------------------- 데이터프레임에서 날짜와 시간을 분리하는 함수 ------------------------- #

def separate_daytime(rawdf) :

  # -------------------- 오전을 AM으로 오후를 PM 으로 변경 ------------------------------#

  rawdf['날짜'] = rawdf['날짜'].map(lambda x : x.replace("오전","AM"))
  rawdf['날짜'] = rawdf['날짜'].map(lambda x : x.replace("오후","PM"))


  # ------------ YYYY년 M월 DD일 PM/AM hh:mm 데이터를 날짜와 시간으로 구분하기 ------------------ # 

  datedata  = rawdf.iloc[:,0]
  date_pattern = re.compile("([0-9]+년 [0-9]+월 [0-9]+일)")
  am_time_pattern = re.compile("AM [0-9]+:[0-9]+") 
  pm_time_pattern = re.compile("PM [0-9]+:[0-9]+") 
  datelst = []
  pm_timelst = []
  am_timelst = []

  for date in datedata :
    dates = date_pattern.findall(date)
    dates = ''.join(dates)
    datelst.append(dates)

  #---------- 정규표현식이 잘안되서 오전 , 오후 시간대를 서로 더해서 시간을 구함 --------- #

  for pm_time in datedata :
    pm_times = pm_time_pattern.findall(pm_time)
    pm_times = ''.join(pm_times)
    pm_timelst.append(pm_times)


  for am_time in datedata :
    am_times = am_time_pattern.findall(am_time)
    am_times = ''.join(am_times)
    am_timelst.append(am_times)

  rawdf['날짜'] = datelst
  rawdf['오전'] = am_timelst
  rawdf['오후'] = pm_timelst


  rawdf['시간']= rawdf.iloc[:,3] + rawdf.iloc[:,4]

  rawdf = rawdf.drop(['오전' , '오후'], axis = 1)

  # -----------------  열 순서 바꾸기 -------------------- # 

  rawdf = rawdf[['날짜' , '시간' , '이름' , '내용']]
  return rawdf


#-------------------------------- 날짜 문자열을 period 타입으로 변환 ------------------------- # 

def changeDateType(rawdf) :
  days=[]
  dd = [] 
  for day in rawdf.iloc[:,0] :
    days.append(pd.to_datetime(day , format = "%Y년 %m월 %d일"))
  for d in days :
    dd.append(d.to_period(freq='D'))
  rawdf["날짜"] = dd 
  return rawdf


#----------------------- ㅋㅋㅋㅋ 나 ㅎㅎ 및 특수 기호를 제외하는 함수 ---------------# 

def data_cleansing_text(rawdf):
    cleandf = copy(rawdf)
    pattern = '[^\w\s]' # 문자나 공백을 제외한 패턴으로 특수기호를 제외하기 위해 사용한다 
    cleandf["내용"] = cleandf["내용"].apply(lambda x : re.sub(pattern,"",x)) # 해당 패턴을 제외
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)' # 한글 자음 모음 패턴으로 ㅋㅋㅋㅋ 나 ㅎㅎ을 제외하기 위해 사용한다 
    cleandf["내용"] = cleandf["내용"].apply(lambda x : re.sub(pattern,"",x)) # 해당 패턴을 제외
    return cleandf  

# ------------------  사진과 이모티콘을 제외하는 함수 -------------------- # 

def changePicEmo(cleandf) : 
  csdf = cleandf
  csdf['내용'] = csdf['내용'].map(lambda x : x.replace("사진",""))
  csdf['내용'] = csdf['내용'].map(lambda x : x.replace("이모티콘",""))
  return csdf

# -----------------------------  사용 단어 빈도수를 나타내는 데이터 프레임 만드는 함수(soynlp 형태소 분석기 사용) -------------------------------------- # 

def usedword(csdf) : 
  word_extractor = WordExtractor(min_frequency=100,
      min_cohesion_forward=0.05, 
      min_right_branching_entropy=0.0
  )
  word_extractor.train(csdf['내용'].values) # list of str or like
  words = word_extractor.extract()

  cohesion_score = {word:score.cohesion_forward for word, score in words.items()}
  tokenizer = LTokenizer(scores=cohesion_score)

  csdf['tokenizer'] = csdf['내용'].apply(lambda x : tokenizer.tokenize(x , remove_r = True))
  csdf['tokenizer']

  words = [] 
  for i in csdf['tokenizer'].values : 
    for k in i:
      words.append(k) 
  count = Counter(words)
  words_dict =dict(count)

  stopword = {'잘','뭐','왜','헐','하면','듯','걍' , 'https' , '그게' ,'내'}

  for word in stopword:
     words_dict.pop(word)
  return words_dict

# ------------------------------------------------- 워드 클라우드 보여주는 함수--------------------------------------------------- # 

def wordcloud(words_dict) : 
  wordcloud = WordCloud(font_path='/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf', width=500, height=500, background_color='white').generate_from_frequencies(words_dict)
  plt.figure(figsize=(10,10))
  plt.imshow(wordcloud)
  plt.axis('off')
  return plt.show()

# ---------------------------- 활동성 점수  ------------------------ #

def sendPicEmo(cleandf) :
  nums = []
  count = 0  
  for i in cleandf.loc[ : , '내용'] :
    if (len(re.compile("(사진 [0-9]장)").findall(i)) > 0) | (i == ' 사진') | (i == ' 이모티콘') :  # 주의! 사진 , 이모티콘 글자 앞에 띄어쓰기가 있음
        nums.append(count)
    count += 1 
  picEmodf = cleandf.loc[nums]
  picEmodf['횟수'] = [1]*len(picEmodf)
  picEmodf['주차'] = picEmodf.날짜.dt.week
  wpicEmodf = picEmodf.groupby('주차').sum()  
  picEmodf = picEmodf.merge(wpicEmodf , how = 'inner' , left_on = '주차' , right_on = '주차')
  picEmodf['활동성점수'] = picEmodf['횟수_x'] / picEmodf['횟수_y']
  picEmodf = picEmodf.drop(['횟수_x' , '횟수_y'], axis = 1)
  picEmodf = picEmodf.groupby(['이름','주차']).sum()
  standard = StandardScaler()
  picEmodf['활동성점수'] = standard.fit_transform(np.array(picEmodf.loc[:,'활동성점수']).reshape(-1, 1))
  rv = norm(loc=0, scale=1)
  picEmodf["활동성점수"] = picEmodf["활동성점수"].apply(lambda x: int(round(rv.cdf(x), 2) * 100))
  return picEmodf

# ----------------------------------------------- 연관성 점수 --------------------------------------------- # 

def word_weight(words_dict,csdf) : 
  word_df = pd.DataFrame([words_dict]).T
  word_df.columns = ['사용횟수']
  word_df = word_df.sort_values('사용횟수' , ascending = False) # 내림차순 
  word_df = word_df.head(100)

  total_num_word = 0 
  for i in word_df['사용횟수'] : 
    total_num_word += i 

  word_df['사용횟수'] = word_df['사용횟수'] / total_num_word 

  weight = [] 
  for i in csdf['tokenizer'] :
    score = 0 
    for j in word_df.index : 
      if j in i :
        score += word_df.loc[j,'사용횟수']
    weight.append(score)   
  csdf['연관성가중치'] = weight

  csdf['주차'] = csdf.날짜.dt.week
  wrawdf = csdf.groupby('주차').sum()
  worddf = csdf.merge(wrawdf , how = 'inner' , left_on = '주차' , right_on = '주차')
  worddf['연관성점수'] = worddf['연관성가중치_x'] / worddf['연관성가중치_y']
  worddf = worddf.drop(['연관성가중치_x' , '연관성가중치_y'], axis = 1)
  worddf = worddf.groupby(['이름','주차']).sum()

  standard = StandardScaler()
  worddf['연관성점수'] = standard.fit_transform(np.array(worddf.loc[:,'연관성점수']).reshape(-1, 1))
  rv = norm(loc=0, scale=1)
  worddf["연관성점수"] = worddf["연관성점수"].apply(lambda x: int(round(rv.cdf(x), 2) * 100))
  return worddf

# ----------------------------- 적극성 점수 ------------------------- # 
def get_question(rawdf) :
  qdf = copy(rawdf)
  qdf = rawdf[rawdf['내용'].str.contains('\?')] # ?를 포함하는 문자가 있는 데이터프레임 행 추출
  qdf['횟수'] = [1]*len(qdf)
  qdf['주차'] = qdf.날짜.dt.week
  wqdf = qdf.groupby('주차').sum()
  qdf = qdf.merge(wqdf , how = 'inner' , left_on = '주차' , right_on = '주차')
  qdf['적극성점수'] = qdf['횟수_x'] / qdf['횟수_y']
  qdf = qdf.drop(['횟수_x' , '횟수_y'] , axis = 1 )
  qdf = qdf.groupby(['이름' , '주차']).sum()
  standard = StandardScaler()
  qdf['적극성점수'] = standard.fit_transform(np.array(qdf.loc[:,'적극성점수']).reshape(-1,1))
  rv = norm(loc = 0 , scale = 1 )
  qdf['적극성점수'] = qdf['적극성점수'].apply(lambda x : int(round(rv.cdf(x) , 2) * 100))
  return qdf

# --------------------------------  활발성 점수-------------------------------------# 

def text_length_score(rawdf) : 
  tdf = copy(rawdf)
  tdf['횟수'] = [1]*len(tdf)
  tdf['주차'] = tdf.날짜.dt.week 
  wtdf = tdf.groupby('주차').sum()
  tdf = tdf.merge(wtdf , how = 'inner' , left_on = '주차' , right_on = '주차')
  tdf['활발성점수'] = tdf['횟수_x'] / tdf['횟수_y']
  tdf = tdf.drop(['횟수_x' , '횟수_y'] , axis = 1)
  tdf = tdf.groupby(['이름' , '주차']).sum()
  standard = StandardScaler()
  tdf['활발성점수'] = standard.fit_transform(np.array(tdf.loc[:,'활발성점수']).reshape(-1,1))
  rv = norm(loc = 0 , scale = 1)
  tdf['활발성점수'] = tdf['활발성점수'].apply(lambda x : int(round(rv.cdf(x), 2)*100)) 
  return tdf

# ----------------------------------- 속도성 점수 ------------------------------------- #

def fast_answer(rawdf) :
  fdf = copy(rawdf)
  rownum = 0
  for time in fdf.loc[:,'시간'] :
    if len(re.compile("PM").findall(time)) > 0 : 
      fdf.loc[rownum, '시간'] = str(pd.to_datetime(re.compile("[0-9]+:[0-9]+").findall(time)[0], format="%H:%M").hour + 12)+':'+str(pd.to_datetime(re.compile("[0-9]+:[0-9]+").findall(time)[0], format="%H:%M").minute)
    if len(re.compile("AM").findall(time)) > 0 : 
      fdf.loc[rownum, '시간'] = str(pd.to_datetime(re.compile("[0-9]+:[0-9]+").findall(time)[0], format="%H:%M").hour)+':'+str(pd.to_datetime(re.compile("[0-9]+:[0-9]+").findall(time)[0], format="%H:%M").minute)
    if re.compile("[0-9]+:").findall(fdf.loc[rownum, '시간'])[0] == "24:" :
      fdf.loc[rownum, '시간'] = '0'+re.compile(":[0-9]+").findall(time)[0]
    rownum += 1
  fdf["시간"] = pd.to_datetime(fdf["시간"], format="%H:%M")
  fdf['간격'] = fdf['시간'] - fdf['시간'].shift(1)
  fdf["답장간격"] = fdf["간격"].apply(lambda x: x.seconds // 60) # 분으로 간격 구하기 
  fdf["동일인물"] = (fdf["이름"] != fdf["이름"].shift(1)) # 연속으로 같은 사람이 나오면 False 값이 된다 
  fdf = fdf[fdf["동일인물"]]
  fdf = fdf.reset_index(drop=True)
  weight = [0]
  for j in range(1,len(fdf)):
    for i in range(24):
      if i * 60 <= fdf.loc[j , '답장간격'] < (i + 1) * 60 : 
        weight.append((24 - i) / 24) # 1시간 내는 23/24, 2시간 내는 22/24 등으로 카톡시간이 늦으면 가중치가 줄어든다 
  fdf['속도성점수'] = weight # test 에 User 별 Interval에 따른 Weight 추가해주기
  fdf['주차'] = fdf.날짜.dt.week 
  wfdf = fdf.groupby('주차').sum()
  fdf = fdf.merge(wfdf , how = 'inner' , left_on = '주차' , right_on = '주차')
  fdf['속도성점수'] = fdf['속도성점수_x'] / fdf['속도성점수_y']
  fdf = fdf.drop(['속도성점수_x' , '속도성점수_y'] , axis = 1)
  fdf = fdf.groupby(['이름' , '주차']).sum()
  fdf = fdf.drop(['답장간격_x' , '답장간격_y' , '동일인물_x' , '동일인물_y'] , axis = 1)
  standard = StandardScaler()
  fdf['속도성점수'] = standard.fit_transform(np.array(fdf.loc[:,'속도성점수']).reshape(-1,1))
  rv = norm(loc = 0 , scale = 1)
  fdf['속도성점수'] = fdf['속도성점수'].apply(lambda x : int(round(rv.cdf(x), 2)*100)) 
  return fdf

# -------------------- 총 점수(가중치는 변경가능) ---------------------- # 

def weightfunc(picEmodf , worddf , qdf , tdf , fdf) : 
  resultdf = picEmodf.join(worddf , how = 'outer').join(qdf , how = 'outer').join(tdf , how = 'outer').join(fdf , how = 'outer')
  resultdf = resultdf.fillna(0)
  resultdf['총점수'] = resultdf['활동성점수']*0.2 + resultdf['연관성점수']*0.2 + resultdf['적극성점수']*0.2 + resultdf['활발성점수']*0.2 + resultdf['속도성점수']*0.2
  return resultdf

# --------------------------- csv 파일로 저장(총점수가 있는 파일) ------------------------------ # 

def savecsv(resultdf) :
  csvdf = copy(resultdf)
  csvdf = csvdf.reset_index() 
  save = csvdf.to_csv('/content/resultKakaoTalkChats.csv')
  return save

# --------------------------- csv 파일로 저장 (날짜 데이터가 있는 파일) ------------------------------ # 
def savecsv2(rawdf) :
  rcsvdf = copy(rawdf)
  rsave = rcsvdf.to_csv('/content/rawresult.csv')
  return rsave

# ------------------ 실행 ----------------------  # 

if __name__ == '__main__' : 
  file = '/content/KakaoTalkChats.txt'
  rawText = get_text(file)
  rawdf = get_dataFrame(rawText)
  rawdf = separate_daytime(rawdf)
  rawdf = changeDateType(rawdf) # 정제되지 않은 df  
  cleandf = data_cleansing_text(rawdf) # 형태소 분석기 실행된 df  
  picEmodf = sendPicEmo(cleandf) # 활동성점수 
  csdf	= changePicEmo(cleandf) # 완전 정제된 df (cleasingdf, 사진/이모티콘 제외)
  words_dict = usedword(csdf)
  #wordcloud(words_dict) # 워드 클라우드를 보고 싶으면 사용
  worddf = word_weight(words_dict, csdf) # 연관성점수
  qdf = get_question(rawdf) # 적극성점수
  tdf = text_length_score(rawdf) # 활발성점수
  fdf =fast_answer(rawdf) # 속도성점수
  resultdf = weightfunc(picEmodf , worddf , qdf , tdf , fdf) # 총 점수
  save = savecsv(resultdf)
  rsave = savecsv2(rawdf)

