# BFS 구조 도서관 챗봇 만들기 (python 버전)


# 라이브러리 설치
# pip install konlpy
# pip install sentence_transformers


# 0. 모델 및 데이터 로드

import time
strat_time = time.time()

import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from keras.models import load_model
from collections import deque

local_path = 'C:/Users/ypd04/OneDrive/바탕 화면/library_chatbot(python)/'

# 의도 분류 CNN 모델 불러오기 (조회, 추천, 문의 분류)
# 압축 풀어야함
intent_classify_model = load_model(local_path+'models/CNN_library_library_involve_name_3_labels.h5')

# SBERT 모델 불러오기 (문의 세부 분류)
print("<System> sbert 다운 중...")
sbert_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
print("<System> sbert 다운 완료...")

# 추천 (베스트셀러 유무)
recommend_data = pd.read_csv(local_path+'data/csv/recommend_bestseller_whether.csv',encoding='utf-8')
recommend_embedding_data = torch.load(local_path+'data/embedding_data/recommend_bestseller_whether_embedding_data.pt')

# 문의 (5 labels)
inquiry_data = pd.read_csv(local_path+'data/csv/inquiry.csv',encoding='cp949')
inquiry_embedding_data = torch.load(local_path+'data/embedding_data/inquiry_embedding_data.pt')

# 로딩 시간 측정
end_time = time.time()
sec = round(end_time - strat_time,1)
min_ =  int(sec//60)
sec = int(sec % 60)

print(f'모델 및 데이터 로딩 시간은 {min_}분 {sec}초가 걸렸습니다.')

# 1. 전처리 객체 정의, 생성

from konlpy.tag import Komoran

import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
import pickle

class Preprocess:
    def __init__(self, word2index_dic=local_path+'data/dic/chatbot_dict.bin' ,userdic=local_path+'data/dic/userdict_intent_classify_v3(library).txt'): # userdic 인자에는 사용자 정의 사전 파일 경로 입력가능
        
        # 단어 인덱스 사전 불러오기 추가
        if(word2index_dic != ''):
            f = open(word2index_dic, "rb")
            self.word_index = pickle.load(f)
            f.close()
            print("단어 사전 로드 완료..")
        else:
            self.word_index = None
            print("단어 사전 로드 실패..")

        # 형태소 분석기 초기화
        self.komoran = Komoran(userdic=userdic)

        # 제외할 품사
        # 참조 : https://docs.komoran.kr/firststep/postypes.html
        # 관계언 제거, 기호 제거
        # 어미 제거
        # 접미사 제거
        self.exclusion_tags = [
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            # 주격조사, 보격조사, 관형격조사, 목적격조사, 부사격조사, 호격조사, 인용격조사
            'JX', 'JC',
            # 보조사, 접속조사
            'SF', 'SP', 'SS', 'SE', 'SO',
            # 마침표,물음표,느낌표(SF), 쉼표,가운뎃점,콜론,빗금(SP), 따옴표,괄호표,줄표(SS), 줄임표(SE), 붙임표(물결,숨김,빠짐)(SO)
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            # 선어말어미, 종결어미, 연결어미, 명사형전성어미, 관형형전성어미
            'XSN', 'XSV', 'XSA'
            # 명사파생접미사, 동사파생접미사, 형용사파생접미사
        ]

    # 형태소 분석기 POS 태거
    def pos(self, sentence):
        return self.komoran.pos(sentence)

    # 불용어 제거 후 필요한 품사 정보만 가져오기
    def get_keywords(self, pos, without_tag=False):
        f = lambda x: x in self.exclusion_tags
        word_list = []
        for p in pos:
            if f(p[1]) is False:
                word_list.append(p if without_tag is False else p[0])
        return word_list

    # 키워드를 단어 인덱스 시퀀스로 변환
    def get_wordidx_sequence(self, keywords):
        if self.word_index is None:
            return []
        w2i = []
        for word in keywords:
            try:
                w2i.append(self.word_index[word])
            except KeyError:
                # 해당 단어가 사전에 없는 경우 OOV 처리
                w2i.append(self.word_index['OOV'])
        return w2i

p = Preprocess(word2index_dic=local_path+'data/dic/chatbot_dict.bin' ,userdic=local_path+'data/dic/userdict_intent_classify_v3(library).txt')

# 2. Node 클래스 정의

class Node:

  def __init__(self,info):

    # 노드 역할 설명
    self.info = info

    # 노드 별 모델
    self.model = None

    # 노드별 emd 데이터
    self.emd_csv = None
    self.emd_pt = None

    # 데이터 (도서명 , 작가명, 장르, 대출 여부 , 대출 예약 여부)
    self.data = {}

    # 각 노드별 역할(task) 구분 키
    # 기본 key = -1
    self.key = -1

    # 다음 depth에서 제거할 노드의 idx
    self.rmv_idx = -1

  def get_info(self):
    return self.info

  def set_info(self,new_info):
    self.info = new_info

  def get_model(self):
    return self.model

  def set_model(self,model_obj):
    self.model = model_obj

  def get_emd_data(self):
    return self.emd_csv , self.emd_pt

  def set_emd_data(self,emd_csv,emd_pt):
    self.emd_csv = emd_csv
    self.emd_pt = emd_pt

  def get_key(self):
    return self.key

  def set_key(self,new_key):
    self.key = new_key

  def get_data(self):
    return self.data

  def set_data(self,key,value):
    data = self.get_data()
    data[key] = value
    self.data = data 

  def get_rmv_idx(self):
    return self.rmv_idx

  def set_rmv_idx(self,rmv_idx):
    self.rmv_idx = rmv_idx
  
  # 노드 별 역할 
  def task(self,node):

    print(node.get_info())
    next_node = None

    # 의도 분류(조회 or 추천 or 문의)
    if node.get_key() == 1:

      intent_classify_model = node.get_model()
      input_label, user_input = intent_classify(intent_classify_model)

      # CNN 의도분류 모델 Labelencoding 시 문의:0, 조회:1, 추천:2 로 설정함
      reversed_label_idx = {1 : '조회(검색, 예약, 반납)', 2 : '추천', 0 : '문의'}

      # 조회
      if input_label == 1:
        print("next_node = node2")
        next_node = node2
        rmv_idx = 0

      # 추천
      elif input_label == 2:
        print("next_node = node3")
        next_node = node3
        rmv_idx = 1

      # 문의
      elif input_label == 0:
        next_node =node4
        rmv_idx = 2

      # 다음 depth 에서 제거할 노드의 idx 설정
      next_node.set_rmv_idx(rmv_idx)

      # 다음 노드에게 정보 전달함
      next_node.set_data('user_input',user_input)

    # 의도 분류(조회) : 도서명 or 작가명 요청
    elif node.get_key() == 2:

      bname = None
      wname = None

      # 도서명 혹은 작가명 받을 때까지 요청
      while bname == None and wname == None:
        bname, wname = check_bname_wname(node.get_data()['user_input'])

        if bname == None and wname == None:
          print("조회 혹은 예약하시려는 도서명, 작가명을 입력해주세요!\n")
          user_input = input()
          node.set_data('user_input',user_input)
      
      next_node = node5

      # node2 ==> node5 자신의 인덱스를 rmv_idx 로
      # rmv_idx 디폴트 값은 -1 이기 때문
      next_node.set_rmv_idx(0)
      next_node.set_data('bname',bname)
      next_node.set_data('wname',wname)

    # 의도 분류(추천) : 모델링 중
    elif node.get_key() == 3:

      print("의도 분류: 추천")
      print("모델링 중...")
      next_node = None

    # 의도 분류(문의) : 문장 유사도 계산 후 답변 출력
    elif node.get_key() == 4:

      sbert_model = node.get_model()
      emd_csv , emd_pt = node.get_emd_data()

      user_inquiry = node.get_data()['user_input']

      inquiry_ans = check_inquiry_ans(user_inquiry, sbert_model, emd_csv ,emd_pt)
      next_node = None

    # DB에 책 있는지 확인
    elif node.get_key() == 5:
      
      can_search ,is_in_bname , is_in_wname = check_is_in_library(node)

      # 찾음
      if can_search == 0:
        next_node = node6
      # 못찾음
      elif can_search == 1:
        next_node = node7

      next_node.set_rmv_idx(can_search)
      next_node.set_data('bname',is_in_bname)
      next_node.set_data('wname',is_in_wname)
    
    # DB 검색 성공
    elif node.get_key() == 6:

      borrow_bname = node.get_data()['bname']
      borrow_wname = node.get_data()['wname']

      can_borrow_label, borrow_bname, borrow_wname = check_can_borrow(borrow_bname,borrow_wname)

      if can_borrow_label == 0:
        next_node = node8
      elif can_borrow_label == 1:
        next_node = node9
      
      next_node.set_rmv_idx(can_borrow_label)
      next_node.set_data('bname',borrow_bname)
      next_node.set_data('wname',borrow_wname)

    # DB 검색 실패 , node1으로
    elif node.get_key() == 7:

      print("<System> DB 검색 실패")
      # next_node = None
    
    # 대출 예약 가능 확인 , 대출 예약 여부 요청
    elif node.get_key() == 8:

      want_borrow_bname = node.get_data()['bname']
      want_borrow_wname = node.get_data()['wname']

      want_borrow_label = check_want_borrow(want_borrow_bname,want_borrow_wname)

      # print("node 8의 태스크 , next_node =None")

      if want_borrow_label == 0:
        next_node = node10
      elif want_borrow_label ==1:
        next_node = node11
      
      next_node.set_rmv_idx(want_borrow_label)
      next_node.set_data('bname',want_borrow_bname)
      next_node.set_data('wname',want_borrow_wname)

    # 대출 예약 불가 , 반납 알림 여부 요청
    elif node.get_key() == 9:

      want_alarm_bname = node.get_data()['bname']
      want_alarm_wname = node.get_data()['wname']

      want_alarm_label = check_want_alarm(want_alarm_bname,want_alarm_wname)

      # print("node 9의 태스크 , next_node =None")
      
      if want_alarm_label == 0:
        next_node = node12
      elif want_alarm_label ==1:
        next_node = node13
      
      next_node.set_rmv_idx(want_alarm_label)
      next_node.set_data('bname',want_alarm_bname)
      next_node.set_data('wname',want_alarm_wname)

    # 대출 예약 요청 확인
    elif node.get_key() == 10:

      print("<System> 대출 예약 요청")
      # next_node = None

    # 대출 예약 미요청 확인
    elif node.get_key() == 11:

      print("<System> 대출 예약 미요청")
      # next_node = None
    
    # 반납 알림 요청 확인
    elif node.get_key() == 12:

      print("<System> 반납 알림 요청")
      # next_node = None

    # 반납 알림 미요청 확인
    elif node.get_key() == 13:

      print("<System> 반납 알림 미요청")
      # next_node = None
      
    if next_node != None:
      print(next_node.get_info())

    return next_node
  
# 3. 각 Node별 task 함수 정의

## 1) node1: 의도 분류 (조회,추천,문의사항)

def intent_classify(intent_classify_model):

  print(f"챗봇: 안녕하세요. 새싹 스마트 도서관입니다.\n 현재 대화 내용은 보다 더 나은 서비스 개선을 위해 수집될 수 있습니다. \n 무엇을 도와드릴까요? \n")
  
  user_input = input()
  user_input_list = []
  user_input_list.append(user_input)
  
  input_predicted = intent_classify_model.predict(sentences_to_idx(user_input_list))
  print("<System> 의도 예측 확률", input_predicted)
  input_predicted = input_predicted.argmax(axis=-1)
  print("<System> 가장 높은 확률 input_predicted: ", input_predicted)

  # input_predicted 는 array([])
  return input_predicted[0], user_input

def sentences_to_idx(intents_list):
  sequences = []
  check_keywords = True
  # text는 모든 문장들의 list
  for sentence in intents_list:

      # 문장을 [(단어1,품사1),(단어2,품사2)...] 로 변환
      pos = p.pos(sentence)

      # get_keywords(pos, without_tag=True) => 불용어 처리 후 품사(태그)없이 단어들만의 list
      # keywords : 불용어 처리된 [(단어1,품사1),(단어2,품사2)...], list형
      keywords = p.get_keywords(pos, without_tag=True)
      print_keywords = p.get_keywords(pos, without_tag=False)

      # 첫번째 keywords 와 sequence[0] 어떻게 대응되는지 체크해보고 싶음
      if check_keywords is True:
        print(print_keywords)
        check_keywords = False
      # 태그없이 '단어'만 있는 keywords에서 [[단어1,단어2],[단어1,단어2,단어3]...]들을 인덱싱해줌
      # 우리가 만든 단어사전에 없으면(OOV token이므로 인덱스 1로 고정)
      seq = p.get_wordidx_sequence(keywords)
      sequences.append(seq)

  # 조회, 추천, 문의 의도 분류 데이터 tokenize 시 최대 형태소 길이
  max_len = 15

  input_test = preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

  return input_test

## 2) node2: (조회) 도서명, 작가명 Tokenizer 작동

def check_bname_wname(user_input):

  # DB 상 외의 모든 도서명, 작가명 파일 필요
  # 예시로 작성
  book_list = ['크리스마스 피그','데미안','유다','유다2','유다3','유다4','파란 책']
  writer_list = ['J.K.롤링','헤르만 헤세','아모스 오즈','정민']

  bname = None
  wname = None

  pos = p.pos(user_input)
  keywords = p.get_keywords(pos, without_tag=False)
  print("<System> 형태소 분해 : ", keywords)
  for keyword, tag in keywords:
    if tag == 'NNP':
      if keyword in book_list:
        print("<System> 도서명 확인")
        bname = keyword
        # 도서명있으면 작가명 알 수 있음 => return
        return bname ,wname
      elif keyword in writer_list:
        print("<System> 작가명 확인")
        wname = keyword
        # 작가명 알지만 이후 토큰에서 도서명까지 받을 수도 있음

  return bname , wname

## 3) node 3 : (추천) 도서명, 장르명, 베스트셀러 기준으로 도서 추천

# 의도 분류_추천_베스트셀러 여부 모델링 중...

## 4) node4 : (문의) 5가지 기타 문의 사항

def check_inquiry_ans(user_inquiry, sbert_model, emd_csv ,emd_pt):

  sentence = user_inquiry
  model = sbert_model
  data = emd_csv
  embedding_data = emd_pt 

  # 띄어쓰기 제거
  sentence = sentence.replace(" ","")
  # 인코딩
  sentence_encode = model.encode(sentence)
  # 텐서화
  sentence_tensor = torch.tensor(sentence_encode)
  # 텐서화된 입력값과 문의 데이터 비교
  cos_sim = util.cos_sim(sentence_tensor, embedding_data)
  # 가장 큰 문장유사도 인덱스
  best_sim_idx = int(np.argmax(cos_sim))
  # 가장 큰 문장유사도 인덱스의 질문
  sentence_qes = data['input'][best_sim_idx]
  print(f"<System> 선택된 질문 = {sentence_qes}")
  print(f'<System> util.cos_sim 활용 코사인 유사도 : {cos_sim[0][best_sim_idx]}')
  
  # 가장 큰 유사도 인데스에 대응하는 답변 출력
  inquiry_ans = data['output'][best_sim_idx]
  print("챗봇 : ",inquiry_ans)

  return

## 5) node5 : 도서명, 작가명 책이 도서관에 있는지 확인

def check_is_in_library(node):

  # 찾음 = 0 , 못찾음 = 1 
  can_search = 1

  is_in_bname = node.get_data()['bname'] 
  is_in_wname = node.get_data()['wname']
  
  db_bname_list = database['bname'].tolist()
  db_wname_list = database['wname'].tolist()

  if is_in_bname in db_bname_list:
    print("<System> 도서명 기반 검색 완료")
    print(database[database['bname']==is_in_bname])
    can_search = 0
  elif is_in_wname in db_wname_list:
    print("<System> 작가명 기반 검색 완료")
    print(database[database['wname']==is_in_wname])
    can_search = 0
  else:
    print("<System> DB 상 존재하지 않는 도서명, 작가명 입니다.")

  return can_search ,is_in_bname , is_in_wname

## 6) node6 : 대출 가능 여부 확인

def check_can_borrow(borrow_bname,borrow_wname):

  # 대출 예약 가능 여부: 대출 예약 가능 == 0, 대출 예약 불가 == 1
  can_borrow_label = 1

  if borrow_bname != None:
    if database.loc[database['bname']==borrow_bname,'borrowable'].iloc[0] == 0:
      can_borrow_label = 0
      print("챗봇 : 현재 ", borrow_bname," 도서 대출이 가능합니다.")

    else:
      print("챗봇 : 현재 ", borrow_bname," 도서는 대출 중입니다.")

  elif borrow_wname != None:
    database_borrow_wname_borrowable_0 = database.loc[(database['wname']==borrow_wname) & (database['borrowable']==0)]

    if len(database_borrow_wname_borrowable_0) > 1:
      can_borrow_label = 0
      print(database_borrow_wname_borrowable_0)
      print("챗봇 : " , borrow_wname," 작가님의 작품들 중 대출 예약이 가능한 도서 목록입니다.")
    elif len(database_borrow_wname_borrowable_0) == 1:
      can_borrow_label = 0
      print(database_borrow_wname_borrowable_0)
      print("챗봇 : " , borrow_wname," 작가님의 작품들 중 대출 예약이 가능한 도서입니다.")
    else:
      print("챗봇 : " , borrow_wname," 작가님의 작품들은 모두 대출 중입니다.")

  return can_borrow_label , borrow_bname, borrow_wname

## 7) node8 : 대출 가능, 대출 여부 확인

def check_want_borrow(want_borrow_bname,want_borrow_wname):

  # 대출 예약 여부 : 0 => 대출 예약 요청 , 1 => 대출 예약 미요청
  want_borrow_label = 1

  if want_borrow_bname != None :

    print("챗봇 : 대출 예약 해드릴까요? \n (네 혹은 아니오를 눌러주세요.")
    user_input = input()
    if user_input == "네":
      want_borrow_label = 0
      database.loc[database['bname']==want_borrow_bname,'borrowable'] = 1
      print("챗봇: ",want_borrow_bname, "가 대출 예약 되었습니다.")
      print("<System> 대출 예약 요청 O 확인")
      print(database)
    else:
      print("챗봇: 다음에 뵙겠습니다. 감사합니다.")
      print("<System> 대출 예약 요청 X 확인")
      print(database)

  elif want_borrow_wname != None:

    database_want_borrow_wname_borrowable_0 = database.loc[(database['wname']==want_borrow_wname) & (database['borrowable']==0)]

    # 원하는 작가님의 도서가 여러개 일 때
    if len(database_want_borrow_wname_borrowable_0) > 1:
      print("챗봇 : " , want_borrow_wname," 작가님의 작품들 중 대출 예약이 가능한 도서 목록입니다.\n (도서 목록 중 대출을 원하시면 네 혹은 아니오를 눌러주세요.")
      user_input = input()
      if user_input == "네":
        print("챗봇 : 원하시는 도서명을 정확히 입력해주세요.")
        user_input2 = input()
        want_writer_book_list = database_want_borrow_wname_borrowable_0['bname'].tolist()
        if user_input2 in want_writer_book_list:
          want_borrow_label = 0
          database.loc[database['bname']==user_input2,'borrowable'] = 1
          print("챗봇 : ", user_input2 , "가 대출 예약 되었습니다. ")
          print("<System> 대출 요청 O 확인")
          print(database)
      else:
        print("챗봇: 다음에 뵙겠습니다. 감사합니다.")
        print("<System> 대출 요청 X 확인")
        print(database)        
    
    # 원하는 작가님의 도서가 한 개 일 때
    elif len(database_want_borrow_wname_borrowable_0) == 1:
      print("챗봇 : 대출 예약 해드릴까요? \n (네 혹은 아니오를 눌러주세요.")
      user_input = input()
      if user_input == "네":
        want_borrow_label = 0
        database.loc[database['wname']==want_borrow_wname,'borrowable'] = 1        
        print("챗봇 : " , database_want_borrow_wname_borrowable_0['bname'],"가 대출 예약 되었습니다.")
        print("<System> 대출 요청 O 확인")
        print(database)
      else:
        print("챗봇: 다음에 뵙겠습니다. 감사합니다.")
        print("<System> 대출 요청 X 확인")
        print(database)

  return want_borrow_label

## 8) node9 : 대출 불가능, 반납 알림 여부 확인

def check_want_alarm(want_alarm_bname,want_alarm_wname):

  # 반납 알림 여부 : 반납 알림 요청 =>0 , 반납 알림 미요청 =>1
  want_alarm_label = 1

  if want_alarm_bname != None:
    print("챗봇 : " , want_alarm_bname,"이(가) 반납되면 알림 드릴까요?\n (반납 알림을 원하시면 네 혹은 아니오를 눌러주세요.)")
    user_input = input()
    if user_input == "네":
      want_alarm_label = 0
      database.loc[database['bname']==want_alarm_bname,'alarm'] = 1

      print("챗봇 : ", want_alarm_bname," 반납되면 알림 드리겠습니다!")
      print("<System> 반납 알림 요청 O")
      print(database)
    else:
      print("챗봇 : 알겠습니다. 다음에 이용해 주세요!")
      print("<System> 반납 알림 요청 X")
      print(database)
  
  elif want_alarm_wname != None:

    database_want_alarm_wname_borrowable_1 = database.loc[(database['wname']==want_alarm_wname) & (database['borrowable']==1)]

    # 원하는 작가님의 도서가 여러개 일 때
    if len(database_want_alarm_wname_borrowable_1) > 1:
      print("챗봇 : " , want_alarm_wname," 작가님의 도서 목록 중 반납 알림을 원하시는 도서가 있나요?.\n (도서 목록 중 반납 알림을 원하시면 네 혹은 아니오를 눌러주세요.")
      user_input = input()
      if user_input == "네":
        print("챗봇 : 반납 알림을 원하시는 도서명을 정확히 입력해주세요")
        user_input2 = input()
        want_writer_book_list = database_want_alarm_wname_borrowable_1['bname'].tolist()
        if user_input2 in want_writer_book_list:
          want_alarm_label = 0
          database[database['bname']==user_input2,'alarm'] = 1
          print("<System> 반납 알림 요청 O")
          print(database)
      else: 
        print("챗봇 : 알겠습니다. 다음에 이용해 주세요!")
        print("<System> 반납 알림 요청 X")
        print(database)
    elif len(database_want_alarm_wname_borrowable_1) == 1:
      want_alarm_bname = database.loc[database['wname']==want_alarm_wname,'bname']
      print("챗봇 : ", want_alarm_wname,"작가님의 도서 ", want_alarm_bname.iloc[0], " 이(가) 반납되면 알림 드릴까요?\n (반납 알림을 원하시면 네 혹은 아니오를 눌러주세요.)")
      user_input = input()
      if user_input == "네":
        want_alarm_label = 0
        database.loc[database['wname']==want_alarm_wname,'alarm'] = 1
        print("<System> 반납 알림 요청 O")
        print(database)
      else:
        print("챗봇 : 알겠습니다. 다음에 이용해 주세요!")
        print("<System> 반납 알림 요청 X")
        print(database)        

  return want_alarm_label

# 4. bfs 기반 챗봇 함수 정의

## BFS 메서드 정의
def bfs (graph, node, visited):

    # 큐 구현을 위한 deque 라이브러리 활용
    queue = deque([node])
    
    # 큐가 완전히 빌 때까지 반복
    while queue:

        # 큐에 삽입된 순서대로 노드 하나 꺼내기
        poped_node = queue.popleft()

        # 현재 노드를 방문 처리
        visited[poped_node.get_key()] = True

        print("<System> 현재 visited: ", visited)

        # 탐색 순서 출력
        # print(poped_node.get_key(), end = ' ')

        next_node = poped_node.task(poped_node)
      
        # 현재 처리 중인 노드에서 방문하지 않은 인접 노드를 모두 큐에 삽입
        for idx, node in enumerate(graph[poped_node.get_key()]):

            if idx != next_node.get_rmv_idx():
              visited[node.get_key()] = True

            print("<System> 현재 visited: ", visited)
            
            if not (visited[node.get_key()]):

                queue.append(node)
                # print("<System> 현재 visited: ", visited)

# 5. node 생성 , 그래프 설정, vistied (방문 여부) 정의

def set_node_list1():

  # node 생성1
  node1 = Node("<System> 의도 분류 모델 작동")
  node1.set_key(1)
  node1.set_model(intent_classify_model)

  node2 = Node("<System> (조회) 도서명,작가명 Tokenizer 작동, 도서명, 작가명, 장르 정보가 들어올때까지 재요청")
  node2.set_key(2)

  node3 = Node("<System> (추천) CNN 모델 작동")
  node3.set_key(3)

  node4 = Node("<System> (문의사항)문장 유사도 모델 작동")
  node4.set_key(4)
  node4.set_model(sbert_model)
  node4.set_emd_data(inquiry_data,inquiry_embedding_data)

  return node1, node2, node3, node4

def set_node_list2():

  # node 생성2
  node5 = Node("<System> DB 접근 후 도서 유무 확인")
  node5.set_key(5)

  node6 = Node("<System> 도서 보유 , 대출 예약 여부 요청")
  node6.set_key(6)

  node7 = Node("<System> 도서 미보유 , node1으로")
  node7.set_key(7)

  node8 = Node("<System> 대출 가능, 대츨 예약 여부 확인")
  node8.set_key(8)

  node9 = Node("<System> 대출 불가능, 반납 알림 여부 확인")
  node9.set_key(9)

  return node5, node6, node7, node8, node9

def set_node_list3():

  # node 생성3

  node10 = Node("<System> 대출 예약 요청")
  node10.set_key(10)

  node11 = Node("<System> 대출 예약 미요청")
  node11.set_key(11)

  node12 = Node("<System> 반납 알림 요청")
  node12.set_key(12)

  node13 = Node("<System> 반낭 알림 미요청")
  node13.set_key(13)

  return node10,node11,node12,node13

def set_graph(node1,node2,node3,node4,
              node5,node6,node7,node8,node9,
              node10,node11,node12,node13):

  # 그래프 설정
  graph = [
    [],
    [node2, node3, node4],
    [node5],
    [],
    [],
    [node6,node7],
    [node8,node9],
    [],
    [node10,node11],
    [node12,node13],
    [],
    [],
    [],
    []
  ]

  # 노드별로 방문 정보를 리스트로 표현
  visited = [False] * 14

  return graph, visited

# 6. 임시 database 설정

## 임시 database 정의

database = pd.read_csv(local_path+'/data/csv/intent_classify_v3_database(library).csv',encoding='cp949')

# 7. 챗봇 실행

## 정의한 BFS 메서드 호출(노드 1을 탐색 시작 노드로 설정)

while True :

  print("<System> 챗봇 초기화!")

  node1, node2, node3, node4 = set_node_list1()
  node5, node6, node7, node8, node9 = set_node_list2()
  node10,node11,node12,node13 = set_node_list3()

  graph, visited = set_graph(node1, node2, node3, node4, 
                             node5, node6, node7, node8, node9,
                             node10,node11,node12,node13)
  
  print("<System> 챗봇 작동 시작!")

  bfs(graph, node1, visited)