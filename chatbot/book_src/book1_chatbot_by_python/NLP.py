# 챗봇 개념

# p24 
# 사용자에게 선택지를 주어 예외처리

# p25
# 챗봇 프레임워크

# p27
# intent : 사용자의 의도
# Entity : intent의 핵심 요소
# Utterances : 질문에 대한 여러가지 표현

# p30
# spaCy : NLP 라이브러리

# spaCy 라이브러리 활용

# 텍스트 처리 프로세스 라이브러리

import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(u'i am learning how to build chatbots')
for token in doc:
    print(token.text, token.pos_)

