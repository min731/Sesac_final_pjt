import re


def analyze(text):
    global numbers

    numbers = re.findall(r'\d+월|\d+일|오전|오후|\d+시|\d+분|\d+명|\d+', text)
    
    text = text.replace('오전 ', '')
    text = text.replace('오전', '')
    text = text.replace('오후 ', '')
    text = text.replace('오후', '')

    # phone_prefixes = [r'02', r'010', r'031']

    for n in numbers:
        if re.match(r'^010', n):
            text = phone_number(numbers.index(n), text)
        if '월' in n:
            text = month(numbers.index(n), text)
        if '일' in n:
            text = day(numbers.index(n), text)
        if '시' in n:
            text = hour(numbers.index(n), text)
        if '분' in n:
            text = minute(numbers.index(n), text)
    
    return text


def phone_number(idx, text):
    phone_dict = ['공', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구']

    separated = text.split(numbers[idx])

    result = ''

    for n in numbers[idx]:
        result += phone_dict[int(n)]

    return separated[0] + result + separated[1]


def month(idx, text):
    month_dict = ['영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구', '십', '십일', '십이']

    separated = text.split(numbers[idx])

    temp = int(numbers[idx].split('월')[0])

    result = month_dict[temp] + '월'

    return separated[0] + result + separated[1]


def day(idx, text):
    day_dict = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구', '십']
    
    separated = text.split(numbers[idx])

    temp = numbers[idx].split('일')[0]

    temp_list = []

    if int(temp) >= 10:
        temp_list = [int(i) for i in temp]
    
    if len(temp_list) > 1:
        if temp_list[1] == 0:
            temp = day_dict[temp_list[0]] + day_dict[10]
        else:
            temp = day_dict[temp_list[0]] + day_dict[10] + day_dict[temp_list[1]]
    else:
        temp = day_dict[int(temp)]
    
    return separated[0] + temp + '일' + separated[1]


def hour(idx, text):
    hour_dict = ['영', '한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열', '열한', '열두']
    meridiem = ['오전 ', '오후 ']

    meridiem_flag = 0

    separated = text.split(numbers[idx])

    temp = int(numbers[idx].split('시')[0])

    if temp == 12:
        meridiem_flag = 1

    if temp > 12:
        temp -= 12
        
        meridiem_flag = 1

        if temp == 12:
            meridiem_flag = 0
            temp = 0
    
    result = meridiem[meridiem_flag] + hour_dict[temp] + '시'

    return separated[0] + result + separated[1]


def minute(idx, text):
    day_dict = ['', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구', '십']
    
    separated = text.split(numbers[idx])

    temp = numbers[idx].split('분')[0]

    temp_list = []

    if int(temp) >= 10:
        temp_list = [int(i) for i in temp]
    
    if len(temp_list) > 1:
        if temp_list[1] == 0:
            temp = day_dict[temp_list[0]] + day_dict[10]
        else:
            temp = day_dict[temp_list[0]] + day_dict[10] + day_dict[temp_list[1]]
    else:
        temp = day_dict[int(temp)]
    
    return separated[0] + temp + '분' + separated[1]


text = "제 23일 전화번호는 01036688460이에요 오후 15시 33분"

print('원본 텍스트:', text)
print('변환된 텍스트:', analyze(text))