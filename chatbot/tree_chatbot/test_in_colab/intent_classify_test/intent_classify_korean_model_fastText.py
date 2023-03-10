import io
import time


from ctypes import cast, py_object

def di(addr):
    return cast(addr, py_object).value

def load_vectors(fname):

    strat_time = time.time()

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    cnt = 0 

    for line in fin:
        cnt += 1
        print(str(cnt/2000000*100)+" (%)")
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        # print(di(data[tokens[0]]))
        

    end_time = time.time()

    sec = round(end_time - strat_time,1)
    min_ =  int(sec//60)
    sec = int(sec % 60)
    print(f'반복문의 실행은 {min_}분 {sec}초가 걸렸습니다.')

    return data

fname = "C:\\backend_study\\final_pjt\\chatbot\\tree_chatbot\\test_in_colab\\intent_classify_test\\embedding_model\\fastText\\embedding_data\\cc.ko.300.vec"
data = load_vectors(fname)