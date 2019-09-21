import numpy as np
from gensim.models.keyedvectors import KeyedVectors
max_len=80
def getseandlabels(file_path):
    model_doc = KeyedVectors.load_word2vec_format("docembedding3.txt")
    se_trigger_entity=open(file_path,encoding='utf-8').readlines()
    doc_name = []
    ses=[]
    trigger=[]
    entity=[]
    for i in range(len(se_trigger_entity)-3):
        if (i)%4==0:
            doc_name.append(model_doc[se_trigger_entity[i].strip()])
            ses.append(se_trigger_entity[i+1].strip().split(' '))
            ti=se_trigger_entity[i+2].strip().split(' ')
            ti_int=[]
            for tt in ti:
               # if int(tt)!=0:
                   # ti_int.append(1)
               # else:
                    ti_int.append(int(tt))
            trigger.append(ti_int)
            ei=se_trigger_entity[i+3].strip().split(' ')
            ei_int=[]
            for tt in ei:
                ei_int.append(int(tt))

            entity.append(ei_int)
    ses_new=[]
    trigger_new=[]
    entity_new=[]
    paddings=[]
    for i in range(len(ses)):
        if len(ses[i])>=80:
            ses_new.append(ses[i][0:80])
            trigger_new.append(trigger[i][0:80])
            entity_new.append(entity[i][0:80])
            padd=[]
            for k in range(80):
                padd.append(1)
            paddings.append(padd)
        else:
            padd=[]
            ss=ses[i]
            tt=trigger[i]
            ee=entity[i]
            for j in range(80-len(ss)):
                ss.append('None')
                tt.append(0)
                ee.append(14)
            for k,ttt in enumerate(ss):
                if ttt=='None':
                    padd.append(0)
                else:
                    padd.append(1)
            ses_new.append(ss)
            trigger_new.append(tt)
            entity_new.append(ee)
            paddings.append(padd)


    return  ses_new,trigger_new,entity_new,paddings,doc_name


def predata():

   #get sequences/trigger label/typelabel  each sentence
   #What,does,that,have,to,do,with,the,war,in,Iraq,?
   #0,0,0,0,0,0,0,0,13,0,0,0
   # 0,0,0,0,0,0,0,0,0,0,2,0
   train_sentences,train_labels,train_entity,train_paddings,train_docs=getseandlabels("train_new.txt")
   test_sentences, test_labels, test_entity,test_paddings,test_docs = getseandlabels("test_new.txt")
   dev_sentences,dev_labels, dev_entity,dev_paddings,dev_docs = getseandlabels("dev_new.txt")

   print(train_sentences[0])
   print(train_labels[0])
   print(train_entity[0])
   print(train_paddings[0])

   return  train_sentences,test_sentences,dev_sentences,\
           train_labels,test_labels,dev_labels,\
           train_entity,test_entity,dev_entity,\
           train_paddings,test_paddings,dev_paddings,\
           train_docs,test_docs,dev_docs




predata()
