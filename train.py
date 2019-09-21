import  numpy as np
import predata as pre
import model_decnn as model
from tensorflow.contrib import learn
import tensorflow as tf
import datetime

from  sklearn.metrics import  accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from  sklearn.metrics import recall_score
from  sklearn.metrics import precision_score
from  sklearn.metrics import f1_score
from gensim.models.keyedvectors import KeyedVectors
import random
from sklearn.metrics import classification_report
lr = 0.001
window_size=80
batch_size=124
class_number=34
l2_lambad=3
drop_keep_out=0.5
epochs=2000
embedding_size=300
np.random.seed(4567)
pretrain_emb_path="../GoogleNews-vectors-negative300.bin"
pretrain_emb_path2='glove.840B.300d_new.txt'
print("learning rate +\n",lr)
print("batch_size +\n",batch_size)
print("drop+\n",drop_keep_out)

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def gettopic():
    model_topic= KeyedVectors.load_word2vec_format("topic_init3.txt")
    emb=[]
    for i in range(10):
        emb.append(model_topic[str(i)].tolist())
    return emb

def getava(true_label,prediction_label,padding):
    real_trigger_number = 0
    real_trigger_label=[]
    pre_trigger_label=[]
    true_label_pin=[]
    prediction_label_pin=[]
    for t in true_label:
        for tt in list(t):
            true_label_pin.append(tt)
    for t in prediction_label:
        for tt in t:
            prediction_label_pin.append(tt)


    #去掉对padding的预测
    true_label_pin_new=[]
    prediction_label_pin_new=[]
    for i in range(len(padding)):
        if padding[i]==1:
            true_label_pin_new.append(true_label_pin[i])
            prediction_label_pin_new.append(prediction_label_pin[i])
    #print(len(true_label_pin_new))



    for tt in list(true_label_pin_new):
        if tt != 0:
            real_trigger_number = real_trigger_number + 1
            real_trigger_label.append(tt)
    pre_trigger_number = 0
    pre_trigger_number_index=[]
    wrong_trigger_number_index=[]

    for i,tt in enumerate(prediction_label_pin_new):
        if tt != 0:
            pre_trigger_number = pre_trigger_number + 1
            pre_trigger_label.append(tt)
    true_number = 0
    for i in range(len(true_label_pin_new)):
        if true_label_pin_new[i] != 0 and true_label_pin_new[i] == prediction_label_pin_new[i]:
            true_number = true_number + 1
            pre_trigger_number_index.append(i)
        elif true_label_pin_new[i] != 0 and true_label_pin_new[i] != prediction_label_pin_new[i]:
            wrong_trigger_number_index.append(i)
    if pre_trigger_number != 0:
        precision = true_number / pre_trigger_number
    else:
        precision = 0
    if real_trigger_number != 0:
        recall = true_number / real_trigger_number
    else:
        recall = 0
    if precision * recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
   # print(real_trigger_number,pre_trigger_number,true_number)
   # print(classification_report(true_label, prediction_label))
    return precision, recall, f1,pre_trigger_number_index,wrong_trigger_number_index,real_trigger_label,pre_trigger_label


def getlabel34(labels):
    y_batch_34_test = []
    for w in range(len(labels)):
        new = []
        for yy in labels[w]:
            label = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            label[yy] = 1
            new.append(label)
        y_batch_34_test.append(new)

    return  y_batch_34_test

def getindex(sentences,word_dict):
    x = []  # [0,1,2,...]
    for tt in sentences:
        se = [word_dict[ttt] for ttt in tt]
        x.append(se)
    return x

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
   # print(data[0])
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1 # 1124/64
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def loadEmbMatrix(pretrain_emb_path,text,embed_size,bina):
    print('Indexing word vectors.')
   # print(text)
    model = KeyedVectors.load_word2vec_format(pretrain_emb_path, binary=bina)
    #print(model['None'])
    embedding_matrix = np.zeros((len(text), embed_size), dtype='float32')
    word_dict = {}
    for i, tt in enumerate(text):
        if tt=='None':
            word_dict[tt]=i
        else:
            if i % 3000 == 0:
                print(i)
            try:
                word_dict[tt] = i
                embedding_matrix[i] = model[tt]
            except KeyError:
                continue

    return embedding_matrix, word_dict


def main():
    train_sentences, test_sentences, dev_sentences, train_trigger_labels, test_trigger_labels, dev_trigger_labels, \
	train_entity_labls, test_entity_labls, dev_entity_labls,\
    train_paddings,test_paddings,dev_paddings,\
    train_docs,test_docs,dev_docs,\
    =pre.predata()
    print(len(train_docs),len(test_docs),len(dev_docs))

    topic_emb = gettopic()

    dev_paddings_all=[]
    for dd in dev_paddings:
        for ddd in dd:
            dev_paddings_all.append(ddd)

    print(len(dev_paddings),len(dev_paddings_all))

    test_paddings_all=[]
    for dd in test_paddings:
        for ddd in dd:
            test_paddings_all.append(ddd)

    train_two=[]
    for tl in train_trigger_labels:
        if max(tl)!=0:
            train_two.append([0,1])
        else:
            train_two.append([1,0])

   # print(train_two)
    dev_two=[]
    for tl in dev_trigger_labels:
        if max(tl) != 0:
            dev_two.append([0, 1])
        else:
            dev_two.append([1, 0])

    test_two=[]
    for tl in test_trigger_labels:
        if max(tl) != 0:
            test_two.append([0, 1])
        else:
            test_two.append([1, 0])


    print(len(test_sentences),len(test_paddings))
    words =train_sentences+test_sentences+dev_sentences
    print(len(words))
    words_new = ['None']  #没有重复的  出现过得 词
    for ww in words:
       # print(ww)
        for www in ww:
            if www not in words_new:
                words_new.append(www)

    emb_matrix, word_dict = loadEmbMatrix(pretrain_emb_path, words_new, embedding_size,bina=False)
    print(len(word_dict))
    emb_matrix2=[]
    emb_matrix2.append(emb_matrix[0].tolist())
    print(emb_matrix2)


    emb_matrix_glove, word_dict_glove = loadEmbMatrix(pretrain_emb_path, words_new, embedding_size,bina=True)
    print(len(word_dict_glove))

    emb_matrix_glove_2=[]
    emb_matrix_glove_2.append(emb_matrix_glove[0].tolist())
    print(emb_matrix_glove_2)

    x_train=getindex(train_sentences,word_dict)
    x_test = getindex(test_sentences, word_dict)
    x_dev=getindex(dev_sentences, word_dict)

    x_train2 = getindex(train_sentences, word_dict_glove)
    x_test2 = getindex(test_sentences, word_dict_glove)
    x_dev2 = getindex(dev_sentences, word_dict_glove)

    #print(emb_matrix)
    gcn = model.Decnn(  #实际是cnn  懒得改了。
        sequence_length=window_size,
        num_classes=class_number,
        vocab_size=len(words_new),
        embedding_size=embedding_size,
        emb_matrix1=emb_matrix[1:],
        emb_matrix2=emb_matrix2,
        emb_matrix_glove_1=emb_matrix_glove[1:],
        emb_matrix_glove_2=emb_matrix_glove_2,
        l2_reg_lambda=l2_lambad,
        topic_emb=topic_emb,
    )


    train_op = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-06).minimize(gcn.loss)
    print(lr)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    def train_step(x_batch,entity_indexs,y_batch,batch_size,padding,x_batch_glove,y_batch_two,topic):
        """
        A single training step
        """
        feed_dict = {
            gcn.input_x: x_batch,
            gcn.input_y: y_batch,
            gcn.entity_index: entity_indexs,
            gcn.dropout_keep_prob: drop_keep_out,
            gcn.batch_size:batch_size,
            gcn.padding_mask:padding,
            gcn.input_x_glove:x_batch_glove,
            gcn.input_y_two:y_batch_two,
            gcn.input_topic: topic,
        }
        _, loss, accuracy, predictions,scores,W_train,loss_sentence = sess.run(
            [train_op, gcn.loss, gcn.accuracy, gcn.predictions,gcn.pred_probas,gcn.W,gcn.loss_sentence], feed_dict)

        return loss, accuracy, predictions,scores,W_train,loss_sentence

    def dev_step(x_batch, entity_indexs, y_batch,batch_size,padding,x_batch_glove,y_batch_two,topic):

        feed_dict = {
            gcn.input_x: x_batch,
            gcn.input_y: y_batch,
            gcn.entity_index:entity_indexs,
            gcn.dropout_keep_prob:1,
            gcn.batch_size:batch_size,
            gcn.padding_mask: padding,
            gcn.input_x_glove: x_batch_glove,
            gcn.input_y_two: y_batch_two,
            gcn.input_topic: topic,
        }
        loss, accuracy, predictions,scores,pob,loss_sentence = sess.run([gcn.loss, gcn.accuracy, gcn.predictions,gcn.loss,gcn.pred_probas,gcn.loss_sentence], feed_dict)

        return loss, accuracy, predictions,scores,pob,loss_sentence


    # Generate batches
    batches = batch_iter(
        list(zip(x_train, train_entity_labls,train_trigger_labels,train_paddings,x_train2,train_two,train_docs)), batch_size, epochs)
    best_f1=0
    best_loss=10
    # Training loop. For each batch...
    for k, batch in enumerate(batches):
       # print(k)
        x_batch,x_entity_batch,y_batch,paddings_batch,x_batch_glove,y_batch_two,x_docs_batch = zip(*batch)
       # print(len(paddings_batch))
        y_batch_34=getlabel34(y_batch)
        y_batch_34_dev=getlabel34(dev_trigger_labels)
        y_batch_34_test=getlabel34(test_trigger_labels)

        loss_train, acc_train, prediction_bath,scores_train,W_train,loss_sentence_train= train_step(x_batch,x_entity_batch,
                                                                                y_batch_34,len(x_batch),paddings_batch,x_batch_glove,y_batch_two,x_docs_batch)
        if k % (int(len(x_train) / batch_size /10)) == 0:

            if k%500==0:
                print(k)

            len_dev=[]
            for a,x in enumerate(dev_entity_labls):
                if len(x)!=80:
                    print(a,len(x),x)
            loss_dev, acc_dev, prediction_dev,scores_dev,pob,loss_sentence_dev = dev_step(x_dev, dev_entity_labls, y_batch_34_dev,len(x_dev),dev_paddings,x_dev2,dev_two,dev_docs)

            prec_dev, recall_dev, f1_dev, pre_trigger_dev, wrong_trigger_dev, real_trigger_label_dev, pre_trigger_label_dev = getava(
                dev_trigger_labels, prediction_dev,dev_paddings_all)
            if  f1_dev>best_f1 or abs(f1_dev-best_f1)<0.005:
                 best_loss=loss_dev
                 best_f1=f1_dev
                 print("train loss {:g}, train sentence loss {:g}".format(loss_train,loss_sentence_train))
                 print("dev loss {:g}, dev sentence loss {:g} acc {:g} precison {:g} recall {:g} f1 {:g}".format(loss_dev,loss_sentence_dev, acc_dev,
                                                                                     prec_dev,
                                                                                     recall_dev, f1_dev))
                 if best_f1>=0.0:

                     loss_test, acc_test, prediction_test, scores_test, pob_test,loss_sentence_test = dev_step(x_test, test_entity_labls, y_batch_34_test,
                                                                          len(x_test), test_paddings,x_test2,test_two,test_docs)

                     prec_test, recall_test, f1_test, pre_trigger_test, wrong_trigger_test, real_trigger_label_test, pre_trigger_label_test = getava(
                     test_trigger_labels, prediction_test,test_paddings_all)
                     print("test loss {:g}, test sentence loss {:g}acc {:g} precison {:g} recall {:g} f1 {:g}".format(loss_test,loss_sentence_test, acc_test,
                                                                                     prec_test,
                                                                                     recall_test, f1_test))



if __name__ == '__main__':
    main()

