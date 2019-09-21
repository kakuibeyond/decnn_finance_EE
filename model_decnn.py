import tensorflow as tf

from tensorflow.python.ops import array_ops
class Decnn(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, emb_matrix1,emb_matrix2,emb_matrix_glove_1,emb_matrix_glove_2, topic_emb,l2_reg_lambda=0.0, ):

        # Placeholders for input, output and dropout
        self.entity_index = tf.placeholder(tf.int32, [None,sequence_length], name="entity")
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_x_glove = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_glove")
        self.input_y = tf.placeholder(tf.float32, [None, sequence_length,num_classes], name="input_y")
        self.input_y_two=tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.padding_mask = tf.placeholder(tf.float32, [None, sequence_length], name="padding")
        self.input_topic = tf.placeholder(tf.float32, [None, 10], name="topic")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.5)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #word
            self.W2 = tf.Variable(initial_value=emb_matrix2, name='embedding_matrix2', dtype=tf.float32, trainable=False)
            self.W1= tf.Variable(initial_value=emb_matrix1, name='embedding_matrix1',dtype=tf.float32, trainable=False)
            self.W=tf.concat([self.W2,self.W1],axis=0)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)#31*300

            self.W2_glove = tf.Variable(initial_value=emb_matrix_glove_2, name='embedding_matrix2_glove', dtype=tf.float32,
                                  trainable=False)
            self.W1_glove = tf.Variable(initial_value=emb_matrix_glove_1, name='embedding_matrix1_glove', dtype=tf.float32, trainable=False)
            self.W_glove = tf.concat([self.W2_glove, self.W1_glove], axis=0)
            self.embedded_chars_glove = tf.nn.embedding_lookup(self.W_glove, self.input_x_glove)  # 31*300


            self.entity1 = tf.Variable(tf.random_uniform(shape=[14, 50], minval=-0.1, maxval=0.1),
                                        name='entity_embedding', dtype=tf.float32, trainable=True)
            self.entity2 = tf.constant(shape=[1, 50],value=0.0,dtype=tf.float32)
            self.entity = tf.concat([self.entity1, self.entity2], axis=0)


            self.entity_embedding = tf.nn.embedding_lookup(self.entity, self.entity_index)
            self.word_position_embedding_qu1=tf.concat([self.embedded_chars,self.entity_embedding],axis=2) #b,80,350
            self.word_position_embedding_qu1_glove = tf.concat([self.embedded_chars_glove, self.entity_embedding],
                                                         axis=2)  # b,80,350
            print(self.word_position_embedding_qu1) #b,80,350

            self.topic_embedding = tf.Variable(initial_value=topic_emb, name='topic_embedding', dtype=tf.float32,
                                               trainable=False)
            self.topic_embedding_last = tf.matmul(self.input_topic, self.topic_embedding)  # b,300

            ##拼接两种embedding
            self.word_last_embedding=tf.concat([self.word_position_embedding_qu1,self.word_position_embedding_qu1_glove],axis=-1)
            print(self.word_last_embedding)

        with tf.name_scope("conv"):
            self.conv1 = tf.layers.conv1d(self.word_last_embedding, 128, 5, padding='same')
           # self.conv1=tf.layers.conv1d(self.word_last_embedding,128,5,padding='same')
            self.conv2=tf.layers.conv1d(self.word_last_embedding,128,3,padding='same')
            self.conv1_conv2=tf.nn.relu(tf.concat([self.conv1,self.conv2],axis=2))
            self.conv1_conv2_drop=tf.nn.dropout(self.conv1_conv2,self.dropout_keep_prob)

            print(self.conv1_conv2_drop)

            self.conv3=tf.layers.conv1d(self.conv1_conv2_drop,256,5,padding='same',activation=tf.nn.relu)
            self.conv3_drop = tf.nn.dropout(self.conv3, self.dropout_keep_prob)

            self.conv4 = tf.layers.conv1d(self.conv3_drop, 256, 5,padding='same', activation=tf.nn.relu)
            self.conv4_drop = tf.nn.dropout(self.conv4, self.dropout_keep_prob)

            self.conv5 = tf.layers.conv1d(self.conv4, 256, 5, padding='same',activation=tf.nn.relu)#b,80,256
            self.scores = tf.layers.dense(self.conv5,num_classes)

            print(self.scores)  ## b,80,34

        with tf.name_scope("bi-lstm"):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(150, forget_bias=1.0, state_is_tuple=True)  # 论文没写参数？
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(150, forget_bias=1.0, state_is_tuple=True)

            self.birnn_outputs = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                self.word_last_embedding,
                dtype=tf.float32,

            )
            self.birnn_outputs_pin = tf.concat(self.birnn_outputs[0], 2) #b,80,300



        batch_size = tf.shape(self.input_x)[0]
        with tf.name_scope("attention"):   #b,80,300
            self.W1=tf.get_variable("W1",shape=[1,300,300],initializer=tf.contrib.layers.xavier_initializer())
            self.W1 = tf.tile(self.W1, [batch_size, 1, 1])#b,300,300

            self.att = tf.matmul(self.birnn_outputs_pin, self.W1)#b,80,300
            self.att = tf.matmul(self.att, tf.expand_dims(self.topic_embedding_last,-1)) #b,80,1
            self.att = tf.nn.tanh(self.att)

            self.soft=softmask(self.att,tf.expand_dims(self.padding_mask,-1))

        self.out = tf.matmul(self.soft, self.birnn_outputs_pin, adjoint_a=True)
        self.out = tf.reduce_sum(self.out,axis=1)




        self.cnn_lstm=tf.concat([self.out,tf.reduce_sum(self.conv5,axis=1)],axis=-1)
        self.scores2 = tf.layers.dense(self.out,2)
        self.loss_sentence = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores2, labels=self.input_y_two)  # b,2
        self.loss_sentence= tf.multiply(self.loss_sentence,self.padding_mask)
        self.loss_sentence=tf.reduce_mean(self.loss_sentence)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.pred_probas = tf.nn.softmax(self.scores, name='class_proba')#b,80,34
            self.predictions = tf.argmax(self.pred_probas, axis=-1, name='class_prediction')  #b,80
            print(self.predictions)

        # Calculate ranking loss
       # with tf.name_scope("loss"):
           # self.labels = tf.argmax(self.input_y, -1)
           # self.labels = tf.cast(self.labels, dtype=tf.int32)
           # self.losses=ranking_loss(self.labels,self.scores,self.batch_size,self.padding_mask)
         #   self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)# b,8
          #  self.losses=tf.multiply(self.losses,self.padding_mask)
          #  self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss
           # self.loss_sentence = tf.reduce_mean(self.losses)

       # # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)# b,8
            self.losses=tf.multiply(self.losses,self.padding_mask)
            self.loss_sentence=tf.constant(0.0)
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss + 0.1*self.loss_sentence

        tvars = tf.trainable_variables()
        print(tvars)
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, -1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

def ranking_loss( labels, logits, batch_size,padding_mask):
    lm = tf.constant(1.0)  # lambda
    m_plus = tf.constant(2.5)
    m_minus = tf.constant(0.5)

    L = tf.constant(0.0)
    i = tf.constant(0)
    cond = lambda i, L: tf.less(i, batch_size)

    def loop_body(i, L):
        padding=padding_mask[i]
        print(padding)
        padding_80=tf.unstack(padding,axis=0)
        labels_80=tf.unstack(labels,axis=1)
        logits_80=tf.unstack(logits,axis=1)
        ll=tf.constant(0.0)
        one=tf.constant(1.0)
        for j in range(80):
           # if padding_80[j] is not None:
                if tf.equal(padding_80[j],one) is not None:
                    cplus = labels_80[j][i]  # positive class label index
            # taking most informative negative class, use 2nd argmax
                    _, cminus_indices = tf.nn.top_k(logits_80[j][i, :], k=2)
                    cminus = tf.cond(tf.equal(cplus, cminus_indices[0]),  # 如果第二大的是正确索引则为第三大的
                             lambda: cminus_indices[1], lambda: cminus_indices[0])

                    splus = logits_80[j][i, cplus]  # score for gold class
                    sminus = logits_80[j][i, cminus]  # score for negative class


            ###splus++  sminus--
                    ll = ll+ tf.log((1.0 + tf.exp((lm * (m_plus - splus))))) + \
                tf.log((1.0 + tf.exp((lm * (m_minus + sminus)))))
           # print(ll)

        return [tf.add(i, 1), tf.add(L, ll)]

    _, L = tf.while_loop(cond, loop_body, loop_vars=[i, L])
    nbatch = tf.to_float(batch_size)
    L = L
    return L

def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.75, gamma=2):
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)

    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)


    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)

    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
 \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))

    return tf.reduce_sum(per_entry_cross_ent)


def softmask(x, mask):
    y = tf.multiply(tf.exp(x), tf.cast(mask, tf.float32))  #b,31,1
    sumx = tf.reduce_sum(y, axis=1, keep_dims=True)
    return y / (sumx + 1e-10)