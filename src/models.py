# 모델을 정의한다.
import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Concatenate, Dropout, add
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import min_max_norm
import numpy as np


input_shape = (30, 100)  # fasttext, mfcc input shape

norm2 = min_max_norm(-2, 2)
norm3 = min_max_norm(-3, 3)


def np_acc(y_true, y_pred):
    return np.mean(np.round(y_true) == np.round(y_pred))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units, attention_only=False):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        self.attention_only = attention_only

    def call(self, values, query):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)
        if self.attention_only:
            return attention_weights, attention_weights

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class ClassificationModel:
    def norm_LSTM(self, units, return_state=False):
        return LSTM(units, dropout=0.4, return_sequences=True, return_state=return_state, recurrent_constraint=norm2, bias_constraint=norm3, kernel_constraint=norm2)

    def attention_block(self, attention_only=False):
        # 양방향 LSTM 어텐션 메커니즘
        inp = Input(shape=input_shape)
        inter = Bidirectional(self.norm_LSTM(16))(inp)
        lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(self.norm_LSTM(4, return_state=True))(inter)

        attention = BahdanauAttention(2, attention_only=attention_only)  # 가중치 크기 정의
        state_h = Concatenate()([forward_h, backward_h])  # 은닉 상태
        context_vector, attention_weights = attention(lstm, state_h)
        return inp, context_vector

    def build_model(self):
        # 모델을 반환한다.
        inp1, context_vector1 = self.attention_block()  # mfcc
        inp2, context_vector2 = self.attention_block()  # fasttext

        inter = add([context_vector1, context_vector2])

        inter = Dropout(0.4)(inter)
        output = Dense(2, activation="softmax", bias_constraint=norm3, kernel_constraint=norm2)(inter)
        model = Model(inputs=[inp1, inp2], outputs=output)

        optimizer = Adam(lr=0.0001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model


if __name__ == "__main__":
    model = ClassificationModel().build_model()
    model.summary()
