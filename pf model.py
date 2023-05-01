import gradio as gr
from keras.layers import TextVectorization
import tensorflow as tf
import pandas as pd

df = pd.read_csv('datasets/train.csv')

X = df['comment_text']
y = df[df.columns[2:]].values

MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)

model = tf.keras.models.load_model('pfilter.h5')


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

    return text


interface = gr.Interface(fn=score_comment,
                         inputs=gr.components.Textbox
                         (lines=2, placeholder='Comment to Score'), outputs='text')
interface.launch(inline=False, share=True)
