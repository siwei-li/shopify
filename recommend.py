import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow import keras
# from keras.preprocessing.sequence import pad_sequences

from app_params import QUERY_MAX_LEN, TOP_K


def text_query(model, image_embeddings, df, tokenizer, text_query):
    seq = keras.preprocessing.sequence.pad_sequences(
    tokenizer.texts_to_sequences(
        [text_query]
    ), maxlen=QUERY_MAX_LEN, padding='pre', truncating='post'
)
    query_vector = model.predict(seq)
    print(text_query, seq, query_vector[0][:5])

    res = cosine_similarity(image_embeddings, query_vector)
    top_ix = res[:, 0].argsort()[::-1][:TOP_K]

    return [df.iloc[ix]['image'] for ix in top_ix]

def image_query(model, image_embeddings, df, filename):
    image_query = model.predict(np.stack(df.loc[df['image'] == filename].image_array.values))
    
    res_im = cosine_similarity(image_embeddings, image_query)
    top_im_ix = res_im[:, 0].argsort()[::-1][:TOP_K]
    return [df.iloc[ix]['image'] for ix in top_im_ix]
