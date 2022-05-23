from flask import Flask, request
from flask import render_template
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

from recommend import text_query, image_query
from app_params import *

app = Flask(__name__, static_url_path='/static')


images = sorted(os.listdir('static/images'))
print("Images loaded!")
df = pd.read_pickle("df.pkl")

# ImageModel = keras.models.load_model("image_model")
# ImageEmbeddings = ImageModel.predict(np.stack(df['image_array'].values))
# TextModel = keras.models.load_model("text_model")
tokenizer = Tokenizer(num_words=10000, filters='', lower=True, split=' ', char_level=False, oov_token=None)
# print("Model loaded!")



@app.route("/home")
def home():
    page = request.args.get('page', default = 1, type = int)
    display_images = [img for img in images[(page-1) * IMAGE_PER_PAGE:page * IMAGE_PER_PAGE]]
    print(display_images[0])
    return render_template("home.html", images = display_images)

@app.route("/similar")
def similar():
    selectedImage = request.args.get("selectedImage")
    similarImages = image_query(ImageModel, ImageEmbeddings, df, selectedImage)
    return render_template("recommend.html", 
    # title="Recommendations", 
    # customstyle="recommend.css", 
    # inputImage = selectedImage,
    images = similarImages,
    # similarityValues = values
    )

@app.route("/search")
def search():
    textQuery = request.args.get("textQuery")
    images = text_query(TextModel, ImageEmbeddings, df, tokenizer, textQuery)
    return render_template("recommend.html", images = images)

if __name__ == "__main__":
  app.run(debug=True)