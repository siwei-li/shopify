cp kaggle.json ~/.kaggle/
chmod 600 /root/.kaggle/kaggle.json
kaggle datasets download -d adityajn105/flickr8k

unzip flickr8k.zip -d static
mv static/Images static/images
rm static/captions.txt