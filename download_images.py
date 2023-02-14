import os
import sys
import pandas as pd



df = pd.read_csv('./data/training_data.csv')
get_url = lambda id: 'https://http2.mlstatic.com/D_' + id + '-F.jpg'

threshold = 0.03

min_th = 10000
for index, row in df.iterrows():
    label = str(row['correct_background?'])
    if(label != "0" and label!="1"):
        continue
    picture_id = row['picture_id']
    
    try:
      picture_url = get_url(picture_id)
      img_file = "./data/train/" + label + '/' + picture_id + '.jpg' 

      os.system("wget -O "+ img_file + " " + picture_url)
      os.system("convert "+img_file+" -resize 224x224 "+img_file)

    except:
      print('error')





