import pyspark
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
import mysql.connector
from mysql.connector import Error
from pyspark.sql.types import *
from pyspark import SparkContext
import numpy as np
import streamlit as st

import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import pickle


with open("BOOKS_KITAPYURDU.pickle",'rb') as read_file:
    df = pickle.load(read_file)

with open("train.pickle",'rb') as read:
    test = pickle.load(read)

def insertVariblesIntoTable(userID,bookID,rating):
    try:
        connection = mysql.connector.connect(host='34.77.128.203',
                                             database='books',
                                             user='root',
                                             password='root')
        cursor = connection.cursor()
        mySql_insert_query = """INSERT INTO ratings(userID,bookID,rating)VALUES(%s, %s, %s) """
        recordTuple = (userID,bookID,rating)


        cursor.execute(mySql_insert_query,recordTuple)
        connection.commit()
        print("Record inserted successfully into Laptop table")

    except mysql.connector.Error as error:
        print("Failed to insert into MySQL table {}".format(error))

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


C = df['PRODUCT_STAR'].mean()
m = df['PRODUCT_RATING'].quantile(0.30)
q_movies = df.copy().loc[df['PRODUCT_RATING'] >= m]


def weighted_rating(x, m=m, C=C):
    v = x['PRODUCT_RATING']
    R = x['PRODUCT_STAR']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)



df['score'] = df.apply(weighted_rating, axis=1)
grouped_best_seller = df[["PRODUCT_NAME","score","IMAGE"]].sort_values(by="score",ascending=False).head(3).reset_index()
grouped_children = df.loc[df.KATEGORI_1=="ÇocukKitaplarıRoman"].sort_values(by="score",ascending=False).head(3).reset_index()
bilim = df.loc[df.KATEGORI_1=="Bilim&MühendislikPopülerBilim"].sort_values(by="score",ascending=False).head(3).reset_index()



image_bilim = []
for i in range(len(bilim)):

    response = requests.get(bilim["IMAGE"][i])
    image_bilim.append(Image.open(BytesIO(response.content)))

image_best = []
for i in range(len(grouped_best_seller)):

    response = requests.get(grouped_best_seller["IMAGE"][i])
    image_best.append(Image.open(BytesIO(response.content)))

image_children = []
for i in range(len(grouped_children)):

    response = requests.get(grouped_children["IMAGE"][i])
    image_children.append(Image.open(BytesIO(response.content)))

#st.sidebar.image(["book.png","student.png"],width=110)

st.sidebar.header("Sevdiğiniz Kitapları Seçin")
st.sidebar.write("---")
DATA = st.sidebar.multiselect("",df["PRODUCT_NAME"],)



train_data = df.loc[df["PRODUCT_NAME"].isin(DATA)][["PRODUCT_ID","score",]]
train_data["id"]=20







spark = pyspark.sql.SparkSession.builder.getOrCreate()

mySchema = StructType([StructField("PRODUCT_ID", IntegerType(), True)\
                       ,StructField("score",FloatType(),True)\
                       ,StructField("id",IntegerType(),True)
                        ])
mySchema_test = StructType([StructField("PRODUCT_ID", IntegerType(), True)\
                       ,StructField("score",FloatType(),True)\
                       ,StructField("id",IntegerType(),True)
                        ])


als = ALS(userCol='id',
          itemCol='PRODUCT_ID',
          ratingCol='score',
          rank=4,
          seed=42)

als_paramgrid = (ParamGridBuilder()
                 .addGrid(als.rank, [2, 4])
                 .addGrid(als.maxIter, [10])
                 .addGrid(als.regParam, [0.1])
                 .addGrid(als.alpha, [2.0, 3.0])
                 .build())

# The evaluation function for determining the best model
rmse_eval = RegressionEvaluator(labelCol='rating',
                                predictionCol='prediction', 
                                metricName='rmse')

# The cross validation instance
als_cv = CrossValidator(estimator=als,
                        estimatorParamMaps=als_paramgrid,
                        evaluator=rmse_eval,
                        numFolds=3, 
                        seed=42)


# function to select a few rows of data and convert to a Pandas dataframe
def preview(df,n=7000):
    return pd.DataFrame(df.take(n),columns=df.columns)

def calculate():
    df_train = spark.createDataFrame(train_data, schema=mySchema)
    df_test = spark.createDataFrame(test, schema=mySchema_test)

    als_model = als.fit(df_train)
    als_pred = als_model.transform(df_test)

    rec_dataframe = preview(als_pred['PRODUCT_ID', 'score', 'id', 'prediction'])

    dataframe = rec_dataframe.loc[rec_dataframe.id == 10]["PRODUCT_ID"].reset_index()

    result = dataframe["PRODUCT_ID"].astype(int).sample(1)

    son = int(result)

    image = df.loc[df.PRODUCT_ID == son].reset_index()
    img_s = image["index"].astype(int)
    image_index = int(img_s)
    response = requests.get(df["IMAGE"][image_index])
    rec_img = Image.open(BytesIO(response.content))

    st.sidebar.image(rec_img)

    st.balloons()
    st.balloons()







st.sidebar.write("---")
BUTTON = st.sidebar.button("Kitap Öner")


if BUTTON:
    calculate()




RATING = st.sidebar.slider("Rating",1,5,3)
RATING_BUTTON = st.sidebar.button("Kitaba puan verin")

if RATING_BUTTON:
    insertVariblesIntoTable("1",son,RATING)


st.write("En yüksek puan alan kitaplar")

st.image(image_best,width=200)
st.write("---")


st.write("Ödüllü çocuk kitapları")
st.image(image_children,width=200)
st.write("---")


st.write("Bilim ve Popüler Kültür")

st.image(image_bilim,width=200,)





