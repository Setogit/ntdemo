# -*- coding: utf-8 -*-
'''
Matrix Factorization for Chemical Fragment-Protein Target activity prediction

# We know 10 chemical compounds and 10 protein targets.
# There are known interactions/activities shown in the comp_target_interaction.
# Each compound is made of a combination of 5 chemical fragments.
# Can we identify latent similarity among the 5 chemical fragments?
'''
from flask import Flask, request
import json
from pyspark.sql import SparkSession
from utils import generate_locally, get_fragment_names, show_fragment_weight

app = Flask(__name__)

spark = None
spark_view = 'score_view'
spark_key = 'score_key'

@app.route("/cosine/score", methods=['GET'])
def cosine_score():
  global spark, spark_view, spark_key
  df = spark.sql("SELECT " + spark_key + " from " + spark_view)
  jsonlist = None
  for record in df.collect():
    jsonlist = record
  return jsonlist

def write_to_spark(cosine_scores):
  global spark, spark_view, spark_key
  scores = {spark_key: json.dumps(cosine_scores)}
  json_file = 'cosine_scores.json'
  with open(json_file, 'w') as f:
    f.write(json.dumps(scores))
  spark = SparkSession \
    .builder \
    .getOrCreate()
  df = spark.read.load(json_file, format="json")
  df.createOrReplaceTempView(spark_view)

def main():
  import os
  fragment_names = get_fragment_names()
  cosine_scores, model = generate_locally()
  write_to_spark(cosine_scores)
  host = os.getenv('NTDEMO_HOST', '0.0.0.0')
  try:
    port = int(os.getenv('NTDEMO_PORT', 3030))
  except:
    port = 3030
  app.run(host=host, port=port)

if __name__ == '__main__':
  main()
