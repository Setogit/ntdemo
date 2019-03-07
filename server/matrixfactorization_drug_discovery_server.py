# -*- coding: utf-8 -*-
'''
Matrix Factorization for Chemical Fragment-Protein Target activity prediction

# We know 10 chemical compounds and 10 protein targets.
# There are known interactions/activities shown in the comp_target_interaction.
# Each compound is made of a combination of 5 chemical fragments.
# Can we identify latent similarity among the 5 chemical fragments?
'''
from flask import Flask
import json
from pyspark.sql import SparkSession
from utils import generate_model, get_fragment_names, show_fragment_weight, show_scores

app = Flask('ntdemo')

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
  return str(jsonlist)

def write_to_spark(names, scores):
  global spark, spark_view, spark_key
  assert len(names) == len(scores)
  cosine_scores = {'names': names, 'scores': scores}
  scores = {spark_key: json.dumps(cosine_scores)}
  json_file = 'cosine_scores.json'
  with open(json_file, 'w') as f:
    f.write(json.dumps(scores))
  spark = SparkSession \
    .builder \
    .getOrCreate()
  df = spark.read.load(json_file, format="json")
  df.createOrReplaceTempView(spark_view)

def flask_app_run(host, port):
  app.run(host=host, port=port, debug=False)

def main():
  import os
  from multiprocessing import Process
  fragment_names = get_fragment_names()
  cosine_scores, model = generate_model()
  write_to_spark(fragment_names, cosine_scores)
  host = os.getenv('NTDEMO_HOST', '0.0.0.0')
  try:
    port = int(os.getenv('NTDEMO_PORT', 3030))
  except:
    port = 3030
  p = Process(target=flask_app_run, args=(host, port, ))
  p.start()
  show_scores(fragment_names, cosine_scores)
  p.join()


if __name__ == '__main__':
  main()
