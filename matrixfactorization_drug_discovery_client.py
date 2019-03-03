# -*- coding: utf-8 -*-
'''
Matrix Factorization for Chemical Fragment-Protein Target activity prediction

# We know 10 chemical compounds and 10 protein targets.
# There are known interactions/activities shown in the comp_target_interaction.
# Each compound is made of a combination of 5 chemical fragments.
# Can we identify latent similarity among the 5 chemical fragments?
'''

import http.client
import json
from server.utils import generate_model, get_fragment_names, show_fragment_weight, show_scores

def get_from_server():
  import os
  host = os.getenv('NTDEMO_HOST', 'localhost')
  try:
    port = str(int(os.getenv('NTDEMO_PORT', '3030')))
  except:
    port = '3030'
  conn = None
  try:
    conn = http.client.HTTPConnection(host+':'+port)
    conn.request('GET', '/cosine/score', )
    res = conn.getresponse()
  except:
    print('Connection to the server failed.')
    if conn:
      conn.close()
    return
  data = None
  if res.status==200 and res.reason=='OK':
    data = res.read().decode("utf-8")
    data = json.loads(data)
    data = {"names": data["names"], "scores": data["scores"]}
  else:
    print('error:', res.status, res.reason)
  conn.close()
  return data

def main():
  import sys

  GET_METRICS_FROM_SERVER = len(sys.argv) == 1
  if GET_METRICS_FROM_SERVER:
    data = get_from_server()
    if data:
      show_scores(data["names"], data["scores"])
  else:
    fragment_names = get_fragment_names()
    scores, model = generate_model()
    show_scores(fragment_names, scores)
    show_fragment_weight(fragment_names, model)

if __name__ == '__main__':
  main()