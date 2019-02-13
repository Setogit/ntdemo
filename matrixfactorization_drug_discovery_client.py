# -*- coding: utf-8 -*-
'''
Matrix Factorization for Chemical Fragment-Protein Target activity prediction

# We know 10 chemical compounds and 10 protein targets.
# There are known interactions/activities shown in the comp_target_interaction.
# Each compound is made of a combination of 5 chemical fragments.
# Can we identify latent similarity among the 5 chemical fragments?
'''

GET_METRICS_FROM_SERVER = True

#
# Data Definition
#________________________________________________

import http.client
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from server.utils import generate_locally, get_fragment_names, show_fragment_weight

def show_scores(labels, scores):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(scores, cmap='pink')
  fig.colorbar(cax)
  # The first [''] is consumed by matplotlib.
  xtick_labels = [''] + labels
  ytick_labels = [''] + labels
  ax.set_title('Fragment Cosine Similarity by Matrix Factorization')
  ax.set_xticklabels(xtick_labels, rotation=90)
  ax.set_yticklabels(ytick_labels)
  # Show label at every tick
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  plt.show()

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
    data = res.read()
    data = data.decode("utf-8")
    data = '{"scores":' + data + '}'
    data = json.loads(data)
  else:
    print('error:', res.status, res.reason)
    conn.close()
    return
  conn.close()
  return data["scores"]

def main():
  import sys
  fragment_names = get_fragment_names()
  GET_METRICS_FROM_SERVER = len(sys.argv) == 1
  if GET_METRICS_FROM_SERVER:
    show_scores(fragment_names, get_from_server())
  else:
    scores, model = generate_locally()
    show_scores(fragment_names, scores)
    show_fragment_weight(fragment_names, model)

if __name__ == '__main__':
  main()