# -*- coding: utf-8 -*-
import json, sys

def load_data(json_file):
  f = None
  data = None
  try:
    f = open(json_file, 'r')
    data = json.loads(f.read())
  except:
    print(sys.exc_info())
    print('Default data is used instead of "{}".'.format(json_file))
    default_data = {
      "fragment_comp": {
        # Compounds (drugs) are made of Chemical Fragments
        #           C O M P O U N D S
        #      c0,c1,c2,c3,c4,c5,c6,c7,c8,c9
        'fA': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0], # fragment A
        'fB': [0, 0, 1, 0, 1, 0, 1, 0, 0, 0], # fragment B
        # fB is contained in compound 2, 4, and 6
        'fC': [1, 0, 1, 0, 0, 0, 1, 0, 0, 1], # fragment C
        # fC is contained in compound 0, 2, 6, and 9
        'fD': [0, 0, 1, 0, 0, 1, 1, 0, 1, 0], # fragment D
        'fE': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1], # fragment E
      },
      # Compounds (drugs) interact with Protein Targets (diseases)
      "comp_target_interaction": [
        #       T A R G E T S
        #t0,t1,t2,t3,t4,t5,t6,t7,t8,t9  # COMPOUNDS
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0], # compound 0
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0], # compound 1
        [1, 0, 1, 0, 1, 0, 0, 1, 1, 0], # compound 2
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1], # compound 3
        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0], # compound 4
        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0], # compound 5
        [1, 1, 0, 1, 1, 0, 0, 1, 1, 1], # compound 6
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0], # compound 7
        [0, 1, 0, 0, 0, 0, 1, 1, 1, 0], # compound 8
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 1], # compound 9
      ]
    }
    data = default_data
  if f:
    f.close()
  return data["fragment_comp"], data["comp_target_interaction"]

fragment_comp, comp_target_interaction = load_data('/data/data.json')
fragment_names = [name for name in fragment_comp]
for frag in fragment_comp:
  assert len(fragment_comp[frag]) == len(comp_target_interaction), \
    'fragment_comp row and comp_target_interaction column should be the same length in "/data/data.json".'

n_compounds = len(comp_target_interaction)
n_targets = len(comp_target_interaction[0])
n_fragments = len(fragment_comp)
n_factors = 3 # n_fragments//2 + 1 # rank of matrix factorization


def get_fragment_names():
  return fragment_names

def get_fragment_target_scores():
  fragment_target_scores = [[0 for _ in range(n_targets)] for _ in range(n_fragments)]
  for frag_ix, frag in enumerate(fragment_comp):
    for target_ix in range(n_targets):
      score = 0
      for comp_ix in range(n_compounds):
        if fragment_comp[frag][comp_ix]:
          score += comp_target_interaction[comp_ix][target_ix]
      fragment_target_scores[frag_ix][target_ix] = score
  # print('Fragment-Target Scores', fragment_target_scores)
  return fragment_target_scores

fragment_target_scores = get_fragment_target_scores() 

#
# Recommender Model
#________________________________________________

def export_onnx_model(model, name='my_model.onnx'):
  import torch.onnx
  dummy_input = [torch.LongTensor([0]), torch.LongTensor([1])]
  torch.onnx._export(model, dummy_input, name, export_params=True)
  return name

def import_onnx_model(name='my_model.onnx'):
  print('PyTorch 1.0 does not yet support importing ONNX.')
  return None

def run_onnx_model_in_caffe2(orig_model, name='my_model.onnx'):
  import numpy as np
  import math, onnx
  from caffe2.python.onnx.backend import run_model
  from caffe2.python import core, workspace

  model_file = onnx.load(name)
  onnx.checker.check_model(model_file)
  print(onnx.helper.printable_graph(model_file.graph))
  is_close = True
  for row in range(n_fragments):
    for col in range(n_targets):
      target = fragment_target_scores[row][col]
      # TODO
      # check_model seems fine.
      # printable_graph shows reasonable model.
      # but, the line below fails with:
      #   RuntimeError: Inferred shape and existing shape differ in rank: (0) vs (1)
      prediction = run_model(model_file, np.array([row, col]).astype(np.float32))
      if not math.isclose(prediction, target, rel_tol=1e-4):
        is_close = False
        print('~~~~~~~~', prediction, target)
  return is_close

def calculate_cosice_scores(model, n_fragments):
  import torch
  cosine_scores = torch.Tensor(n_fragments, n_fragments)
  CS = torch.nn.CosineSimilarity()
  for i in range(n_fragments):
    i_vec = model.fragment_factors.weight[i].unsqueeze(dim=0)
    for k in range(n_fragments):
      k_vec = model.fragment_factors.weight[k].unsqueeze(dim=0)
      cosine_scores[i, k] = CS(i_vec, k_vec).mul(100).type(torch.LongTensor)
  cosine_scores = cosine_scores.numpy().tolist()
  return cosine_scores

def generate_model(n_epoch=1500):
  import torch
  torch.manual_seed(7)

  # Model Definition
  #___________________

  class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_fragments, n_targets, n_factors=2):
      super(MatrixFactorization, self).__init__()
      self.fragment_factors = torch.nn.Embedding(n_fragments, n_factors)
      self.target_factors = torch.nn.Embedding(n_targets, n_factors)
    # The matrix factorization prediction is the dot product
    # between user and item latent feature vectors.
    def forward(self, vec_ixs):
      user = self.fragment_factors(vec_ixs[0])
      item = self.target_factors(vec_ixs[1])
      pred = (user * item).sum(1)
      return pred

  # Model Training
  #___________________

  model = MatrixFactorization(n_fragments, n_targets, n_factors=n_factors)
  loss_func = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  rows, cols = list(range(n_fragments)), list(range(n_targets))

  for epoch in range(n_epoch):
    for row in range(n_fragments):
      for col in range(n_targets):
        target = torch.FloatTensor([fragment_target_scores[row][col]])
        prediction = model([torch.LongTensor([row]), torch.LongTensor([col])])
        loss = loss_func(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch%100 == 0:
      print('Training epoch:{}, Loss:{}'.format(epoch, loss.item()))

  onnx_file = export_onnx_model(model)
  imported_model = import_onnx_model(onnx_file)
  model = imported_model if imported_model else model

  # Results
  #___________________

  cosine_scores = calculate_cosice_scores(model, n_fragments)

  return cosine_scores, model

#
# Visualization
#________________________________________________
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import dash
import dash_html_components as html
import dash_core_components as dcc

def show_scores(labels, scores):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(scores, cmap='pink')
  fig.colorbar(cax)
  # The first [''] is consumed by matplotlib.
  xtick_labels = [''] + [label[:2] for label in labels]
  ytick_labels = [''] + labels
  ax.set_title('Fragment Cosine Similarity by Matrix Factorization')
  ax.set_xticklabels(xtick_labels, rotation=90)
  ax.set_yticklabels(ytick_labels)
  # Show label at every tick
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
  plt.show()

def show_fragment_weight(labels, model):
  app = dash.Dash('matrix_factorization')
  
  print(model.fragment_factors.weight)
  fragment_weights = model.fragment_factors.weight.t()
  fragment_weights = fragment_weights.squeeze().detach().numpy().tolist()
  # print([*zip(*fragment_weights)])

  def scatter_plot_3d(
          x = fragment_weights[0],
          y = fragment_weights[1],
          z = fragment_weights[2],
          size = 10,
          color = 10,
          xlabel = 'X',
          ylabel = 'Y',
          zlabel = 'Z',
          plot_type = 'scatter3d',
          markers = [] ):

      def axis_template_3d(title):
          return dict(
              showbackground = True,
              backgroundcolor = 'rgb(255, 255, 255)',
              gridcolor = 'rgb(200, 200, 200)',
              title = title,
              type = 'linear',
              zerolinecolor = 'rgb(255, 0, 0)'
          )

      data = [ dict(
          x = x,
          y = y,
          z = z,
          mode = 'markers+text',
          text = labels,
          type = plot_type,
      ) ]

      layout = dict(
          font = dict( family = 'Helvetica' ),
          margin = dict( r=20, t=30, l=20, b=20 ),
          showlegend = False,
          title='Fragment Embedding Weights (ranks={})'.format(n_factors),
          scene = dict(
              xaxis = axis_template_3d(xlabel),
              yaxis = axis_template_3d(ylabel),
              zaxis = axis_template_3d(zlabel),
              camera = dict(
                  up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=0.08, y=2.2, z=0.08)
              )
          )
      )
      return dict(data=data, layout=layout)

  FIGURE = scatter_plot_3d()
  app.layout = html.Div([
              dcc.Graph(id='3d-graph',
                        style=dict(width='1000px'),
                        hoverData=dict( points=[dict(pointNumber=0)] ),
                        figure=FIGURE ),
          ], className='3d-graph', style=dict(textAlign='center'))

  app.run_server()
