import os
import networkx as nx
import json

def get_graph(kernel_name, all_graphs):
  if kernel_name in all_graphs.keys():
    return all_graphs[kernel_name]
  g_path = os.path.join("train_data", "data", "graphs", f"{kernel_name}_processed_result.gexf")
  g = nx.read_gexf(g_path)
  all_graphs[kernel_name] = g
  return g

def get_src_code(kernel_name, all_src_codes):
  if kernel_name in all_src_codes.keys():
    return all_src_codes[kernel_name]
  g_path = os.path.join("train_data", "data", "sources", f"{kernel_name}_kernel.c")
  with open(g_path, "r") as f:
    src = f.read()
  all_src_codes[kernel_name] = src
  return src

class Design:
  def __init__(self, kernel_name, version, design, all_graphs, all_src_codes):
    self.kernel_name = kernel_name
    self.version = version
    self.design = design['point']
    self.graph = get_graph(kernel_name, all_graphs)
    self.src_code = get_src_code(kernel_name, all_src_codes)
    self.valid = design['valid']
    self.perf = design['perf']
    self.res_util = design['res_util']

if __name__ == '__main__':
    all_graphs = {}
    all_src_codes = {}
    train_designs = []
    for version in ['v18', 'v20', 'v21']:
      design_path = os.path.join("train_data", "data", "designs", f"{version}")
      for fname in os.listdir(design_path):
	if 'json' not in fname:
	  continue
	with open(os.path.join(design_path, fname), 'r') as f:
	  design_points = json.load(f)
	kernel_name = fname.split('.')[0]
	if kernel_name == 'stencil':
	  kernel_name = 'stencil_stencil2d'
	for key, points in design_points.items():
	  data = Design(kernel_name, version, points)
	  train_designs.append(data)
    print(len(train_designs))
    print(train_designs[0].src_code)
    print(train_designs[0].graph)
    print(train_designs[0].design)
    print(train_designs[0].valid)
    print(train_designs[0].perf)
    print(train_designs[0].res_util)
