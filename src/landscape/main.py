import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
import torch
from mutation import vectorizeSequence, get_vee_seqs
from models import load_vee_model, CNN
import numpy as np
import umap
df = get_vee_seqs(nbr_bases=4,size=400)
wmodel = load_vee_model()
input_x = np.array([vectorizeSequence(v) for v in df.seq])
layer_out, pred= wmodel.extract_layer_output(torch.Tensor(input_x),"conv3") # linear, conv1 conv2 conv3
current = 0
grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
# pred = MinMaxScaler().fit_transform(pred)
scores = pred.flatten()
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,target_metric="l1")
# conv_output_reduce = reducer.fit_transform(layer_out,y=pred)
conv_output_reduce = reducer.fit_transform(layer_out)
conv_output_reduce = MinMaxScaler().fit_transform(conv_output_reduce)
index = df[df.distance == 0].index[0]
highlight_point = conv_output_reduce[index]
highlight_score = scores[index]
grid_data = griddata(conv_output_reduce, scores, (grid_x, grid_y), method="cubic") # linear, cubic, nearest
print(current, max(scores), min(scores), np.nan_to_num(grid_data).max(),np.nan_to_num(grid_data).min())
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
ax.set_axis_off()
surface =ax.plot_surface(grid_x, grid_y,  grid_data, cmap='viridis',alpha=0.5,antialiased=True)
ax.scatter(highlight_point[0], highlight_point[1], highlight_score, color="black", s=50, label="Highlighted Sequence",
           edgecolors='k')
high, low = max(scores), min(scores)
cbar = fig.colorbar(surface, ax=ax,ticks=[])
cbar.set_label('Fitness', rotation=270, labelpad=15)
ax.set_title(f"min_dist")
plt.show()