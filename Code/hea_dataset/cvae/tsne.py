# coding: utf-8
import numpy as np
import argparse
import cudf
from cuml.manifold import TSNE
from cuml import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
mpl.use('Agg')

def tsne_plot(x_pred_encoded, y_train, plot_name):
    Dmax = y_train
    [n,s] = np.histogram(Dmax, 11)
    d = np.digitize(Dmax, s)
    cmi = plt.get_cmap('jet')
    cNorm = mpl.colors.Normalize(vmin=min(Dmax), vmax=max(Dmax))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cmi)

    x = x_pred_encoded[:,0]
    y = x_pred_encoded[:,1]

    plt.scatter(x, y, c=scalarMap.to_rgba(Dmax), marker="o", s=1, alpha=0.5) 

    scalarMap.set_array(Dmax)
    plt.colorbar(scalarMap)

    plt.savefig(plot_name, dpi=600)
    plt.clf()


parser = argparse.ArgumentParser()
parser.add_argument("-e", default="embeddings.npy", type=str,
                    help="latent space array")
parser.add_argument("-le", default="label_eng.npy", type=str,
                    help="label file")
parser.add_argument("-lt", default="label_T.npy", type=str,
                    help="label file")
args = parser.parse_args()
emb_file = args.e
X = np.load(emb_file)
X = np.squeeze(X)
print("embedded shape:", X.shape)


X1 = X[::1] 
X_embedded = TSNE(n_components=2).fit_transform(X1)
print("after TSNE operation: embedded shape", X_embedded.shape)

np.save("encoded_TSNE_2D.npy", X_embedded)
#lab_eng = args.le
lab_T = args.lt
#label_eng = np.load(lab_eng); 
#label_eng = np.squeeze(label_eng)
label_T = np.load(lab_T)
label_T = np.squeeze(label_T)

#print("label shape:", label_eng.shape);  
#tsne_plot(X_embedded, label_eng, "VAE_ENG")
tsne_plot(X_embedded, label_T, "VAE_T")

