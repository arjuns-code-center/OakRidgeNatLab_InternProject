import numpy as np
import CVAE
from sklearn.metrics import accuracy_score
import h5py 
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("-i", default='cvae_input.h5', 
                    help="Input: contact map h5 file")
parser.add_argument("-w", default='cvae_weight.h5', 
                    help="Input: model weight h5 file")
parser.add_argument("-d", default=3, type=int, 
                    help="Number of dimensions in latent space") 
parser.add_argument("-o", default='cvae_output', 
                    help="Output: pred contact map and embedding npy file") 
parser.add_argument("-n", default=1.0, type=float, 
                    help="norm factor") 

args = parser.parse_args()
input_file=args.i
model_weight=args.w
hyper_dim=args.d
output_file=args.o
norm = args.n

cutoff = 8.0/norm
npred = 100000

cm_data = h5py.File(input_file,'r')['contact_maps'].value

cvae = CVAE.autoenc(cm_data.shape[1:], hyper_dim)
cvae.model.load_weights(model_weight)

cm_pred = np.array([cvae.decode(np.expand_dims(sample,axis=0)) for sample in cm_data[:npred]])
cm_emb = np.array([cvae.return_embeddings(np.expand_dims(sample,axis=0)) for sample in cm_data])

np.save("%s-embeddings.npy"%output_file, cm_emb)

np.savez("%s-samples.npz"%output_file, pred=np.squeeze(cm_pred[:15000]), gtruth=np.squeeze(cm_data[:15000]))
cm_pred = cm_pred.astype(cm_data.dtype)

if cm_data.dtype not in (np.int, np.int16, np.int8): 
    acc = accuracy_score((cm_data[:npred].flatten()<cutoff)*1,(cm_pred[:npred].flatten()<cutoff)*1)
else:
    acc = accuracy_score(cm_data[:npred].flatten(),cm_pred[:npred].flatten())

print("Accuracy: ", acc)