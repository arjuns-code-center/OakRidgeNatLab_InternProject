import numpy as np
from vae import VAE
from sklearn.metrics import accuracy_score
import h5py 
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument("-i", default='vae_input.npy', 
                    help="Input: npy file")
parser.add_argument("-w", default='vae_weight.h5', 
                    help="Input: model weight h5 file")
parser.add_argument("-d", default=3, type=int, 
                    help="Number of dimensions in latent space") 
parser.add_argument("-o", default='vae_output', 
                    help="Output: pred contact map and embedding npy file") 

args = parser.parse_args()
input_file=args.i
model_weight=args.w
hyper_dim=args.d
output_file=args.o

npred = 1000

cm_data = np.load(input_file).astype('float')

vae = VAE(cm_data.shape[1:], hyper_dim)
vae.model.load_weights(model_weight)

cm_pred = np.array([vae.decode(np.expand_dims(sample,axis=0)) for sample in cm_data[:npred]])
cm_emb = np.array([vae.return_embeddings(np.expand_dims(sample,axis=0)) for sample in cm_data])

#np.savez(output_file, pred=cm_pred, emb=cm_emb)
cm_pred = cm_pred.astype(cm_data.dtype)

np.save("%s-embeddings.npy"%output_file, cm_emb)

np.savez("%s-samples.npz"%output_file, pred=np.squeeze(cm_pred[:10]), gtruth=np.squeeze(cm_data[:10]))

acc = accuracy_score((cm_data[:npred].flatten()<0.5)*1,(cm_pred[:npred].flatten()<0.5)*1)
print("Accuracy: ", acc)

