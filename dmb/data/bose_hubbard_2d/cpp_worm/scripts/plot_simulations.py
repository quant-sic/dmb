import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from dmb.data.dim_2.worm.helpers.sim import WormSimulation
from pathlib import Path
from dmb.data.dim_2.dataset import BoseHubbardDataset
from dmb.utils.paths import REPO_DATA_ROOT
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    ds = BoseHubbardDataset(data_dir=REPO_DATA_ROOT/"bose_hubbard_2d")

    print(f"Number of simulations: {len(ds)}")
        
    for i in tqdm(range(len(ds))):
            
        for observable in ds[i][1].keys():

            # plot the density distribution, its error and the chemical potential in one row
            fig,ax = plt.subplots(1,4,figsize=(15,5))

            try:
                inputs, outputs = ds[i]

                ax[0].imshow(outputs[observable]["mean"]["value"].reshape(int(np.sqrt(len(outputs[observable]["mean"]["value"]))),-1))
                ax[0].set_title(observable)

                ax[1].imshow(outputs[observable]["mean"]["error"].reshape(int(np.sqrt(len(outputs[observable]["mean"]["error"]))),-1))
                ax[1].set_title(f"{observable} Error")

                ax[2].imshow(inputs.reshape(int(np.sqrt(len(inputs))),-1))
                ax[2].set_title("Chemical Potential")

                # relative error
                ax[3].imshow(outputs[observable]["mean"]["error"].reshape(int(np.sqrt(len(outputs[observable]["mean"]["error"]))),-1)/outputs[observable]["mean"]["value"].reshape(int(np.sqrt(len(outputs[observable]["mean"]["value"]))),-1))
                ax[3].set_title("Relative Error")

                for a in ax:
                    a.set_xticks([])
                    a.set_yticks([])

                # add colorbars
                fig.colorbar(ax[0].imshow(outputs[observable]["mean"]["value"].reshape(int(np.sqrt(len(outputs[observable]["mean"]["value"]))),-1)),ax=ax[0])
                fig.colorbar(ax[1].imshow(outputs[observable]["mean"]["error"].reshape(int(np.sqrt(len(outputs[observable]["mean"]["error"]))),-1)),ax=ax[1])
                fig.colorbar(ax[2].imshow(inputs.reshape(int(np.sqrt(len(inputs))),-1)),ax=ax[2])
                fig.colorbar(ax[3].imshow(outputs[observable]["mean"]["error"].reshape(int(np.sqrt(len(outputs[observable]["mean"]["error"]))),-1)/outputs[observable]["mean"]["value"].reshape(int(np.sqrt(len(outputs[observable]["mean"]["value"]))),-1)),ax=ax[3])

                plt.savefig(ds.sim_dirs[i]/"plots/sim_result.png",dpi=100)
                plt.close()
            
            except:
                continue
            