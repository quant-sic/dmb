from torch.utils.data import Dataset
from dmb.data.dim_2d.worm.helpers.sim import WormSimulation
from pathlib import Path
from functools import cached_property
from pathlib import Path
from typing import  List,Union
from torch.utils.data import Dataset
from dmb.utils import create_logger
from tqdm.auto import tqdm
import os
import torch
from joblib import delayed
from dmb.utils.io import ProgressParallel
import itertools
import math
import numpy as np

log = create_logger(__name__)

class BoseHubbardDataset(Dataset):

    """Dataset for the Bose-Hubbard model."""

    def __init__(self, data_dir:Union[str,Path],
                 observables:List[str] = ["Density_Distribution","Density_Matrix","DensDens_CorrFun","DensDens_CorrFun_local_0","DensDens_CorrFun_local_1","DensDens_CorrFun_local_2","DensDens_CorrFun_local_3","DensDens_Diff_0","DensDens_Diff_1","DensDens_Diff_2","DensDens_Diff_3","DensDens_Diff_Diag_0","DensDens_Diff_Diag_1","DensDens_Diff_Diag_2","DensDens_Diff_Diag_3","DensDens_CorrFun_local_2_step_0","DensDens_CorrFun_local_2_step_1","DensDens_CorrFun_local_2_step_2","DensDens_CorrFun_local_2_step_3","DensDens_CorrFun_local_diag_0","DensDens_CorrFun_local_diag_1","DensDens_CorrFun_local_diag_2","DensDens_CorrFun_local_diag_3","DensDens_CorrFun_sq_0","DensDens_CorrFun_sq_1","DensDens_CorrFun_sq_2","DensDens_CorrFun_sq_3","DensDens_CorrFun_sq_0_","DensDens_CorrFun_sq_1_","DensDens_CorrFun_sq_2_","DensDens_CorrFun_sq_3_","DensDens_CorrFun_sq_diag_0","DensDens_CorrFun_sq_diag_1","DensDens_CorrFun_sq_diag_2","DensDens_CorrFun_sq_diag_3","DensDens_CorrFun_sq_diag_0_","DensDens_CorrFun_sq_diag_1_","DensDens_CorrFun_sq_diag_2_","DensDens_CorrFun_sq_diag_3_","DensDens_CorrFun_sq_0","Density_Distribution_squared"], 
                 base_transforms=None, 
                 train_transforms=None,
                 clean=True):

        self.data_dir = Path(data_dir).resolve()
        self.observables = observables

        log.info(f"Loading {self.__class__.__name__} dataset from {self.data_dir}")

        self.base_transforms = base_transforms
        self.train_transforms = train_transforms

        self.clean = clean

    @cached_property
    def sim_dirs(self):
        sim_dirs = sorted(self.data_dir.glob("*"))

        if self.clean:
            sim_dirs = self._clean_sim_dirs(self.observables,sim_dirs)

        return sim_dirs

    @staticmethod
    def _clean_sim_dirs(observables,sim_dirs):

        def filter_fn(sim_dir):

            try:
                sim = WormSimulation.from_dir(sim_dir)
            except:
                return False

            if "clean" in sim.record and sim.record["clean"] == False:
                return False
            elif "clean" in sim.record and sim.record["clean"] == True:
                valid = True
            else:
                try:
                    sim.results.observables["Density_Distribution"]["mean"]["value"]
                    valid = True
                except:
                    valid = False

                finally:
                    sim.record["clean"] = valid

                    # after saving the record, return if not valid
                    if not valid:
                        return False

            # general purpose validation
            sim.record["clean"] = valid

            # check if all observables are present
            if valid:

                if "observables" not in sim.record:
                    sim.record["observables"] = list(sim.results.observables.keys())
                
                obs_present = set(observables).issubset(set(sim.record["observables"]))
                valid = valid and obs_present

            return valid
        
        
        sim_dirs = list(itertools.compress(sim_dirs,ProgressParallel(n_jobs=10, total=len(sim_dirs), desc="Filtering Dataset",use_tqdm=False)(delayed(filter_fn)(sim_dir) for sim_dir in sim_dirs)))

        return sim_dirs

    def __len__(self):
        return len(self.sim_dirs)
    
    @property
    def loaded_samples(self):
        """Dict that stores loaded samples."""
        if not hasattr(self,"_loaded_samples"):
            self._loaded_samples = {}
        return self._loaded_samples

    @staticmethod
    def get_ckeckerboard_projection(mu_input):
        """
            Determines and returns the checkerboard version (out of two possible) that has the largest correlation with the input mu.
        """

        if len(mu_input.shape) == 1:
            mu_input = mu_input.view(int(math.sqrt(mu_input.shape[0])),int(math.sqrt(mu_input.shape[0])))
        elif len(mu_input.shape) == 2:
            if not mu_input.shape[0] == mu_input.shape[1]:
                raise ValueError("Input mu has to be square")
        else:
            raise ValueError("Input mu has to be either 1D or 2D")

        cb_1 = torch.ones_like(mu_input)
        cb_1[::2,::2]=0
        cb_1[1::2,1::2]=0

        cb_2 = torch.ones_like(mu_input)
        cb_2[1::2,::2]=0
        cb_2[::2,1::2]=0

        corr_1 = torch.abs(torch.sum(mu_input*cb_1))
        corr_2 = torch.abs(torch.sum(mu_input*cb_2))

        if corr_1 > corr_2:
            return cb_1
        else:
            return cb_2
        

    def load_sample(self,idx,reload=False):

        if idx not in self.loaded_samples or reload:
            sim_dir = self.sim_dirs[idx]

            inputs_path = sim_dir / "inputs.pt"
            outputs_path = sim_dir / "outputs.pt"

            if not inputs_path.exists() or not outputs_path.exists() or reload:

                sim = WormSimulation.from_dir(sim_dir)
                
                inputs = torch.concat([
                    torch.from_numpy(sim.input_parameters.mu).float()[None,:],
                    self.get_ckeckerboard_projection(torch.from_numpy(sim.input_parameters.mu).float()).flatten()[None,:],
                    torch.from_numpy(sim.input_parameters.U_on).float()[None,:],
                    torch.from_numpy(sim.input_parameters.V_nn).float().view(2, -1)[[0]]],dim=0)
                
                # stack observables
                outputs = torch.stack([torch.from_numpy(sim.results.observables[obs]["mean"]["value"]).float() for obs in self.observables],dim=0)

                # reshape
                outputs = outputs.view(outputs.shape[0],int(math.sqrt(outputs.shape[1])),int(math.sqrt(outputs.shape[1])))
                inputs = inputs.view(inputs.shape[0],int(math.sqrt(inputs.shape[1])),int(math.sqrt(inputs.shape[1])))

                #save to .npy files
                torch.save(inputs,inputs_path)
                torch.save(outputs,outputs_path)
            
            else:
                inputs = torch.load(inputs_path)
                outputs = torch.load(outputs_path)

            self.loaded_samples[idx] = (inputs,outputs)

        return self.loaded_samples[idx]

    def __getitem__(self, idx,reload=False):

        inputs,outputs = self.load_sample(idx,reload=reload)

        return inputs, outputs

    def get_sim(self,idx):
        
        sim_dir = self.sim_dirs[idx]
        sim = WormSimulation.from_dir(sim_dir)

        return sim

    def get_parameters(self,idx):

        sim_dir = self.sim_dirs[idx]
        sim = WormSimulation.from_dir(sim_dir)

        return sim.input_parameters
    
    def phase_diagram_position(self,idx):

        pars = self.get_parameters(idx)
        U_on = pars.U_on
        mu = float(pars.mu_offset)
        J = pars.t_hop
        V_nn = pars.V_nn

        # return mu,U_on,J
        return 4*V_nn[0]/U_on[0],mu/U_on[0],4*J[0]/U_on[0]

    

    def get_dataset_ids_from_indices(self,indices:Union[int,List[int]]) -> Union[str,List[str]]:

        ids = [self.sim_dirs[idx].name for idx in (indices if isinstance(indices,list) else [indices])]

        if isinstance(indices,int):
            ids = ids[0]

        return ids

        