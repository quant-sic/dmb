from pathlib import Path
from dmb.data.dim_2d.dataset import BoseHubbardDataset
import pytorch_lightning as pl
from pathlib import Path
from typing import   Optional, Any, List,Callable,Dict

from typing import   Optional,List
from dmb.utils import create_logger

from torch.utils.data.dataloader import default_collate
from dmb.data.utils import chain_fns
from dmb.data.mixins import DataModuleMixin
from dmb.utils import create_logger
from dmb.data.utils import collate_sizes


log = create_logger(__name__)


class BoseHubbardDataModule(DataModuleMixin):

    def __init__(self, 
                 data_dir:Path, 
                 train_val_test_split:List[float] = (0.8,0.1,0.1),
                 batch_size:int = 64, 
                 num_workers:int=0, 
                 clean=True,
                 base_transforms=None,
                 train_transforms=None,
                 resplit:Optional[List[Dict]] = None,
                 split_usage:Dict[str,int] = {"train":0,"val":1,"test":2},
                 pin_memory:bool = False,
                 observables: List[str] = [
                     "Density_Distribution","Density_Matrix","DensDens_CorrFun","DensDens_CorrFun_local_0","DensDens_CorrFun_local_1","DensDens_CorrFun_local_2","DensDens_CorrFun_local_3","DensDens_Diff_0","DensDens_Diff_1","DensDens_Diff_2","DensDens_Diff_3","DensDens_Diff_Diag_0","DensDens_Diff_Diag_1","DensDens_Diff_Diag_2","DensDens_Diff_Diag_3","DensDens_CorrFun_local_2_step_0","DensDens_CorrFun_local_2_step_1","DensDens_CorrFun_local_2_step_2","DensDens_CorrFun_local_2_step_3","DensDens_CorrFun_local_diag_0","DensDens_CorrFun_local_diag_1","DensDens_CorrFun_local_diag_2","DensDens_CorrFun_local_diag_3","DensDens_CorrFun_sq_0","DensDens_CorrFun_sq_1","DensDens_CorrFun_sq_2","DensDens_CorrFun_sq_3","DensDens_CorrFun_sq_0_","DensDens_CorrFun_sq_1_","DensDens_CorrFun_sq_2_","DensDens_CorrFun_sq_3_","DensDens_CorrFun_sq_diag_0","DensDens_CorrFun_sq_diag_1","DensDens_CorrFun_sq_diag_2","DensDens_CorrFun_sq_diag_3","DensDens_CorrFun_sq_diag_0_","DensDens_CorrFun_sq_diag_1_","DensDens_CorrFun_sq_diag_2_","DensDens_CorrFun_sq_diag_3_","DensDens_CorrFun_sq_0","Density_Distribution_squared"
                     ],
                 save_split_indices:bool = True,
                 ):
        super().__init__()
        self.save_hyperparameters()


    def get_dataset(self):
        return BoseHubbardDataset(self.hparams["data_dir"], clean=self.hparams["clean"],observables=self.hparams["observables"])

    def get_collate_fn(self) -> Optional[Callable]:

        collate_fns: List[Callable[[Any], Any]] = [collate_sizes]

        return chain_fns(collate_fns)