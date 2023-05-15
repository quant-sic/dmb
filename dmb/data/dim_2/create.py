from dmb.data.dim_2.worm.helpers.sim import WormInputParameters, WormSimulation
import numpy as np
from pathlib import Path
from dmb.utils.io import create_logger
import os

log = create_logger(__name__)

if __name__ == "__main__":

    L = 16
    mu = np.zeros((L, L))
    mu[3:10,3:10] = 1

    muU = 1.5
    U_on = np.full(shape=(L,L),fill_value= 4 / 0.1)

    mu = np.ones((L, L))* (muU*U_on)

    V_nn = np.expand_dims(U_on/4,axis=0).repeat(2,axis=0)
    t_hop = np.ones((2,L,L))

    #print(mu/U_on,4*V_nn/U_on,4/U_on)

    thermalization = 10000
    sweeps = 1000000
    p = WormInputParameters(Lx=L,Ly=L,Nmeasure2=100,t_hop=t_hop,U_on=U_on,V_nn=V_nn,thermalization=thermalization,mu=mu,sweeps=sweeps)

    save_dir=Path("/u/bale/paper/worm/runs/test_non_uniform")
    import shutil
    shutil.rmtree(save_dir,ignore_errors=True)

    sim = WormSimulation(p,worm_executable="/u/bale/paper/worm/build_non_uniform/qmc_worm_mpi",save_dir=save_dir)

    if not "SLURM_PROCID" in os.environ or os.environ["SLURM_PROCID"] == "0":
        log.info("Saving Parameters")
        sim.save_parameters()

    log.info("Running Simulation")
    sim.run_until_convergence(tune=False)

