from dmb.model.lit_model import DMBLitModel
from dmb.utils import REPO_ROOT,REPO_LOGS_ROOT
from dmb.data.bose_hubbard_2d.phase_diagram import model_predict,phase_diagram_uniform_inputs,plot_phase_diagram
import argparse

def plot_model(ckpt_dir):

    for ckpt_path in list(sorted(ckpt_dir.glob("*.ckpt"),key=lambda x: int(x.stem.split("=")[-1]))):

        print(ckpt_path)

        net = DMBLitModel.load_from_checkpoint(ckpt_path)

        check="max-min"
        
        #mu,ztU,out = model_predict(net,batch_size=512)
        figures = plot_phase_diagram(net,n_samples=100,zVU=1.0)

        save_path = ckpt_path.parent.parent.parent / "figures" / (ckpt_path.stem + check + ".png")
        save_path.parent.mkdir(exist_ok=True,parents=True)

        figures[check].savefig(save_path)
        
        figures = plot_phase_diagram(net,n_samples=100,zVU=1.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str, help="Path to checkpoint directory")
    args = parser.parse_args()

    ckpt_dir = REPO_LOGS_ROOT / args.ckpt_dir

    plot_model(ckpt_dir)