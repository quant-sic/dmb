import argparse

from dmb.data.bose_hubbard_2d.phase_diagram import model_predict, \
    phase_diagram_uniform_inputs, plot_phase_diagram
from dmb.model.lit_model import DMBLitModel
from dmb.utils import REPO_LOGS_ROOT, REPO_ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",
                        type=str,
                        help="Path to checkpoint directory")
    args = parser.parse_args()

    ckpt_dir = REPO_LOGS_ROOT / args.ckpt_dir

    for ckpt_path in list(
            sorted(
                ckpt_dir.glob("*.ckpt"),
                key=lambda x: int(x.stem.split("=")[-1]),
                reverse=True,
            )):
        model = DMBLitModel.load_from_checkpoint(ckpt_path)

        save_dir = ckpt_path.parent.parent.parent / "figures"
        file_name_stem = ckpt_path.stem

        model.plot_model(save_dir=save_dir, file_name_stem=file_name_stem)
