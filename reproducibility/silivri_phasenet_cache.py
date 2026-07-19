import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
import seisbench.models as sbm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kfold_environment import KFoldEnvironment


OUTPUT = Path(__file__).resolve().parent.parent / "phasenet_instance_silivri_scores.npz"


def main():
    tf.config.set_visible_devices([], "GPU")
    model = sbm.PhaseNet.from_pretrained("instance")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    p_index = list(model.labels).index("P")
    environment = KFoldEnvironment(
        "SILIVRI2019",
        apply_resampling=False,
        resample_eq_ratio=0.5,
        resample_while_keeping_total_waveforms_fixed=False,
    )
    _, _, _, generator = environment.get_generators(0)
    outputs = []
    with torch.inference_mode():
        for batch_index in range(len(generator)):
            batch = np.asarray(generator[batch_index], dtype=np.float32)
            batch = np.transpose(batch[:, :, [2, 1, 0]], (0, 2, 1))
            batch = batch - batch.mean(axis=2, keepdims=True)
            batch = batch / (batch.std(axis=2, keepdims=True) + 1e-10)
            batch = np.pad(batch, ((0, 0), (0, 0), (0, 1)))
            prediction = model(torch.from_numpy(batch).to(device))
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            if prediction.min() < 0 or prediction.max() > 1:
                prediction = torch.softmax(prediction, dim=1)
            outputs.append(prediction[:, p_index, :].amax(dim=1).cpu().numpy())
            print(f"PhaseNet completed:{batch_index + 1}/{len(generator)}")
    scores = np.concatenate(outputs)
    np.savez_compressed(OUTPUT, scores=scores)
    print(OUTPUT)


if __name__ == "__main__":
    main()
