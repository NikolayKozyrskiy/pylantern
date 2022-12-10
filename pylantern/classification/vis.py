import imageio
from matches.loop import Loop
import wandb

from ..common.utils import tensor_to_image
from .pipeline import ClassificationPipeline


def images_gt(pipeline: ClassificationPipeline):
    return pipeline.gt_images().detach().clone()


def log_to_wandb_gt_pred_labels(
    loop: Loop, pipeline: ClassificationPipeline, prefix: str, img_num: int = 10
):
    with loop.mode(mode="valid"):
        pred_labels = pipeline.predict_labels().detach().cpu()
        images = pipeline.gt_images().detach().cpu()

    ep = loop.iterations.current_epoch
    data = {"epochs": ep}
    for name, im, pred_l in zip(
        pipeline.batch.name,
        images[:img_num],
        pred_labels[:img_num],
    ):
        id_ = f"{prefix}/{name}_pred_{pred_l}"
        root = loop.logdir / f"history/image/{prefix}"
        root.mkdir(exist_ok=True, parents=True)

        res_path = root / f"{name}__{ep:03d}.jpg"
        imageio.v3.imwrite(
            res_path, tensor_to_image(im, keepdim=False, val_range=(-1.0, 1.0))
        )
        data[f"images/{id_}"] = wandb.Image(str(res_path.resolve()), caption=f"ep={ep}")

    wandb.log(
        data,
        commit=False,
    )
