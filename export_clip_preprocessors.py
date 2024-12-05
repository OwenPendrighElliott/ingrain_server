import open_clip
import os
import json


def main():
    config_dir = os.path.join("inference", "triton_open_clip", "preprocessors")
    os.makedirs(config_dir, exist_ok=True)

    exported_model_configs = set(f.split(".")[0] for f in os.listdir(config_dir))
    checkpoints = open_clip.list_pretrained()

    models = []

    for model, pretrained in checkpoints:
        if model in models or model in exported_model_configs:
            continue

        try:
            _, _, preprocess = open_clip.create_model_and_transforms(
                model, pretrained=pretrained
            )
        except RuntimeError as e:
            print(f"Error creating model {model}: {e}")
            continue

        models.append(model)
        transforms = preprocess.transforms
        size = transforms[0].size
        interpolation = transforms[0].interpolation
        mean = transforms[-1].mean
        std = transforms[-1].std

        resize_mode = "squash"

        if not isinstance(size, int):
            resize_mode = "shortest"

        cfg_dict = {
            "size": size,
            "mode": "RGB",
            "mean": mean,
            "std": std,
            "interpolation": str(interpolation.value),
            "resize_mode": resize_mode,
            "fill_color": 0,
        }

        with open(os.path.join(config_dir, f"{model}.json"), "w") as f:
            json.dump(cfg_dict, f, indent=4)


if __name__ == "__main__":
    main()
