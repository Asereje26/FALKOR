from piper_sample_generator.__main__ import generate_samples as _generate_samples

MODEL_PATH = "/content/piper-sample-generator/models/en-us-libritts-high.pt"

def generate_samples(*args, **kwargs):
    kwargs.pop("auto_reduce_batch_size", None)
    kwargs["model"] = MODEL_PATH
    return _generate_samples(*args, **kwargs)
