def get_pipeline_class(name):
    if name == "HF_Pipeline":
        from src.pipelines.transformers import HF_Pipeline

        return HF_Pipeline
    elif name == "DS_Pipeline":
        from src.pipelines.ds import DS_Pipeline

        return DS_Pipeline
    else:
        raise NotImplementedError(f"Unsupported pipeline class: {name}")
