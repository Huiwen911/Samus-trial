from SAMUS.models.segment_anything.build_sam import sam_model_registry
from SAMUS.models.segment_anything_samus.build_sam_us import samus_model_registry

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint="C:\\Users\\owner\\Downloads\\sam_vit_b_01ec64.pth")
    elif modelname == "SAMUS":
        model = samus_model_registry['vit_b'](checkpoint="C:\\Users\\owner\\Downloads\\sam_vit_b_01ec64.pth")
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
