from timm import create_model

def get_mixer_b16():
    base_model = create_model('mixer_b16_224', pretrained=True)
    return base_model