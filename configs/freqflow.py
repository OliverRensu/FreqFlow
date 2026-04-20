import ml_collections
from dataclasses import dataclass


@dataclass
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

model = Args(
    channels = 4,
    num_classes = 1001,
    block_grad_to_lowres = False,
    norm_type = "TDRMSN",
    stage_configs = [
            Args(
                block_type = "TransformerBlock", 
                dim = 960,  # channel
                hidden_dim = 1920,
                num_attention_heads = 16,
                num_blocks = 39,  # depth
                max_height = 16,
                max_width = 16,
                image_input_ratio = 1,
                input_feature_ratio = 2,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
            Args(
                block_type = "ConvNeXtBlock", 
                dim = 480, 
                hidden_dim = 960, 
                kernel_size = 7, 
                num_blocks = 20,
                max_height = 32,
                max_width = 32,
                image_input_ratio = 1,
                input_feature_ratio = 1,
                final_kernel_size = 3,
                dropout_prob = 0,
            ),
    ],
)

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl_ema.pth'
    )

    config.train = d(
        n_steps=1500000,
        batch_size=1024,
        mode='cond',
        log_interval=100,
        eval_interval=5000,
        save_interval=50000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    global model
    config.nnet = d(
        name='freqflow',
        model_args=model,
    )
    config.loss_coeffs = [1/4, 1]
    
    config.dataset = d(
        name='imagenet256_features',
        path='/tmp/rsc/DataSet/imagenet_sd_ema_feature',
        cfg=True,
        p_uncond=0.1
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=50,  # the decoder is large
        algorithm='dpm_solver',
        cfg=True,
        scale=0.4,
        path=''
    )

    return config
