# Quick run

After installing MMipt successfully, now you are able to play with MMipt! To generate an image from text, you only need several lines of codes by MMipt!

```python
from mmipt.apis import MMiptInferencer
sd_inferencer = MMiptInferencer(model_name='stable_diffusion')
text_prompts = 'A panda is having dinner at KFC'
result_out_dir = 'output/sd_res.png'
sd_inferencer.infer(text=text_prompts, result_out_dir=result_out_dir)
```

Or you can just run the following command.

```bash
python demo/mmipt_inference_demo.py \
    --model-name stable_diffusion \
    --text "A panda is having dinner at KFC" \
    --result-out-dir ./output/sd_res.png
```

You will see a new image `sd_res.png` in folder `output/`, which contained generated samples.

What's more, if you want to make these photos much more clear,
you only need several lines of codes for image super-resolution by MMipt!

```python
from mmipt.apis import MMiptInferencer
config = 'configs/esrgan/esrgan_x4c64b23g32_1xb16-400k_div2k.py'
checkpoint = 'https://download.openmmlab.com/mmediting/restorers/esrgan/esrgan_x4c64b23g32_1x16_400k_div2k_20200508-f8ccaf3b.pth'
img_path = 'tests/data/image/lq/baboon_x4.png'
editor = MMiptInferencer('esrgan', model_config=config, model_ckpt=checkpoint)
output = editor.infer(img=img_path,result_out_dir='output.png')
```

Now, you can check your fancy photos in `output.png`.
