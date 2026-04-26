https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m

https://huggingface.co/Marqo/marqo-fashionSigLIP

https://huggingface.co/fashn-ai/fashn-vton-1.5

https://www.kaggle.com/datasets/hserdaraltan/deepfashion-inshop-clothes-retrieval/data

https://github.com/RenxingIntelligence/OpenVTON-Bench

https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct

## Image embedding System Design

## Virtual Try On System Design

Virtual Try On Inference Engine = Comfyui + export api.json


Je veux toujours pour chaque user avoir une image globale qui concatene les image temp intermediaire produis par le pipeline. Cette image globale dans laquelle il y a 6 frame sur deux ligne avec titre le nom de l'objet au dessus, au dessous une description et au centre le contenu: une image. La première frame c'est l'image original, la deuxieme c'est l'image top1 retrieve, la troisieme c'est le garment detecter et la 4eme c'est le résultat VTON, la cinquieme c'est un effet de superposition représentant l'évaluation Pixel et la sixième un plot de la SSIM Map. estce que c'est clair ?