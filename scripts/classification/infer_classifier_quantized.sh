# python3 -m pylantern.classification infer-quantization logs/quant_cifar10_qresnet18/230129_2102/config.py -c "best" -o "infer_best" -g 0
python3 -m pylantern.classification infer-quantization logs/deit_tiny_patch16_224/config_quant.py -c "none" -o "infer_quant_w4a32" -g 0
# python3 -m pylantern.classification infer logs/deit_tiny_patch16_224/config.py -c "none" -o "infer" -g 0
python3 -m pylantern.classification infer-quantization logs/deit_tiny_distilled_patch16_224/config_quant.py -c "none" -o "infer_quant_w4a32" -g 0
# python3 -m pylantern.classification infer logs/deit_tiny_distilled_patch16_224/config.py -c "none" -o "infer" -g 0
python3 -m pylantern.classification infer-quantization logs/deit_small_patch16_224/config_quant.py -c "none" -o "infer_quant_w4a32" -g 0
# python3 -m pylantern.classification infer logs/deit_small_patch16_224/config.py -c "none" -o "infer" -g 0
python3 -m pylantern.classification infer-quantization logs/deit_small_distilled_patch16_224/config_quant.py -c "none" -o "infer_quant_w4a32" -g 0
# python3 -m pylantern.classification infer logs/deit_small_distilled_patch16_224/config.py -c "none" -o "infer" -g 0
python3 -m pylantern.classification infer-quantization logs/deit_base_patch16_224/config_quant.py -c "none" -o "infer_quant_w4a32" -g 0
# python3 -m pylantern.classification infer logs/deit_base_patch16_224/config.py -c "none" -o "infer" -g 0
python3 -m pylantern.classification infer-quantization logs/deit_base_distilled_patch16_224/config_quant.py -c "none" -o "infer_quant_w4a32" -g 0
# python3 -m pylantern.classification infer logs/deit_base_distilled_patch16_224/config.py -c "none" -o "infer" -g 0