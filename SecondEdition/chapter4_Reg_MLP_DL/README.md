## Chapter 4 code

* Iris data
* Wine Quality data

## Updated Deployment code and repo

* https://github.com/rcalix1/ConstraintsMLprediction/tree/main/Deployment/RealData/Run10
* LEPINE
* 

```

import torch

# Your trained model
model_Forward.eval().float()


dummy_input = torch.randn(1, 7, dtype=torch.float32)

# Export to ONNX (single output called "output1")
torch.onnx.export(
    model_Forward,
    dummy_input,
    "ONNXmodels/LEPINE_model_Forward.onnx",
    input_names=["input1"],
    output_names=["output1"],
    opset_version=15,              # fine for onnxruntime-web
    do_constant_folding=True,
    dynamic_axes={
        "input1": {0: "batch"},
        "output1": {0: "batch"}
    }
)
print("ONNX model saved")


```

## Important Papers:

* Vanishing gradients problem discovered in paper by Glorot and Bengio: Understanding the difficulty of training deep feedforward neural networks (https://www.semanticscholar.org/paper/Understanding-the-difficulty-of-training-deep-Glorot-Bengio/ea9d2a2b4ce11aaf85136840c65f3bc9c03ab649)
