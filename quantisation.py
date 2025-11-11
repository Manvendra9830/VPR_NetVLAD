import torch
import torch.quantization as quant
from torch.quantization import prepare, convert, prepare_qat

def static_quantization(model, data_loader, device):
    """
    Applies post-training static quantization to a model.
    Requires calibration with a data_loader.
    """
    print("Applying static quantization...")
    # Force quantization to run on CPU to avoid cuDNN errors
    quant_device = torch.device('cpu')
    model = model.to(quant_device)
    
    model.eval()
    # Specify quantization configuration for CPU
    model.qconfig = quant.get_default_qconfig('fbgemm')
    
    # Do not quantize the NetVLAD layer
    model.pool.qconfig = None

    # Fuse layers for better performance and compatibility
    print("Fusing modules...")
    # The VGG encoder is a nn.Sequential, we can iterate and fuse
    layers_to_fuse = []
    for i in range(len(model.encoder)):
        if isinstance(model.encoder[i], torch.nn.Conv2d) and \
           i + 1 < len(model.encoder) and \
           isinstance(model.encoder[i+1], torch.nn.ReLU):
            layers_to_fuse.append([str(i), str(i+1)])
    
    if layers_to_fuse:
        torch.quantization.fuse_modules(model.encoder, layers_to_fuse, inplace=True)
        print(f"Fused {len(layers_to_fuse)} Conv-ReLU pairs.")

    # Prepare the model for static quantization. This inserts observers in the model
    # that will observe activation tensors during calibration.
    quant.prepare(model, inplace=True)
    
    # Calibrate the model with a small subset of data
    print("Calibrating model...")
    with torch.no_grad():
        for i, (input, _) in enumerate(data_loader):
            # The model forward pass is used to calibrate the observers
            if input.dim() == 4: # Ensure input is a batch of images
                model(input.to(quant_device)) # Move calibration data to CPU
            # Use a small number of batches for calibration
            if i > 10:
                break
    print("Calibration complete.")
    
    # Convert the observed model to a quantized model.
    quant.convert(model, inplace=True)
    print("Static quantization applied.")
    
    # Move the model back to the original device
    model = model.to(device)
    return model

def qat_quantization(model, data_loader, device, epochs=1):
    """
    Applies Quantization-Aware Training (QAT).
    """
    print("Applying Quantization-Aware Training...")
    # Force quantization to run on CPU to avoid cuDNN errors
    quant_device = torch.device('cpu')
    model = model.to(quant_device)

    model.train()
    # Specify quantization configuration for QAT on CPU
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    
    # Fuse layers for better performance and compatibility
    print("Fusing modules for QAT...")
    # The VGG encoder is a nn.Sequential, we can iterate and fuse
    # QAT requires fusion before preparing the model
    layers_to_fuse = []
    for i in range(len(model.encoder)):
        # Check for Conv-BN-ReLU triplets
        if (i + 2 < len(model.encoder) and
            isinstance(model.encoder[i], torch.nn.Conv2d) and
            isinstance(model.encoder[i+1], torch.nn.BatchNorm2d) and
            isinstance(model.encoder[i+2], torch.nn.ReLU)):
            layers_to_fuse.append([str(i), str(i+1), str(i+2)])
        # Check for Conv-ReLU pairs
        elif (i + 1 < len(model.encoder) and
              isinstance(model.encoder[i], torch.nn.Conv2d) and
              isinstance(model.encoder[i+1], torch.nn.ReLU)):
            layers_to_fuse.append([str(i), str(i+1)])

    if layers_to_fuse:
        torch.quantization.fuse_modules(model.encoder, layers_to_fuse, inplace=True)
        print(f"Fused {len(layers_to_fuse)} layer groups for QAT.")

    # Prepare the model for QAT.
    quant.prepare_qat(model, inplace=True)
    
    # QAT requires a few epochs of fine-tuning
    print("Fine-tuning model for QAT...")
    # Note: In a real scenario, you would use your actual training optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) 
    for epoch in range(epochs):
        print(f"QAT Epoch {epoch+1}/{epochs}")
        for i, (input, _) in enumerate(data_loader):
            if input.dim() == 4: # Ensure input is a batch of images
                optimizer.zero_grad()
                output = model(input.to(quant_device)) # Move QAT data to CPU
                # In a real scenario, you would compute a loss based on your task
                # For this example, we use a dummy loss
                loss = output.sum() 
                loss.backward()
                optimizer.step()
            # Use a small number of batches for demonstration
            if i > 10:
                break
    print("QAT fine-tuning complete.")

    # Convert the trained model to a quantized model
    model.eval()
    quant.convert(model, inplace=True)
    print("QAT quantization applied.")
    
    # Move the model back to the original device
    model = model.to(device)
    return model
