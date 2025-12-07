import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from BitNetMCU import QuantizedModel
from models import FCMNIST
import yaml
import sys
import os

# Must match the logic in training.py to find the file
def create_run_name(hyperparameters):
    runname = hyperparameters["runtag"] + hyperparameters["scheduler"] + '_lr' + str(hyperparameters["learning_rate"]) + ('_Aug' if hyperparameters["augmentation"] else '') + '_BitMnist_' + hyperparameters["WScale"] + "_" +hyperparameters["QuantType"] + "_" + hyperparameters["NormType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"])  + "_bs" + str(hyperparameters["batch_size"]) + "_epochs" + str(hyperparameters["num_epochs"])
    return runname

def export_to_hfile(quantized_model, filename, runname, image_resolution=16):
    """
    Exports the quantized model to an Ansi-C header file.
    """
    if not quantized_model.quantized_model:
        raise ValueError("quantized_model is empty")

    max_n_activations = max([layer['outgoing_weights'] for layer in quantized_model.quantized_model])

    print(f"Writing C header file to: {filename}")
    with open(filename, 'w') as f:
        f.write(f'// Automatically generated header file\n')
        f.write(f'// Date: {datetime.now()}\n')
        f.write(f'// Model: {runname}\n\n')

        f.write('#include <stdint.h>\n\n')
        f.write('#ifndef BITNETMCU_MODEL_H\n')
        f.write('#define BITNETMCU_MODEL_H\n\n')

        # Resolution Macros
        f.write(f'// Image Resolution Definitions\n')
        f.write(f'#define MODEL_INPUT_DIM {image_resolution}\n')
        f.write(f'#define MODEL_INPUT_LEN ({image_resolution} * {image_resolution})\n\n')

        f.write(f'#define NUM_LAYERS {len(quantized_model.quantized_model)}\n')
        f.write(f'#define MAX_N_ACTIVATIONS {max_n_activations}\n\n')

        for layer_info in quantized_model.quantized_model:
            layer = f'L{layer_info["layer_order"]}'
            incoming = layer_info['incoming_weights']
            outgoing = layer_info['outgoing_weights']
            bpw = layer_info['bpw']
            weights = np.array(layer_info['quantized_weights'])
            q_type = layer_info['quantization_type']

            if (bpw * incoming % 32) != 0:
                raise ValueError(f"Size mismatch: Incoming weights must be packed to 32bit boundary.")

            data_type = np.uint32
            
            # Logic to encode weights based on quantization type
            if q_type == 'Binary':
                encoded_weights = np.where(weights == -1, 0, 1)
                QuantID = 1
            elif q_type == '2bitsym': 
                encoded_weights = ((weights < 0).astype(data_type) << 1) | (np.floor(np.abs(weights))).astype(data_type)
                QuantID = 2
            elif q_type == '4bitsym': 
                encoded_weights = ((weights < 0).astype(data_type) << 3) | (np.floor(np.abs(weights))).astype(data_type)
                QuantID = 4
            elif q_type == '4bit': 
                encoded_weights = np.floor(weights).astype(int) & 15
                QuantID = 12
            else:
                print(f'Warning: Quantization type {q_type} logic might be missing.')
                encoded_weights = np.zeros_like(weights) # Fallback
                QuantID = 0

            # Pack bits into 32 bit words
            weight_per_word = 32 // bpw 
            reshaped_array = encoded_weights.reshape(-1, weight_per_word)
            bit_positions = 32 - bpw - np.arange(weight_per_word, dtype=data_type) * bpw
            packed_weights = np.bitwise_or.reduce(reshaped_array << bit_positions, axis=1).view(data_type)
            
            # Write macros
            f.write(f'// Layer: {layer} ({q_type})\n')
            f.write(f'#define {layer}_active\n')
            f.write(f'#define {layer}_bitperweight {QuantID}\n')
            f.write(f'#define {layer}_incoming_weights {incoming}\n')
            f.write(f'#define {layer}_outgoing_weights {outgoing}\n')

            # Write Array
            f.write(f'const uint32_t {layer}_weights[] = {{')         
            for i, data in enumerate(packed_weights.flatten()):
                if i & 7 == 0: f.write('\n\t')
                f.write(f'0x{data:08x},')
            f.write('\n};\n\n')
           
        f.write('#endif\n')

if __name__ == '__main__':
    paramname = 'trainingparameters.yaml'
    print(f'Loading parameters from: {paramname}')
    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runname = create_run_name(hyperparameters)
    
    # Get Resolution
    img_res = hyperparameters.get("image_resolution", 16)

    # Setup Data (for testing)
    transform = transforms.Compose([
        transforms.Resize((img_res, img_res)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST(root='data', train=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=hyperparameters["batch_size"], shuffle=False)

    # Load Model
    model = FCMNIST(
        network_width1=hyperparameters["network_width1"], 
        network_width2=hyperparameters["network_width2"], 
        network_width3=hyperparameters["network_width3"], 
        image_res=img_res,
        QuantType=hyperparameters["QuantType"], 
        NormType=hyperparameters["NormType"],
        WScale=hyperparameters["WScale"],
        quantscale=hyperparameters["quantscale"]
    ).to(device)

    model_path = f'modeldata/{runname}.pth'
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model: {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Run training.py first.")
        sys.exit()

    # Quantize
    print('Quantizing model...')
    quantized_model = QuantizedModel(model, quantscale=hyperparameters["quantscale"])
    print(f'Total model size: {quantized_model.totalbits()/8/1024:.2f} kbytes')

    # Check Accuracy (Optional but recommended)
    print("Checking accuracy of quantized model...")
    total_correct = 0
    total_samples = 0
    for input_data, labels in test_loader:
        input_data_np = input_data.view(input_data.size(0), -1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        result = quantized_model.inference_quantized(input_data_np)
        predict = np.argmax(result, axis=1)
        total_correct += (predict == labels_np).sum()
        total_samples += input_data_np.shape[0]

    print(f'Quantized Model Test Accuracy: {total_correct / total_samples * 100:.2f}%') 

    # Export
    export_to_hfile(quantized_model, 'BitNetMCU_model.h', runname, image_resolution=img_res)
    print("Done.")