import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tvm
from tvm import relay
from tvm.runtime import vm as vmexec

# Load pre-trained densenet-161 model from PyTorch
pytorch_model = models.densenet161(pretrained=True)
# pytorch_model = models.resnet152(pretrained=True)
pytorch_model = pytorch_model.eval()
# Load the image
image = Image.open('ostrich.jpg')

# Define the transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply the transformations to the image and add a batch dimension
input_data = transform(image).unsqueeze(0)

# Convert the PyTorch model to a TVM Relay computation graph
input_name = "input0"
input_shape = list(input_data.shape)
shape_list = [(input_name, input_shape)]
scripted_model = torch.jit.trace(pytorch_model, input_data).eval()
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# Perform the inference using TVM
target = 'llvm'
with tvm.transform.PassContext(opt_level=3):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

# Create a TVM runtime and load the compiled module
ctx = tvm.cpu()
vm = vmexec.VirtualMachine(vm_exec, ctx)

# Set the input parameters for the TVM module
for name, value in params.items():
    vm.set_input(name, value)#errors out "InternalError: Check failed: (it != global_map.end()) is false: Cannot find function aten::linear_0.bias in executable"
input_data_tvm = tvm.nd.array(input_data.numpy(), ctx)
vm.set_input("main", **{input_name: input_data_tvm})

# Run the module
tvm_output = vm.run()

# Convert the numpy array to a PyTorch tensor
tvm_output_torch = torch.from_numpy(tvm_output.asnumpy())

# Load the labels
with open('imagenet-classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# Get the index of the max log-probability
_, predicted_class = torch.max(tvm_output_torch, 1)

print('Predicted Class: ', labels[predicted_class.item()])
