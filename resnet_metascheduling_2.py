import argparse
import json
import logging
from distutils.util import strtobool
import numpy as np
import tvm
import os
import pytest
from tvm.contrib import graph_executor
from tvm.relay import testing
from tvm import meta_schedule as ms
from tvm import relay, auto_scheduler
from tvm.meta_schedule.testing import relay_workload
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base
from tvm.tir.tensor_intrin import *
from datetime import datetime

def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--target",
        type = str,
        required=True
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    return parsed


batch_size = 1
num_class = 1000
input_image_shape = (3,224,224)
data_shape = (batch_size,) + input_image_shape
output_shape = (batch_size, num_class)
dtype = "float32"

ARGS = _parse_args()
mod, params = testing.resnet.get_workload(
   batch_size = batch_size, image_shape=input_image_shape,layout="NCHW",dtype="float32")

opt_level = 3
work_dir = "/home/ryb2kor/tvm/quantization_task/scripts/tf_model_quant-master"
tgt = tvm.target.Target("cuda")
dev = tvm.device(str("cuda"), 0)
target = tvm.target.cuda(arch='sm_86')

database = ms.relay_integration.tune_relay(
    mod=mod,
    target=ARGS.target,
    work_dir=work_dir,
    max_trials_global=64,
    num_trials_per_iter=1,
    params=params,
    space=ms.space_generator.PostOrderApply(
            sch_rules="cuda-tensorcore", postprocs="cuda-tensorcore", mutator_probs="cuda-tensorcore")
)

rt_mod1 = ms.relay_integration.compile_relay(
    database=database,
    mod=mod,
    target=ARGS.target,
    params=params,
)
print(rt_mod1)



current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
str_current_datetime = str(current_datetime)

path = os.path.join("/home/ryb2kor/" + str_current_datetime + "resnet_imagenet_metacheuled.so")
rt_mod1.export_library(path)

a_np = np.random.randint(0, 255, size=(1,3,224,224)).astype(dtype)
data = tvm.nd.array(a_np, dev)
module = graph_executor.GraphModule(rt_mod1["default"](dev))
module.set_input("data", data)
module.run()
timer = module.module.time_evaluator("run", dev, number=10, repeat=3)
print(timer())