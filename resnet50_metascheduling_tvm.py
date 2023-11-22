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
input_image_shape = (224,224,3)
data_shape = (batch_size,) + input_image_shape
output_shape = (batch_size, num_class)
dtype = "float16"

ARGS = _parse_args()
mod, params = testing.resnet.get_workload(
    num_layers=50,batch_size = batch_size, image_shape=input_image_shape,layout="NHWC",dtype="float16")

opt_level = 3
work_dir = "/home/ryb2kor/tvm/quantization_task/scripts/tf_model_quant-master"
tgt = tvm.target.Target("cuda")
dev = tvm.device(str("cuda"), 0)
target = tvm.target.cuda(arch='sm_86')

tune_tasks = ms.relay_integration.extract_tasks(mod, ARGS.target, params)
tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
    extracted_tasks=tune_tasks,
    work_dir=work_dir,
    space=ms.space_generator.PostOrderApply(
            sch_rules="cuda-tensorcore", postprocs="cuda-tensorcore", mutator_probs="cuda-tensorcore")
)
database = ms.tune.tune_tasks(
    tasks=tasks,
    task_weights=task_weights,
    work_dir=work_dir,
    max_trials_per_task=4,
    max_trials_global=150,
)
with database, tvm.transform.PassContext(
    opt_level=3,
    config={"relay.backend.use_meta_schedule": True},
):
    lib = relay.build(mod, target=target, params=params)


current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
str_current_datetime = str(current_datetime)

path = os.path.join("/home/ryb2kor/" + str_current_datetime + "resnet_imagenet_metacheuled.so")
lib.export_library(path)

a_np = np.random.randint(0, 255, size=(1,224,224,3)).astype(dtype)
data = tvm.nd.array(a_np, dev)
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("data", data)
module.run()
timer = module.module.time_evaluator("run", dev, number=10, repeat=3)
print(timer())