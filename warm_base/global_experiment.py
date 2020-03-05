import json
import os
import subprocess
import torch
import numpy as np

with open("basic_config.json", "r") as f:
    base_config = json.load(f)
from collections import defaultdict
from allennlp.nn import util


def execute(command):
    p = subprocess.Popen(command.split())
    p.wait()


# We delete the previous copies and logs
execute("rm -rf " + os.getcwd() + "/copies/*")
execute("rm -rf " + os.getcwd() + "/logs/*")
# Here is the list of all classes
classes = [
    "DATE",
    "ORG",
    "GPE",
    "EVENT",
    "LOC",
    "FAC",
    "CARDINAL",
    "QUANTITY",
    "NORP",
    "ORDINAL",
    "WORK_OF_ART",
    "PERSON",
    "LANGUAGE",
    "LAW",
    "MONEY",
    "PERCENT",
    "PRODUCT",
    "TIME",
]

CUDA_DEVICE = 0 # CUDA device for each of the 18 classes above, -1 for CPU

for random_seed in range(1, 3):
    base_config["dataset_reader"]["random_seed"] = random_seed
    num_classes_per_exp = 2 # Run 2 experiments for a class at a time
    for class_start_idx in range(0, len(classes), num_classes_per_exp):
        processes = []
        for exp_class_idx in range(num_classes_per_exp):
            class_idx = class_start_idx + exp_class_idx
            exp_class = classes[class_idx]
            # Here we edit the config for a particular experiment
            base_config["dataset_reader"]["valid_class"] = exp_class
            base_config["dataset_reader"]["drop_empty"] = False
            base_config["trainer"]["cuda_device"] = CUDA_DEVICE
            this_dir = os.getcwd().split("/")[-1]

            copy_directory = (
                os.getcwd()[: -(len(this_dir) + 1)]
                + "/copies/"
                + this_dir
                + "/pnet_"
                + exp_class
                + "_"
                + str(random_seed)
            )

            if not os.path.exists(copy_directory):
                os.makedirs(copy_directory)

            # We create a copy of all the code and run it in this directorty.
            execute("rm -rf " + copy_directory)
            execute("cp -r " + os.getcwd() + "/base " + copy_directory)

            model_directory = (
                os.getcwd()[: -(len(this_dir) + 1)]
                + "/models/"
                + this_dir
                + "/pnet_"
                + exp_class
                + "_"
                + str(random_seed)
            )

            if not os.path.exists(model_directory):
                os.makedirs(model_directory)

            # Here we save our config
            with open(copy_directory + "/config.json", "w") as outfile:
                json.dump(base_config, outfile)

            # Here we substitute our config instead of the old one
            cmd = "rm -f " + model_directory + "/config.json"
            p = subprocess.Popen(cmd.split())
            p.wait()

            with open(model_directory + "/config.json", "w") as outfile:
                json.dump(base_config, outfile)

            # Delete old logs (if there are some)
            execute("rm -f " + model_directory + "/stdout.log")
            execute("rm -f " + model_directory + "/stderr.log")

            # We need to edit the model after warming to load it

            try:
                # if we have done it already we would have loaded this flag successfully
                flag = np.load(model_directory + "/flag.npy")
            except:
                # We need to change the content of saved dictionary in the epoch we load from (the last one).
                for i_epoch in range(4, 5):
                    dic = torch.load(
                        model_directory + "/training_state_epoch_{}.th".format(i_epoch),
                        map_location=util.device_mapping(-1),
                    )
                    p_keys = dic["optimizer"]["state"].keys()
                    for p_key in p_keys:
                        dic["optimizer"]["state"] = defaultdict(dict)
                    torch.save(
                        dic,
                        model_directory + "/training_state_epoch_{}.th".format(i_epoch),
                    )
                np.save(model_directory + "/flag.npy", np.array([0]))

            # Then we run our experiment
            cmd = "python3 " + copy_directory + "/my_run.py train "
            cmd += copy_directory + "/config.json -s "
            cmd += model_directory + " --recover"
            p = subprocess.Popen(cmd.split())
            processes.append(p)

        for i, process in enumerate(processes):
            process.wait()
