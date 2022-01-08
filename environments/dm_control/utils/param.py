import toml
import numpy as np

def overload_par_dict(par_dict, **kwargs):
    for key, value in kwargs.items():
        if key in par_dict.keys():
            par_dict[key] = np.asarray(value) if type(value) == list else value

def write_config_file(default_par_list, par_dict, path, filename):
    with open(path + filename + ".toml", 'w') as file:
        for x in default_par_list:
            file.write("#" + x.description + "\n")
            toml.dump({x.name: par_dict[x.name]}, file)
            file.write("\n")

def get_par_from_config_file(path):
    with open(path) as file:
        return toml.load(file)

def set_par_from_config_file(par_dict, path):
    overload_par_dict(par_dict, **get_par_from_config_file(path))