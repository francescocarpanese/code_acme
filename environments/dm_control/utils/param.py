"""Utils for parameters serelization/storing/reading"""
import toml
import numpy as np

def overload_par_dict(par_dict, **kwargs):
    """Overloads value in parameter dictionary from inputs"""
    for key, value in kwargs.items():
        if key in par_dict.keys():
            par_dict[key] = (
                np.asarray(value) if isinstance(value,list)
                else value
                )

def write_config_file(default_par_list, par_dict, path, filename):
    """Writes toml file from parameter dictionary"""
    with open(path + filename + ".toml", 'w', encoding='UTF-8') as file:
        for parameter in default_par_list:
            file.write("#" + parameter.description + "\n")
            toml.dump({parameter.name: par_dict[parameter.name]}, file)
            file.write("\n")

def get_par_from_config_file(path):
    """Returns parameter from toml file"""
    with open(path, encoding='UTF-8') as file:
        return toml.load(file)

def set_par_from_config_file(par_dict, path):
    """Reads parameters from parameter file and overload parameter dictionary"""
    overload_par_dict(par_dict, **get_par_from_config_file(path))
