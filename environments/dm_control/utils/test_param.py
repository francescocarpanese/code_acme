"""Test utils for parameters serializing/storing/reading"""
from collections import namedtuple
import os
from environments.dm_control.utils import param

def test_overloading():
    """Test overloading of value in parameter dictionary"""
    dict_default = {'first': 1, 'second': [1.,2.]}
    dict_new = {'second': [2.,2.]}
    param.overload_par_dict(dict_default, **dict_new)
    assert all( dict_default['second'] == dict_new['second'])

def test_config_file(tmpdir):
    """ Test parameter configuration file handling"""
    # Generate parameters with default value and descriptions
    par = namedtuple('par', 'name value description')
    default_par_list = [
        par('a',1.,'test float'),
        par('b',[0.,0.],'test list')
    ]

    # Generate dictionary from parameter list
    par_dict = {x.name:x.value for x in default_par_list}

    # Write parameters into toml configuration file
    fname = "tmp"
    path = str(tmpdir)
    full_path = path + fname + ".toml"
    param.write_config_file(default_par_list, par_dict, path, fname)
    assert os.path.exists(full_path), 'Configuration file not written'

    # Load parameters from config file
    par_from_file = param.get_par_from_config_file(full_path)

    # Check read from file is the same as written data serialized into file file
    for key in par_from_file:
        assert par_from_file[key] == par_dict[key],\
             f'Parameter {key} different when read from file'
