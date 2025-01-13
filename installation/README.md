# Installation tips

Below are some steps to keep in mind when installing OPM Flow from source, with Python bindings, using a Python virtual environment on Linux, without sudo rights. These mainly serve as reminders when I mess up my installation again. Some of these flags are probably superfluous.

## Use a Python virtual environment

I used `venv` to create a virtual environment in `/{prefix}/.venv/`. Make sure to install `pybind11` using

    pip install pybind11

Make sure to install `numpy` and `scipy` similarly.

## Install Dune
Install instructions can be found here https://www.dune-project.org/installation/installation-buildsrc/.

I found it handy to use the following bash script in the `dune` folder to bump the version, when needed:

    #!/bin/bash
    for repo in ./*/
    do
        repo=${repo%*/}
        cd $repo
        git checkout v2.9.0
        cd ..
    done

    ./dune-common/bin/dunecontrol all


## Install opm flow

A full set of instructions can be found here:
https://opm-project.org/?page_id=231

Nevertheless, the following flags are important to remember. On this page, I use the syntax 
that can directly be entered in a `settings.json` file in VSCode as

    "cmake.configureSettings": {
        "FLAG": value,
    }

The flags can be checked by using `ccmake .` in the `build` folder.

### Flags for finding Dune

If you install Dune from source, we need to tell OPM where to find dune-common by setting the flag

    "dune-common_DIR": "/{prefix}/dune/dune-common/build-cmake",

or more generally for each Dune module:

    "dune-{module}_DIR": "/{prefix}/dune/dune-{module}/build-cmake"

### Flags for the Python bindings

First of all, follow the instructions from
https://github.com/OPM/opm-simulators/tree/master/python
and 
https://opm.github.io/opm-python-documentation/master/flow-in-python.html.
It mainly boils down to setting the flags

    "OPM_ENABLE_PYTHON": "ON",
    "OPM_ENABLE_EMBEDDED_PYTHON": "ON",
    "OPM_INSTALL_PYTHON": "ON",

If you don't have `sudo` rights, we need to tell cmake to install somewhere else

    "CMAKE_INSTALL_PREFIX": "/{opm-folder}",

Additionally, we need to tell cmake where the `pybind11` folders are. I don't know if all of the following are necessary, but I managed to get it running with these flags:

    "Python3_EXECUTABLE": "/{prefix}/.venv/bin/python",
    "PYTHON_INSTALL_PREFIX": "/{prefix}/.venv/lib/python3.10/site-packages",
    "pybind11_DIR": "/{prefix}/.venv/lib/python3.10/site-packages/pybind11/share/cmake/pybind11",
    "PYBIND11_CMAKECONFIG_INSTALL_DIR": "/{prefix}/.venv/lib/python3.10/site-packages/pybind11/share/cmake/pybind11"

## Remember to make install

If you've made changes in opm flow, then the compiled files need to be copied to the correct Python folders by running `make install`. To avoid rebuilding the entire project, we include the flag

    "CMAKE_SKIP_INSTALL_ALL_DEPENDENCY": true,

and instead of compiling `all`, we can get away with recompiling only the necessary components. Usually, `simulators` in `opm-simulators/build/python/opm/simulators/` is sufficient.

Remember to run `make install` on `opm-common`, `opm-grids`, and `opm-simulators`. Otherwise there may be seg-faults.

## Install opmcpg to handle corner point grids

The package `opmcpg` 
(https://pypi.org/project/opmcpg/)
is essential 
and can be installed using 

    pip install opmcpq

## The TPSA forks of opm-common and opm-simulators

### PYACTION does not recognize keyword "SOURCE"

You should add the needed keyword to the list of `valid_keywords` in the following file
[opm-common/opm/input/eclipse/Schedule/Action/PyAction.cpp](https://github.com/OPM/opm-common/blob/53af14efb2e86bacaa89349a349066b2332e592e/opm/input/eclipse/Schedule/Action/PyAction.cpp#L40)

### The keyword ROCKBIOT

We define the solid pressure as
$$
p_s = \lambda(\nabla \cdot u) - \alpha p_f
$$
which rewrites the mass conservation law
$$
\nabla \cdot q + \partial_t (\alpha \lambda^{-1} p_s + (\eta + \alpha^2 \lambda ^{-1})p_f) = f
$$
In a fixed-stress splitting scheme, we lag the solid pressure by one iteration, thus moving the term with $p_s$ to the right-hand side. The remaining equation then has the same structure as the original mass conservation eqaution with a modified storativity term:
$$
\eta \rightarrow \eta + \alpha^2 \lambda ^{-1}
$$
This "additional compressibility" is implemented by including a new keyword `ROCKBIOT` in opm-common and opm-simulators.
