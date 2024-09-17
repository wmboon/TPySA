# Installation tips

## Install opm flow

### Flags for Dune

### Flags for Python bindings
https://github.com/OPM/opm-simulators/tree/master/python

### Install opmcpg
https://pypi.org/project/opmcpg/

## More problems

### PYACTION does not recognize keyword "SOURCE"

Simply add the needed keyword to the list of valid_keywords in the following file
[opm-common/opm/input/eclipse/Schedule/Action/PyAction.cpp](https://github.com/OPM/opm-common/blob/53af14efb2e86bacaa89349a349066b2332e592e/opm/input/eclipse/Schedule/Action/PyAction.cpp#L40)
