# Notes about data parsing and preparation

The "dump_dispersion_constants.py" script under tools is used to dump pickle
files with the constants needed for dispersion D3 calculations.

The pickle files are loaded directly by the DispersionD3 module through the
"constants" module, which parses the raw pickle lists into manageable
structures.

The actual calculation of the dispersion interaction is handled in the
dispersion module.
