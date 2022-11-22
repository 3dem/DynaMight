#!/usr/bin/env python

"""
Module for star-file I/O
"""

from collections import OrderedDict


def load_star(filename):
    datasets = OrderedDict()
    current_data = None
    current_colnames = None

    BASE = 0  # Not in a block
    COLNAME = 1  # Parsing column name
    DATA = 2  # Parsing data
    mode = BASE

    for line in open(filename):
        line = line.strip()
        
        # remove comments
        comment_pos = line.find('#')
        if comment_pos > 0:
            line = line[:comment_pos]

        if line == "":
            if mode == DATA:
                mode = BASE
            continue

        if line.startswith("data_"):
            mode = BASE
            data_name = line[5:]
            current_data = OrderedDict()
            datasets[data_name] = current_data

        elif line.startswith("loop_"):
            current_colnames = []
            mode = COLNAME

        elif line.startswith("_"):
            if mode == DATA:
                mode = BASE
            token = line[1:].split()
            if mode == COLNAME:
                current_colnames.append(token[0])
                current_data[token[0]] = []
            else:
                current_data[token[0]] = token[1]

        elif mode != BASE:
            mode = DATA
            token = line.split()
            if len(token) != len(current_colnames):
                raise RuntimeError(
                    f"Error in STAR file {filename}, number of elements in {token} "
                    f"does not match number of column names {current_colnames}"
                )
            for idx, e in enumerate(token):
                current_data[current_colnames[idx]].append(e)        
        
    return datasets
