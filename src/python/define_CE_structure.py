# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 16:07:20 2022

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt


def divide(grids, z, r, rs, zs):
    rp = np.interp(z, zs, rs)
    return np.where(r.reshape(1, -1) < rp.reshape(-1, 1), grids+1, grids)


def read_CE_structure(filename, time_str):
    def read_line(line):
        tmp = line.lstrip('r:').lstrip('z:').rstrip('\n').split(',')
        return np.array([i.lstrip(' ').rstrip(' ') for i in tmp], dtype='float')

    with open(filename) as f:
        content = f.readlines()

    select = []
    select_idx = 0
    for line in content:
        if select_idx == 1:
            if line != '\n':
                select.append(line)
            if '#####' in line:
                break
        if time_str in line:
            select_idx = 1

    inner_r = read_line(select[1])*1000
    inner_z = read_line(select[2])*1000
    moat_r = read_line(select[4])*1000
    moat_z = read_line(select[5])*1000
    outer_r = read_line(select[7])*1000
    outer_z = read_line(select[8])*1000

    return inner_r, inner_z, moat_r, moat_z, outer_r, outer_z


if __name__ == '__main__':
    inner_r, inner_z, moat_r, moat_z, outer_r, outer_z = \
        read_CE_structure('define_CE_structure.txt', '2019-08-08_14_05_00')

    r = np.arange(1000, 800000.1, 1000)
    z = np.arange(200, 23000.1, 400)

    grids = np.zeros([z.shape[0], r.shape[0]])

    grids = divide(grids, z, r, inner_r, inner_z)
    grids = divide(grids, z, r, moat_r, moat_z)
    grids = divide(grids, z, r, outer_r, outer_z)

    plt.contour(r, z, grids)
    plt.plot(inner_r, inner_z)
    plt.plot(moat_r, moat_z)
    plt.plot(outer_r, outer_z)
    plt.show()
