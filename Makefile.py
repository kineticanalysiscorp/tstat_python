#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:57:38 2025

@author: agastraa
"""

import os
import subprocess

OBJS = ["tstat"]

TAOS_SRC = os.getenv("TAOS_SRC")
TAOS_FLIBS = os.getenv("TAOS_FLIBS")
CARGS = os.getenv("CARGS", "")
NCMOD = os.getenv("NCMOD", "")
NCARGS = os.getenv("NCARGS", "")

CARGL = f"{CARGS} -fbounds-check"
PGINC = subprocess.getoutput("pg_config --includedir")
PGLIB = subprocess.getoutput("pg_config --libdir")

def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error running command: {command}\n{result.stderr.decode()}")
    return result

def build_tstat():
    # Compile lognorm.for
    run_command(f"{os.getenv('FC')} -c {CARGL} {TAOS_FLIBS}/lognorm.for")
    
    # Compile sql_subs.c
    run_command(f"gcc -O2 -c -I{PGINC} sql_subs.c")
    
    # Link and create the tstat executable
    command = (
        f"{os.getenv('FC')} {CARGL} -o tstat "
        f"{TAOS_FLIBS}/parsecsv_module.f90 "
        f"{TAOS_FLIBS}/dtg_module.f90 "
        f"{TAOS_FLIBS}/grid_module.f90 "
        f"{TAOS_FLIBS}/asset_module.f90 "
        f"{TAOS_FLIBS}/damage_module.f90 "
        f"{TAOS_FLIBS}/damage_module_code.f90 "
        f"{TAOS_FLIBS}/asset_module_code.f90 "
        f"{TAOS_FLIBS}/statistics_module.f90 "
        f"{NCMOD} "
        f"lognorm.o global.f90 ncsubs.f90 fit_data.f90 solve.f90 main.f90 sql_subs.o {NCARGS} {PGLIB}/libpq.so.5"
    )
    run_command(command)

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python build.py [target]")
        sys.exit(1)

    target = sys.argv[1]

    if target == "tstat":
        build_tstat()
    else:
        print(f"Unknown target: {target}")
        sys.exit(1)

if __name__ == "__main__":
    main()