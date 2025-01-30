#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:41:50 2025

@author: agastraa
"""

import psycopg2
from psycopg2 import sql

# Global/persistent variables to manage connections
exp_conn = None
exp_res = None
exp_num = 0

fl_conn = None
fl_res = None
fl_num = 0
var_num = 0

out_conn = None
out_res = None

con = [None] * 40
res = [None] * 40
num_threads = 0

# Routines to push data directly to SQL table (single thread)
def open_outsql(server, db):
    global out_conn
    conn_str = f"host={server} dbname={db}"
    out_conn = psycopg2.connect(conn_str)
    print(f"Opened {db} on {server} for output")
    return

def push_to_sql(table, outdata):
    global out_res
    cursor = out_conn.cursor()
    insert_query = sql.SQL("INSERT INTO {table} VALUES ({values});").format(
        table=sql.Identifier(table),
        values=sql.SQL(outdata)
    )
    cursor.execute(insert_query)
    out_conn.commit()
    cursor.close()

def cleanup_outsql():
    global out_conn, out_res
    if out_res:
        out_res.close()
    if out_conn:
        out_conn.close()
    print("Closed Output SQL Connection.")
    return

def initialize_new_table(table):
    global out_res
    cursor = out_conn.cursor()
    create_table_query = f"""
    DROP TABLE IF EXISTS {table};
    CREATE UNLOGGED TABLE {table} (
        var TEXT,
        initialt TIMESTAMP,
        validt TIMESTAMP,
        expid TEXT,
        valid BOOLEAN,
        value DOUBLE PRECISION
    );
    """
    cursor.execute(create_table_query)
    out_conn.commit()
    cursor.close()

def send_sql_command(sqlcmd):
    global out_res
    cursor = out_conn.cursor()
    cursor.execute(sqlcmd)
    out_conn.commit()
    cursor.close()

# Routines to push data directly to SQL table (multi-thread)
def open_outsql_multi(server, db, nthreads):
    global con, num_threads
    if nthreads > 40:
        nthreads = 40
    num_threads = nthreads
    conn_str = f"host={server} dbname={db}"

    for i in range(num_threads):
        con[i] = psycopg2.connect(conn_str)
        print(f"Opened connection {i} to {db} on {server} for output")

def push_to_sql_multi(table, outdata, thread):
    global res
    cursor = con[thread].cursor()
    insert_query = sql.SQL("INSERT INTO {table} VALUES ({values});").format(
        table=sql.Identifier(table),
        values=sql.SQL(outdata)
    )
    cursor.execute(insert_query)
    con[thread].commit()
    res[thread] = cursor
    cursor.close()

def clear_outsql_res():
    global res
    for i in range(num_threads):
        if res[i]:
            res[i].close()
    return

def cleanup_outsql_multi():
    global con, res
    for i in range(num_threads):
        if res[i]:
            res[i].close()
        if con[i]:
            con[i].close()
    print("Closed Output SQL Connections.")

def send_sql_command_multi(sqlcmd):
    global res
    cursor = con[0].cursor()
    cursor.execute(sqlcmd)
    con[0].commit()
    res[0] = cursor
    cursor.close()

# Routines to get file list, query SDS metadata
def open_flist(server, db, table, sub):
    global fl_conn, fl_res, fl_num
    conn_str = f"host={server} dbname={db}"
    fl_conn = psycopg2.connect(conn_str)
    print(f"Opened {db} on {server}")

    cursor = fl_conn.cursor()
    query = f"SELECT netcdf_path FROM {table} {sub};"
    cursor.execute(query)
    fl_res = cursor.fetchall()
    fl_num = 0
    cursor.close()
    return len(fl_res)

def get_sds_varname(stdn):
    global fl_res
    cursor = fl_conn.cursor()
    query = f"SELECT orig_name FROM sds_vars WHERE std_name = %s;"
    cursor.execute(query, (stdn,))
    result = cursor.fetchone()
    cursor.close()
    return result[0] if result else "BAD NAME"

def get_all_sds_varname():
    global fl_res, var_num
    cursor = fl_conn.cursor()
    query = "SELECT std_name, orig_name FROM sds_vars ORDER BY std_name;"
    cursor.execute(query)
    fl_res = cursor.fetchall()
    var_num = 0
    cursor.close()
    return len(fl_res)

def get_next_sds_var():
    global fl_res, var_num
    if var_num >= len(fl_res):
        return None, None
    stdnam, ncnam = fl_res[var_num]
    var_num += 1
    return stdnam, ncnam

def fetch_fname_in_sequence():
    global fl_res, fl_num
    if fl_num >= len(fl_res):
        return None
    fname = fl_res[fl_num][0]
    fl_num += 1
    return fname

def cleanup_flist():
    global fl_conn, fl_res
    if fl_res:
        fl_res.close()
    if fl_conn:
        fl_conn.close()
    print("Closed SDS Connection.")
    return

# Exposure fetching routines
def open_exposure(server, db, table, id_col, lat_col, lon_col, sub):
    global exp_conn, exp_res, exp_num
    conn_str = f"host={server} dbname={db}"
    exp_conn = psycopg2.connect(conn_str)
    print(f"Opened {db} on {server}")

    cursor = exp_conn.cursor()
    query = f"SELECT {id_col}, {lat_col}, {lon_col} FROM {table} {sub};"
    cursor.execute(query)
    exp_res = cursor.fetchall()
    exp_num = 0
    cursor.close()
    return len(exp_res)

def fetch_exposure_in_sequence():
    global exp_res, exp_num
    if exp_num >= len(exp_res):
        return None
    exposure = exp_res[exp_num]
    exp_num += 1
    return exposure

def fetch_results():
    global out_res, exp_num
    result = out_res[exp_num]
    return ",".join(map(str, result))

def cleanup_exposure():
    global exp_conn, exp_res
    if exp_res:
        exp_res.close()
    if exp_conn:
        exp_conn.close()
    print("Closed Exposure Connection.")
    return