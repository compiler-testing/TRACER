# -*- coding: utf-8 -*-

import argparse
import sys
import time
import datetime
from utils.config import Config
from utils import *

import os
from utils.logger_tool import log
reported_errors = []

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--opt', required=True,
                            choices=['generator', 'load','fuzz'])
    arg_parser.add_argument('--sqlName',required=True)
    arg_parser.add_argument('--mode',default='multi-branch',choices=['api', 'chain', 'multi-branch'])
    arg_parser.add_argument('--Mut',default='0',choices=['0', '1', '2', '3'])  #no mix rep mut
    arg_parser.add_argument('--DT',default='dt',choices=['r', 'rr','c','dt'])
    arg_parser.add_argument('--debug',default='0',choices=['0', '1'])
    arg_parser.add_argument('--path',required=True)
    return arg_parser.parse_args(sys.argv[1:])

def load_cases(config: Config,folder_path):
    sys.path.append('../')

    mlir_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mlir')]  
    count=0;
    
    from generator.tosaGen import seedAnalysis

    for file in mlir_files:
        dialects,candidate_lower_pass,operations = seedAnalysis(config,file)
 
        print("loading", file)
        with open(file, 'r') as f:
            content = f.read()
        
        if len(content)>3000000:
            continue
        try:
            sql = "insert into "+ config.seed_pool_table + \
                    " (preid,source,mtype,dialect,operation,content,n, candidate_lower_pass) " \
                    "values ('%s','%s','%s','%s','%s','%s','%s','%s')" \
                    % \
                    (0,'G','',dialects,operations, content, 0, candidate_lower_pass)
            # print(sql)
            dbutils.db.executeSQL(sql)
            count = count+ 1
        except Exception as e:
            log.error('sql error', e)
        if count==config.count:
            break;
    return

import re
def load_cases_sql(config: Config):
    target_sql = 'test'

    if 'LE' in config.seed_pool_table:
        target_sql = "seed_pool_lemon_copy"
    elif 'MU' in config.seed_pool_table:
        target_sql = "seed_pool_muffin__copy"
    elif 'NN' in config.seed_pool_table:
        target_sql = "seed_pool_nnsmith_copy"
    elif 'MS' in config.seed_pool_table:
        target_sql = "seed_pool_MS1_copy"
    elif 'MF' in config.seed_pool_table:
        target_sql = "seed_pool_mlirfuzzer_copy"
    elif 'NE' in config.seed_pool_table:
        target_sql = "seed_pool_NERR1_copy1"
        

    sql = "select * FROM " + target_sql+ " where source = 'G' and operation != ' ' "  
    dataList = dbutils.db.queryAll(sql)

    count = 0
    for data in dataList:
        try:
            sql = "insert into " + config.seed_pool_table +  \
                    " (preid,source,mtype,dialect,operation,content,n, candidate_lower_pass) " \
                    "values ('%s','%s','%s','%s','%s','%s','%s','%s')" \
                    % \
                    (0,'G','',data[-5],data[-4], data[-3], 0, data[-1])
            dbutils.db.executeSQL(sql)
            count = count+1
            print("Loading  ", count)
        except Exception as e:
            log.error('sql error', e)
        if count==config.count:
            break;
    
    return


def main():
    args = get_args()  
    #==================need to modify with your path=============#
    config_path = '/home/ty/compiler-testing/fuzz_tool/conf/conf.yml'
    # config_path = './conf/conf.yml'   # Run
    #============================================================#
    
    conf = Config(config_path,args.sqlName,args.path)
    
    logger_tool.get_logger()
    dbutils.db = dbutils.myDB(conf)
    if args.opt == 'generator':
        from generator.tosaGen import generate_user_cases,create_new_table
        # initialize database
        create_new_table(conf)
        # generate tosa graphs
        generate_user_cases(conf, conf.count,args.mode)

    if args.opt == 'load':
        from generator.tosaGen import create_new_table
        # initialize database
        create_new_table(conf)
        load_cases_sql(conf)



    elif args.opt == 'fuzz':
        from fuzz.fuzz import Fuzz
        fuzzer = Fuzz(conf)
        start = datetime.datetime.now()
        end = start + datetime.timedelta(minutes=conf.run_time)
        st= start.timestamp()
        nt = st
        while (nt-st<43200):
            now = datetime.datetime.now()
            nt= now.timestamp()
            print(datetime.datetime.now())
            conf.Iter +=1    
            if args.debug != '0':
                fuzzer.debug()
                break
            fuzzer.process(args.Mut,args.DT)
            if now.__gt__(end):
                break
        print("time out!!!")



# RUN : python main.py --opt=fuzz
if __name__ == '__main__':
    main()
