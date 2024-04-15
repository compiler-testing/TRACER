# -*- coding: utf-8 -*-
# import sys
# sys.path.append(r"/../Documents/MLIR/MLIR_code")


# v0.10.1及以下
import os
import subprocess
import sys
import pymysql
import signal  

sys.path.append('../')
from utils import *
from utils.logger_tool import log
from utils.config import Config
from fuzz.fuzz import *

def seedAnalysis(config, target_file):
    IRanalysis(target_file, config)
    with open(config.analysis_seed_file, 'r', encoding='utf8') as temp:
        print(config.analysis_seed_file)
        json_data = json.load(temp)
        dialect_list = " "
        lowerPass_list = " "
        operations_list = " "
        if int(json_data["DialectNum"]) != 0:
            dialect_list = json_data["dialect"]
        if int(json_data["LowerPassNum"]) != 0:
            lowerPass_list = json_data["LowerPass"]
        if int(json_data["LowerPassNum"]) != 0:
            operations_list = json_data["operation"]

        print("operation",operations_list)

    

    dialects = ','.join(dialect_list)
    lowerPasses = " ".join(lowerPass_list)
    operations = ','.join(operations_list)
    return dialects,lowerPasses,operations


def generate_user_cases(config: Config, seeds_count, mode):
    count=0;
    i = 0
    start = datetime.datetime.now()
    st= start.timestamp()
        
    while(count< seeds_count):
        log.info("======generate seed : " + str(count))
        target_file = config.temp_dir + str(count) + '.mlir'
        genrateOpt = "-tosaGen"
        if mode=="api":
            genrateOpt = "-tosaGenU"
        elif mode=="chain":
            genrateOpt = "-tosaGenC"

        cmd = '%s %s %s -o %s' % (config.mlirfuzzer_opt, config.empty_func_file, genrateOpt,target_file)

        pro = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True, encoding="utf-8",preexec_fn=os.setsid)
        try:
            i = i+1
            stdout, stderr = pro.communicate(timeout=5)
            returnCode = pro.returncode
            log.info(cmd)
            if not os.path.exists(target_file):

                log.error(stderr)
                continue
            dialects,candidate_lower_pass,operations = seedAnalysis(config,target_file)
          
            log.info(candidate_lower_pass)
            log.info(operations)
            with open(target_file, 'r') as f:
                content = f.read()
            log.info(len(content))
            if len(content)>10:
                try:
                    sql = "insert into "+ config.seed_pool_table + \
                          " (preid,source,mtype,dialect,operation,content,n, candidate_lower_pass) " \
                          "values ('%s','%s','%s','%s','%s','%s','%s','%s')" \
                          % \
                          (0,'G','','tosa', operations,content, 0, candidate_lower_pass)
                    dbutils.db.executeSQL(sql)
                    count = count+ 1
                except Exception as e:
                    log.error('sql error', e)
                os.remove(target_file)
            else:
               log.info("Insufficient seed length")
        except subprocess.TimeoutExpired:
            # pro.kill()
            os.killpg(pro.pid,signal.SIGTERM) 
            stdout = ""
            stderr = "timeout, kill this process"
            returnCode = -9

    print(count/i)
    now = datetime.datetime.now()
    nw= now.timestamp()
    print(nw-st)

def create_new_table(conf: Config):
    with open(conf.project_path + '/fuzz_tool/conf/init.sql', 'r',encoding="utf-8") as f:
        sql = f.read().replace('seed_pool_table', conf.seed_pool_table) \
            .replace('result_table', conf.result_table) \
            .replace('report_table', conf.report_table)
    try:
        dbutils.db.connect_mysql()
        sql_list = sql.split(';')[:-1]
        for item in sql_list:
            dbutils.db.cursor.execute(item)
        dbutils.db.db.commit()
        print("database init success!")
    except pymysql.Error as e:
        print("SQL ERROR:=======>", e)
    finally:
        dbutils.db.close()