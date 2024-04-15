# -*- coding: utf-8 -*-
import random
import datetime
import subprocess
import sys
import json
from time import time
import Levenshtein
from random import randint
import os
import numpy as np
import re
import signal  


sys.path.append('../')
from utils.config import Config
from utils import *
from utils.dbutils import myDB

from utils.logger_tool import log
from fuzz.pass_enum import *

def get_rand_pass_pipe(pass_list: list,N) -> str:
    maxlen = len(pass_list)
    if len(pass_list)>=N:
        maxlen = N

    count = random.randint(1, maxlen)
    sub_list = random.sample(pass_list, count)
    return sub_list

def get_all_pass_pipe(pass_list: list) -> str:
    pass_pipe = " ".join(pass_list)
    return pass_pipe

def IRanalysis(temp_file, config:Config):
    """
    分析mlir文本的方言和pass
    """
    cmd = '%s %s -allow-unregistered-dialect -GetDialectName ' % (config.mlirfuzzer_opt, temp_file)
    start_time = int(time() * 1000)

    pro = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, encoding="utf-8",preexec_fn=os.setsid)
    try:
        stdout, stderr = pro.communicate(timeout=30)
        return_code = pro.returncode
        # log.info(cmd)

    except subprocess.TimeoutExpired:
        # pro.kill()
        os.killpg(pro.pid,signal.SIGTERM) 
        stdout = ""
        stderr = "timeout, kill this process"
        return_code = -9


def getDiff(raw_mlir,processed_mlir):
    sim = Levenshtein.ratio(raw_mlir, processed_mlir)
    return 1-sim

def verifyIR(dialects,result):
    log.info("===Identification of invalid IR")
    """
    无效IR判定
    1.只有一个方言的情况，只有func、spv方言
    2.只有func和arith
    """
    vaild = True
    if (len(dialects) == 1):
        if "func" or "spv" or "llvm" in dialects:
            vaild = False
    elif (len(dialects) == 2):
        if "func" and "arith" in dialects:
            vaild = False
    if (vaild == False):
        log.info("=== invalid IR, report this bug")

    return vaild

def getFileContent(file):
    with open(file, 'r',encoding="utf-8") as f:
        content = f.read()
    return content


def save_mutate_seed(seed_file,temp_file,result, config: Config, flag):


    with open(seed_file, 'r',encoding="utf-8") as f:
        seed_mlir = f.read()
    
    with open(temp_file, 'r',encoding="utf-8") as f:
        processed_mlir = f.read()

    if seed_mlir==processed_mlir:
        print("diff = 0")
    else:
        with open(seed_file, 'w') as f:
            f.write(processed_mlir)




def analysis_and_save_seed(seed_file,temp_file,result, config: Config, flag):
    """
    计算变异前后种子差异
    如果相似度大于阈值，分析其方言和降级pass，将其存入数据库
    否则不保存
    计算优化前后种子差异
    如果相似度大于阈值，分析其方言和降级pass
    判断IR是否有效，如果IR无效，报告错误
    否则将其存入数据库
    """

    with open(seed_file, 'r',encoding="utf-8") as f:
        seed_mlir = f.read()
    
    with open(temp_file, 'r',encoding="utf-8") as f:
        processed_mlir = f.read()

    if seed_mlir==processed_mlir:
        print("diff = 0")
    else:
        IRanalysis(temp_file, config)
        from generator.tosaGen import seedAnalysis
        dialects,lowerPasses,operations = seedAnalysis(config,config.analysis_seed_file)
        if (dialects !=' '):
            lowerPasses = lowerPasses.replace('  ', ' ')
            fuzzer = DTFuzz(config)
            DTFuzz.mutate_success_handler(fuzzer, result["sid"], result["mutate_type"], dialects,operations, processed_mlir,
                                                lowerPasses,flag)


def setOptSeq(dia):
    list = []

    savefile = "/../MLIRFuzzing/bugpre/data/vulpass.npy"
    npy_data = np.load(savefile,allow_pickle=True)
    if dia=="affine":
        list.extend(get_rand_pass_pipe(npy_data.tolist()["Affine"],5))

    random.shuffle(list)

    passSeq = " ".join(list)
    pass_list = []
    pass_list.append(passSeq)

    return pass_list

def setLowerSeq(lower_pass):
    # log.info("======combine candidates and randPass")
    list = []
    candidates = get_rand_pass_pipe(str_to_list(lower_pass),3)
    randoms = get_rand_pass_pipe(LowerPass.all,3)

    for i, x in enumerate(candidates) :
        if x.find("linalg-to")!=-1:
            # print(candidates[i])
            candidates[i] = "-func-bufferize -linalg-bufferize " +candidates[i]

    for i, x in enumerate(randoms):
        if x.find("linalg-to") != -1:
            # print(randoms[i])
            randoms[i] = "-func-bufferize -linalg-bufferize " + randoms[i]

    for i, x in enumerate(randoms):
        if x.find("convert-parallel-loops-to-gpu") != -1:
            # print(randoms[i])
            randoms[i] = "-gpu-map-parallel-loops " + randoms[i]


    return " ".join(candidates)


def setOptSeqR():
    """
    1.启用脆弱性分析：优化序列：VP + 随机优化
    2.随机优化
    """
    list = []
    list.extend(get_rand_pass_pipe(OptimizePass.all,20))
    random.shuffle(list)
    passSeq = " ".join(list)
    # pass_list = []
    # pass_list.append(passSeq)

    return passSeq



def fixlowerpass(singlePass):
    newpass = []
    passlist = str_to_list(singlePass)
    if len(passlist) == 1:
        newpass.append(singlePass)
    else:
        passlist = list(filter(lambda x: x != '', passlist))
        locs = []
        for i, x in enumerate(passlist):
            if x.find("-pass-pipeline") >= 0:
                locs.append(i)
                
        if len(locs) == 0:
            newpass.append(singlePass)
        else:
            l = locs[0]
            if l == 0:
                newpass.append(passlist[0])
            else:
                newpass.append(' '.join(passlist[0:l]))
                newpass.append(passlist[l])
            if (len(locs) == 1 and l!=len(passlist)-1):
                newpass.append(' '.join(passlist[l + 1:]))
            else:
                for i in locs[1:]:
                    if (i - l > 1):
                        newpass.append(' '.join(passlist[l + 1:i]))
                    newpass.append(passlist[i])
                    if i == locs[-1]:
                        if i != len(passlist) - 1:
                            newpass.append(' '.join(passlist[i + 1:]))
                    l = i
    #log.info(newpass)
    return newpass



def execute_pass1(input_file,output_file, sid, raw_mlir, singlePass, config:Config,flag):

    if singlePass.find("func.func") >= 0:
        newpass = []
        newpass = fixlowerpass(singlePass)
        cmd_ = '%s %s -allow-unregistered-dialect %s' % (config.mlir_opt, input_file, newpass[0])
        cmd1 = cmd_
        if len(newpass)>1:
            for seg in newpass[1:]:
                cmd2 = '| %s %s' % (config.mlir_opt, seg)
                # cmd2 = '| %s %s %s' % (config.mlir_opt, input_file, seg)
                cmd1 = cmd1 + cmd2
            cmd_ = cmd1
        cmd=  cmd_  + ' -o %s' % (output_file)
    else:
        cmd = '%s %s -allow-unregistered-dialect %s -o %s' % (config.mlir_opt, input_file, singlePass, output_file)
    
    s1 = cmd.split("seed.mlir ")[1]
    save_pass = s1.split(" -o")[0]

    start_time = int(time() * 1000)
    pro = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, encoding="utf-8",preexec_fn=os.setsid)
    try:
        stdout, stderr = pro.communicate(timeout=30)
        return_code = pro.returncode
        log.info(cmd)
        if(stderr!=""):
            log.error(stderr)
    except subprocess.TimeoutExpired:
        # pro.kill()
        os.killpg(pro.pid,signal.SIGTERM) 
        stdout = ""
        stderr = "timeout, kill this process"
        return_code = -9

    stderr = stderr.replace('\'', '')
    end_time = int(time() * 1000)
    duration = int(round(end_time - start_time))

    if (flag == "mutate"):
        mutate_type =singlePass
    else:
        mutate_type = ''

    result = {
        "sid":sid,
        "input":raw_mlir,
        "mutate_type":mutate_type,
        "cmd":save_pass,
        "return_code":return_code,
        "stdout":stdout,
        "stderr":stderr,
        "duration":duration
    }

    return result

def RepMut(output_file,result,OPdict):
    raw_mlir = result["input"]
    output = raw_mlir
    result["return_code"] = 1
    if "affine" in OPdict:
        sub1 = re.sub(r'affine.load', 'memref.load', raw_mlir)
        output = re.sub(r'affine.store', 'memref.store', sub1)
        print("affine mutation")
        result["return_code"] = 0
    with open(output_file, 'w') as f:
        f.write(output)
    return result 

def execute_mlir(input_file, output_file, sid, raw_mlir, singlePass, config: Config, flag,OPdict) -> str:
    """
    执行编译
    """
    result = {
        "sid":sid,
        "input":raw_mlir,
        "mutate_type":"",
        "cmd":"",
        "return_code":"",
        "stdout":"",
        "stderr":"",
        "duration":"",
    }

    if singlePass.find("func.func") >= 0:
        newpass = []
        newpass = fixlowerpass(singlePass)
        cmd_ = '%s %s  %s' % (config.mlir_opt, input_file, newpass[0])
        cmd1 = cmd_
        if len(newpass)>1:
            for seg in newpass[1:]:
                cmd2 = '| %s %s' % (config.mlir_opt, seg)
                cmd1 = cmd1 + cmd2
            cmd_ = cmd1
        cmd=  cmd_  + ' -o %s' % (output_file)
    else:
        cmd = '%s %s -allow-unregistered-dialect %s -o %s' % (config.mlir_opt, input_file, singlePass, output_file)
    
    s1 = cmd.split("seed.mlir ")[1]
    save_pass = s1.split(" -o")[0]

    start_time = int(time() * 1000)
    pro = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True, encoding="utf-8",preexec_fn=os.setsid)
    try:
        stdout, stderr = pro.communicate(timeout=30)
        return_code = pro.returncode
        log.info(cmd)
        if(stderr!=""):
            log.error(stderr)
    except subprocess.TimeoutExpired:
        # pro.kill()
        os.killpg(pro.pid,signal.SIGTERM) 
        stdout = ""
        stderr = "timeout, kill this process"
        return_code = -9

    stderr = stderr.replace('\'', '')
    end_time = int(time() * 1000)
    duration = int(round(end_time - start_time))

    result["cmd"] = save_pass
    result["return_code"] = return_code
    result["stdout"] = stdout
    result["stderr"] = stderr
    result["duration"] = duration

    fuzzer = DTFuzz(config)
    from fuzz.fuzz import Fuzz
    if (result["return_code"] != 0):
        Fuzz.failer_handler(fuzzer, result,flag,config)

    return result


def runPass_OnlyLower(input_file, output_file, result, candidate_lowerPass, config: Config,flag):
    result = execute_pass(input_file, output_file, result, candidate_lowerPass, config,flag)
    fuzzer = Fuzz(config)
    if (result["return_code"] != 0):
        DTFuzz.failer_handler(fuzzer, result,flag,config)

    return result



def str_to_list(content: str) -> list:
    list = content.split(' ')
    return list

def filter_seed(dialects,candidate_lowerPass)->bool:
    is_filter = True
    for item in AllWeakLowerPass.all_weak_lower_pass:
        if item in candidate_lowerPass:
            is_filter = False
    return is_filter

def select_emiPass(dialects,operation)->str: 
    emi_pass = ""
    if "cf" not in dialects:
        emi_pass =  random.choice(["" ,"-BCF"])

    pass_list = []
    pass_list.append(emi_pass)
    return pass_list

def calculate_cooperation(dialect_list:list, config:Config)->float:
    cooperation = config.cooperation
    cooperation_result = 0
    for dialect in dialect_list:
        if dialect == "llvm":
            continue
        for key, value in cooperation.items():
            if dialect == key :
                cooperation_result = cooperation_result + value

    result = cooperation_result/75
    return result

class reportObject:
    def __init__(self, ids, stderror, returnCode, mlirContent):
        self.ids = ids
        self.stderror = stderror
        self.returnCode = returnCode
        self.mlirContent = mlirContent





# Directed Testing
class DTFuzz:
    def __init__(self, config: Config):
        self.config = config

    def select_seed(self,Nmax,num):
        sql = "select * from " + self.config.seed_pool_table + " where n < '%s' " % Nmax + "ORDER BY rand() limit %s" %num

        seed_types = dbutils.db.queryAll(sql)
        return seed_types

    def select_seed_DT(self,Nmax,dia,num):
        sql = "select * from " + self.config.seed_pool_table + " where n < '%s' and dialect like '%%" % Nmax +dia+"%%' ORDER BY rand() limit %s" %num
        seed_types = dbutils.db.queryAll(sql)

        return seed_types
    
    def select_seed_DT_op(self,Nmax,op,num):
        sql = "select * from " + self.config.seed_pool_table + " where n < '%s' and operation like '%%" % Nmax +op+"%%' ORDER BY rand() limit %s" %num
        seed_types = dbutils.db.queryAll(sql)
        return seed_types
    

    def select_seed_DT_limit(self,Nmax,dia,num):
        sql = "select * from " + self.config.seed_pool_table + " where n < '%s' and dialect like '%%" % Nmax +dia+"%%' ORDER BY rand() limit %s" %num
        seed_types = dbutils.db.queryAll(sql)
        return seed_types
    
    def success_handler(self, sid, result_type, content, candidate_lower_pass, cooperation, from_content, from_cmd):
        try:
            sql = "insert into " + self.config.seed_pool_table + \
                  " (dialect,content,n, candidate_lower_pass, cooperation, from_sid, from_content, from_cmd) " \
                  "values ('%s','%s','%s','%s','%s','%s','%s','%s')" \
                  % \
                  (result_type, content, 0, candidate_lower_pass, cooperation, sid, from_content, from_cmd)
            dbutils.db.executeSQL(sql)
            self.update_nindex(sid)
        except Exception as e:
            log.error('sql error', e)

    def mutate_success_handler(self, sid, mtype,dialects,operations, content, candidate_lower_pass,flag):
        if (flag=="opt"):
            source = "O"
        elif (flag=="mutate"):
            source = "M"
        elif (flag == "lower"):
            source = "L"
        try:
            sql = "insert into " + self.config.seed_pool_table + \
                  " (preid,source,mtype,dialect,operation, content,n, candidate_lower_pass) " \
                  "values ('%s','%s','%s','%s','%s','%s','%s','%s')" \
                  % \
                  (sid,source,mtype,dialects, operations,content, 0, candidate_lower_pass)

            dbutils.db.executeSQL(sql)
            log.info(content)
            log.info("======save the result as seed")
        except Exception as e:
            log.error('sql error', e)

    def SplitContent(self,content:str)->str:
        content_list = content.split(("\n"))
        if content.find("LLVM ERROR:") >= 0:
            for item in content_list:
                if item.find("LLVM ERROR:") >= 0:
                    return "LLVM ERROR:" + item.split("LLVM ERROR:")[1]
        errorFunc = ''
        errorMessage = ''
        if content.find("Assertion") >= 0:
            for i in range(0,len(content_list)-1):
                if content_list[i].find("Assertion") >= 0:
                    errorMessage = content_list[i].split("Assertion")[1]
                if content_list[i].find("__assert_fail_base") >= 0:
                    errorFunc = content_list[i+2]
                    errorFunc = errorFunc[23:]
                    break
                


            return "Assertion:" + errorMessage + "\n" + errorFunc

        if content.find("Segmentation fault") >= 0:
            for i in range(0,len(content_list)-1):
                if content_list[i].find("__restore_rt") >= 0:
                    errorFunc = content_list[i+1]
                    errorFunc = errorFunc.split("(/home")[0]
                    errorFunc = errorFunc[22:]
                    break
            return "Segmentation fault:" + errorFunc

    def updateReportModel(self,sid,error,report_stack_dict,stderr,return_code,content,report_model_dict):

        error = error.replace('\'','')
        if error in report_stack_dict.keys() and return_code!=-9 and error.find("PLEASE submit a bug report")<=0:
            log.info("===== update sids in report table =====")

        else:
            log.info("===== add new report record in reportObject =====")
            model = reportObject(ids=sid, stderror=stderr, returnCode=return_code, mlirContent=content)
            report_model_dict[error] = model
    

    def stackStatistic(self,sid,return_code,stderr,content,conf):
        log.info("===== starting analysis error =====")
        sql = "select * from " + self.config.report_table + " where stderr is not NULL and stderr != ''"
        data = dbutils.db.queryAll(sql)
        report_model_dict = {}   
        report_stack_dict = {}
        for item in data:
            stack_error = item[0]
            sids = item[1]
            report_stack_dict[stack_error] = sids

        firstLineInStderr = stderr.split("\n")[0]

        if stderr.find("Assertion") >= 0 or stderr.find("LLVM ERROR:") >=0 or stderr.find("Segmentation fault (core dumped)")>=0:
            error = DTFuzz.SplitContent(self,stderr)
        else:  # time out & others
            error = firstLineInStderr

        DTFuzz.updateReportModel(self,sid,error,report_stack_dict,stderr,return_code,content,report_model_dict)
         
        for key,value in report_model_dict.items():
            log.info("===== new report record insert table =====")
            now = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')

            returnCode = value.returnCode

            mlirContent = value.mlirContent.replace('\'', '\\\'')

            
            try:
                sql = "insert into " + self.config.report_table  + \
                " (stack,sids,datetime,stderr,returnCode,mlirContent) " \
                " values('%s','%s','%s','%s','%s','%s')" \
                % \
                (key, str(conf.Iter), now, stderr, returnCode, mlirContent)

                dbutils.db.executeSQL(sql)
                log.info("======report this error successfully")
            except Exception as e:
                log.error('sql error', e)



    def failer_handler(self,result,flag,conf):
        result_list = list(result.values());
        sid, content = result_list[0:2]
        if (flag=="opt"):
            source = "O"
        elif (flag=="mutate"):
            source = "M"
        elif (flag == "lower"):
            source = "L"

        cmd, return_code, stdout,stderr, duration = result_list[3:]


        returnCode_list = [-9,134,139]

        if return_code in returnCode_list:
            DTFuzz.stackStatistic(self,sid,return_code,stderr,content,conf)

        now = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
        try:
            sql = "insert into  " + self.config.result_table + \
                  " (sid, content,phase,cmd,returnCode,stdout,stderr,duration,datetime) " \
                  "values ('%s','%s','%s','%s','%s','%s','%s','%s','%s')" \
                  % \
                  (sid, content, source, cmd, return_code, stdout, stderr, duration, now)
            # log.info(sql)
            dbutils.db.executeSQL(sql)
            log.info("======save this error successfully")
        except Exception as e:
            log.error('sql error', e)
            with open(conf.temp_dir + '/saveerror.txt', mode='a+') as f:
                f.write(sql)
        self.update_nindex(sid,False)

    def update_nindex(self, sid, reset=False):
        if reset:
            sql = "update " + self.config.seed_pool_table + " set n = 0 where sid = '%s'" % sid
            dbutils.db.executeSQL(sql)
        else:
            sql = "update " + self.config.seed_pool_table + " set n = n +1 where sid = '%s'" % sid
            dbutils.db.executeSQL(sql)


    def generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file):
        if '.' in dia:
            op = dia
            seeds = DTFuzz.select_seed_DT_op(self,conf.Nmax,op,100)
        elif dia =="":
            seeds = DTFuzz.select_seed(self,conf.Nmax,100)
        else:
            seeds = DTFuzz.select_seed_DT(self,conf.Nmax,dia,100)

        for selected_seed in seeds:
            sid = selected_seed[0]
            dialects,operation,raw_mlir,n,lowerPass = selected_seed[-5:]

            dialect_list = dialects.split(',')
            OPdict = {key: [] for key in dialect_list}
            if operation !=' ':
                word_list = operation.split(',')
                for item in word_list:
                    d1, d2 = item.split('.',1)
                    if d1 in OPdict:
                        if d2 not in OPdict[d1]:
                            OPdict[d1].append(d2)
                

            f = open(seed_file, 'w', encoding="utf-8")
            f.write(raw_mlir)
            f.close()

            if Mut=='3':
                Mut = np.random.choice(['0','1','2'])

            if(Mut!='0'):
                log.info("================== Enable mutation ====================")
                flag = "mutate"

                import fuzz.fuzz as Fuzz
                result = Fuzz.Mutator(seed_file, mut_file, sid, raw_mlir, Mut, conf, dialects,operation)
                if (result["return_code"] == 0) :
                    save_mutate_seed(seed_file,mut_file,result, conf,flag)


            log.info(sid)
            log.info("================== Enable Lowering ====================")

            if target == "scf.for":
                lower_pass = "-lower-affine"
            if target == "scf.parallel":
                lower_pass = "-linalg-bufferize -convert-linalg-to-parallel-loops"
            if target =="affine.parallel":
                lower_pass = "-affine-parallelize"
            if target == "affine":
                lower_pass = "-linalg-bufferize -convert-linalg-to-affine-loops -affine-super-vectorize=\"virtual-vector-size=32,256 test-fastest-varying=1,0\""
            if target == "linalg":
                lower_pass = "-pass-pipeline=\"builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))\""
            if target == "gpu":
                lower_pass = "-gpu-map-parallel-loops -convert-parallel-loops-to-gpu"
            if target == "async":
                lower_pass = "-async-parallel-for"
            if target == "spirv":
                lower_pass = "-convert-math-to-spirv"
            if target == "gpu.launch_func":
                lower_pass = "-gpu-kernel-outlining"
            if target == "omp":
                lower_pass = "-convert-scf-to-openmp"
            if target == "nvvm":
                lower_pass = "--convert-gpu-to-nvvm"
            if target == "rocdl":
                lower_pass = "--convert-gpu-to-rocdl"
            if target =="":
                lower_pass = setLowerSeq(lowerPass)


            flag = "lower"
            result = execute_mlir(seed_file, lower_file, sid, raw_mlir, lower_pass, conf, flag,[])
            from fuzz.fuzz import Fuzz
            Fuzz.FuzzingSave(result,flag,conf,dialects,operation)
            if (result["return_code"] == 0) :
                analysis_and_save_seed(seed_file, lower_file, result, conf, flag)
            else:
                log.error("current seed need to analysis，without dialects and lowerPass")

            log.info("================== Enable optimization ====================")
            flag = "opt"
            opt_pass= setOptSeqR()
            result = execute_mlir(seed_file, opt_file, sid, raw_mlir, opt_pass, conf,flag,[])
            Fuzz.FuzzingSave(result,flag,conf,dialects,operation)
            if (result["return_code"] == 0):
                analysis_and_save_seed(seed_file,opt_file,result, conf,flag)
        

    def DirectedLower(self,conf,seed_file,lower_file,opt_file,Mut,mut_file):
        
        dia = "tosa"
        target ="linalg"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)

        dia ="linalg"
        target ="affine"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)

        dia ="linalg"
        target ="scf.parallel"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)

        dia ="affine"
        target ="affine.parallel"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)

        dia ="affine"
        target ="scf.for"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)


        dia ="scf.parallel"
        target ="gpu"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)

        dia ="gpu"
        target ="async"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)

        dia ="math"
        target ="spirv"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)
        dia ="gpu.launch"
        target ="gpu.launch_func"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)

        dia ="scf.parallel"
        target ="omp"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)

        dia =""
        target =""
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)


        dia ="gpu.launch"
        target ="nvvm"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)

        dia ="gpu.launch"
        target ="rocdl"
        log.info(dia + "-->" + target)
        DTFuzz.generateLowIR(self,dia,target,conf,seed_file,lower_file,opt_file,Mut,mut_file)

    def process(self,Mut):
        conf = self.config
        log.info("Iter :"+ str(conf.Iter))
        seed_file = conf.temp_dir + "seed" + ".mlir"

        if not os.path.exists(seed_file):
            os.system(r"touch {}".format(seed_file))
        mut_file = conf.temp_dir + "mut" + ".mlir"
        opt_file = conf.temp_dir + "opt" + ".mlir"
        lower_file = conf.temp_dir + "lower" + ".mlir"

        log.info("================== select seed at random====================")


        dia = "affine"
        seeds = self.select_seed_DT_limit(conf.Nmax,dia,50)

        for selected_seed in seeds:
            sid = selected_seed[0]
            dialects,operation,raw_mlir,n,lowerPass = selected_seed[-5:]

            dialect_list = dialects.split(',')
            OPdict = {key: [] for key in dialect_list}
            if operation !=' ':
                word_list = operation.split(',')
                for item in word_list:
                    d1, d2 = item.split('.',1)
                    if d1 in OPdict:
                        if d2 not in OPdict[d1]:
                            OPdict[d1].append(d2)
                

            f = open(seed_file, 'w', encoding="utf-8")
            f.write(raw_mlir)
            f.close()

            log.info(sid)


            if(Mut=='1'):
                log.info("================== Enable mutation ====================")
                flag = "mutate"
                mutate_pass = select_emiPass(dialects,operation)

                if(mutate_pass[0]!=""):
                    result = execute_mlir(seed_file, mut_file, sid, raw_mlir, mutate_pass, conf, flag,OPdict)
                    if (result["return_code"] == 0) :
                        analysis_and_save_seed(seed_file,mut_file,result, conf,flag)

        
            log.info("================== Enable optimization ====================")
            flag = "opt"
            opt_pass= setOptSeq(dia)
            result = execute_mlir(seed_file, opt_file, sid, raw_mlir, opt_pass, conf,flag,[])
            if (result["return_code"] == 0):
                analysis_and_save_seed(seed_file,opt_file,result, conf,flag)


            log.info("================== Enable Lowering ====================")
            flag = "lower"
            lower_pass = setLowerSeq(lowerPass)
            result = execute_mlir(seed_file, lower_file, sid, raw_mlir, lower_pass, conf, flag,[])

            if (result["return_code"] == 0) :
                analysis_and_save_seed(seed_file, lower_file, result, conf, flag)
            else:
                log.error("current seed need to analysis，without dialects and lowerPass")


    def affineGen(self):

        sql = "select content FROM tosa_testcase" 

        dataList = dbutils.db.queryAll(sql)
        conf = self.config
        i = 0
        for data in dataList:
            seed_file = conf.temp_dir + "seed" + ".mlir"
            if not os.path.exists(seed_file):
                os.system(r"touch {}".format(seed_file))
            

            f = open(seed_file, 'w', encoding="utf-8")
            f.write(data[0])
            f.close()

            i = i+1
 
            log.info("======generate seed : " + str(i))
            target_file = conf.temp_dir + str(i)+ ".mlir"

    
            singlePass = "-pass-pipeline=\"builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))\" -linalg-bufferize -convert-linalg-to-affine-loops -affine-parallelize"
            

            cmd = '%s %s -allow-unregistered-dialect %s -o %s' % (conf.mlir_opt, seed_file, singlePass, target_file)

            pro = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True, encoding="utf-8",preexec_fn=os.setsid)
            try:
                stdout, stderr = pro.communicate(timeout=30)
                returnCode = pro.returncode
                log.info(cmd)
                if not os.path.exists(target_file):
                    log.error(stderr)
                    continue
                from generator.tosaGen import seedAnalysis
                dialects,candidate_lower_pass,operations = seedAnalysis(conf,target_file)
                log.info(candidate_lower_pass)
                log.info(operations)
                with open(target_file, 'r') as f:
                    content = f.read()
                log.info(content)
                try:
                    sql = "insert into "+ conf.seed_pool_table + \
                        " (preid,source,mtype,dialect,operation,content,n, candidate_lower_pass) " \
                        "values ('%s','%s','%s','%s','%s','%s','%s','%s')" \
                        % \
                        (0,'L4','',dialects, operations,content, 0, candidate_lower_pass)
                    dbutils.db.executeSQL(sql)

                except Exception as e:
                    log.error('sql error', e)
                os.remove(target_file)
             
            except subprocess.TimeoutExpired:
                os.killpg(pro.pid,signal.SIGTERM) 
                stdout = ""
                stderr = "timeout, kill this process"
                returnCode = -9

            if i==800:
                break


