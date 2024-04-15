### 控制测试执行12小时


### 根据关键字kill进程
```
ps -ef | grep 'NNDT3 fuzzer11'| awk '{print $2}' | xargs kill -9
```
ps -ef | grep 'NNDT3 fuzzer11'| awk '{print $2}' | xargs kill -9
### 查看文件数量
```
find ./ -type f | wc -l
```
2023.12.7
NN 1-5
LE 6-10
MU 11-15
NE 16-20
NNC2 21

MFM1-3  1-3 
MFR1-3  4-6
MFRM1-3 7-9

MFMC1-3  11-12 
MFRC1-3  13-14
MFRMC1-3  15 16

2023.12.8
MSC 1-5
MFC1 6-10

chain 11-15
api 16-20
screen -S MFapi1
docker exec -it mlirfuzzer20 bash
cd data/mlirfuzzer_exp/fuzz_tool/
./generator.sh MFapi5 api

docker exec -it mlirfuzzer3 bash
cd data/mlirfuzzer_exp/fuzz_tool/
./generator.sh MFRM1 multi-branch

/data/anaconda3/bin/python ./src/main.py --opt=generator  --sqlName=MFRM1

screen -S MSC4
docker exec -it mlirfuzzer3 bash
cd data/mlirfuzzer_exp/fuzz_tool/
./fuzz.sh MSC5


docker exec -it mlirfuzzer15 bash
cd data/mlirfuzzer_exp/fuzz_tool/
./loadInput.sh MUC5

```
NNC2
docker exec -it mlirfuzzer21 bash
cd data/mlirfuzzer_exp/fuzz_tool/
./fuzz.sh NNC2
```

### 清除所有gcda文件
```
./rmgcda.sh
```
```
rm -rf $(find . -name '*.gcda')

```
### 清除数据库记录
```
nohup python3 coverage.py NNDT1 fuzzer11 >/dev/null 2>&1 &
```
cd /home/ty/compiler-testing/fuzz_tool/
python3 ./src/main.py --opt=generator  --path=/home/ty/fuzzer16/llvm-project-16  --sqlName=MFRMC2

./run_fuzz.sh MFM1 fuzzer1 1


cd /home/ty/compiler-testing/fuzz_tool/src/coverage/
```

./run_fuzz.sh MS1 fuzzer1
./run_fuzz.sh MS2 fuzzer2
./run_fuzz.sh MS3 fuzzer3
./run_fuzz.sh MS4 fuzzer4
./run_fuzz.sh MS5 fuzzer5

./run_fuzz.sh MFMC1 fuzzer11 1
./run_fuzz.sh MFMC2 fuzzer12 1
./run_fuzz.sh MFRC1 fuzzer13 2
./run_fuzz.sh MFRC2 fuzzer14 3
./run_fuzz.sh MFRMC1 fuzzer15 3
./run_fuzz.sh MFRMC2 fuzzer16 3


./run_fuzz.sh LEDT1 fuzzer1
python3 coverage.py LEDT1 fuzzer1
```
 python3 ./src/main.py --opt=load  --sqlName=MS1 --path=/home/ty/fuzzer1/llvm-project-16
 
```
./run_fuzz.sh LEDT2 fuzzer2
python3 coverage.py LEDT2 fuzzer2
```
```
./run_fuzz.sh LEDT3 fuzzer3
python3 coverage.py LEDT3 fuzzer3
```


```
./run_fuzz.sh MUDT1 fuzzer4
python3 coverage.py MUDT1 fuzzer4
``

```
./run_fuzz.sh MUDT2 fuzzer5
python3 coverage.py MUDT2 fuzzer5
``

```
./run_fuzz.sh MUDT3 fuzzer6
python3 coverage.py MUDT3 fuzzer6
``

```
./run_fuzz.sh NNDT1 fuzzer7
python3 coverage.py NNDT1 fuzzer7
``
```
./run_fuzz.sh NNDT2 fuzzer8
python3 coverage.py NNDT2 fuzzer8
``
```
./run_fuzz.sh NNDT3 fuzzer9
python3 coverage.py NNDT3 fuzzer9
``


```
./run_fuzz.sh NEDT1 fuzzer10
python3 coverage.py NEDT1 fuzzer10
```
```
./run_fuzz.sh NEDT2 fuzzer11
python3 coverage.py NEDT2 fuzzer11
```
```
./run_fuzz.sh NEDT3 fuzzer12
python3 coverage.py NEDT3 fuzzer12
```


```
./run_fuzz.sh MFDT1 fuzzer13
python3 coverage.py MFDT1 fuzzer13
```
```
./run_fuzz.sh MFDT2 fuzzer14
python3 coverage.py MFDT2 fuzzer14
```
```
./run_fuzz.sh MFDT3 fuzzer15
python3 coverage.py MFDT3 fuzzer15
```

./run_fuzz.sh test fuzzer16

8-16
cd /home/ty/fuzzer16/llvm-project-16
git pull origin master

cd /home/ty/fuzzer15/llvm-project-16/build
ninja -j 20


