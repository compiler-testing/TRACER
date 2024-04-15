# -*- coding: utf-8 -*-
import random
from time import time
from random import randint
import re


def conv_2d_nhwc_fhwc(sub1):
    pattern = r"linalg.conv_2d_nhwc_fhwc"

    if pattern in sub1:

        match = re.search(pattern, sub1)
        start = match.start()
        end = match.end()

        sub1 = re.sub(pattern, "linalg.conv_2d_nchw_fchw", sub1, count=1)
        
        start0 = end
        for _ in range(5):
            pattern = r'(tensor<)(\d+x)(\d+x)(\d+x)(\d+x)(f32|f64)(>)'
            match = re.search(pattern,sub1[start0:])
            if match:
                sub1 = sub1[:start0+match.start()] + match.group(1) + match.group(2) + match.group(5) + match.group(3) + match.group(4) + match.group(6) + match.group(7) + sub1[start0+match.end():]
                start0 = start0+match.end()

        
        substrings = re.findall( r'%(.*?)[,:]', sub1[end:])
        input =  "%"+substrings[0]
        kernel = "%"+substrings[1]
        output = "%"+substrings[2]
    

        for _ in range(20):
            start = sub1.rfind("\n", 0, start - 1)
            if start == -1:
                break
        start += 1
        

        pattern = input+r'(: tensor<)(\d+x)(\d+x)(\d+x)(\d+x)(f32|f64)(>)'
        match = re.search(pattern,sub1[start:end])
        #nhwc->nchw
        if match:
            sub1 = sub1[:match.start()] + input + match.group(1) + match.group(2) + match.group(5) + match.group(3) + match.group(4) + match.group(6) + match.group(7) + sub1[match.end():]
        

        pattern = kernel+r'(: tensor<)(\d+x)(\d+x)(\d+x)(\d+x)(f32|f64)(>)'
        match = re.search(pattern,sub1[start:end])
        #nhwc->nchw
        if match:
            sub1 = sub1[:match.start()] + kernel + match.group(1) + match.group(2) + match.group(5) + match.group(3) + match.group(4) + match.group(6) + match.group(7) + sub1[match.end():]

        pattern = output+r'(: tensor<)(\d+x)(\d+x)(\d+x)(\d+x)(f32|f64)(>)'
        match = re.search(pattern,sub1[start:end])
        #fhwc->fschw
        if match:
            sub1 = sub1[:match.start()] + output + match.group(1) + match.group(2) + match.group(5) + match.group(3) + match.group(4) + match.group(6) + match.group(7) + sub1[match.end():]
        
        if sub1[match.end() + 1:match.end() + len(" -> ") + 1] == " -> ":
            start1 = match.end() + len(" -> ")
            pattern = r'(tensor<)(\d+x)(\d+x)(\d+x)(\d+x)(f32|f64)(>)'
            match = re.search(pattern,sub1[start1:end])
            #fhwc->fschw
            if match:
                sub1 = sub1[:start1+match.start()] + match.group(1) + match.group(2) + match.group(5) + match.group(3) + match.group(4) + match.group(6) + match.group(7) + sub1[start1 + match.end():]
    return sub1


def linalg_pooling_nchw(sub1):
    pattern1 = r"linalg.pooling_nhwc_max " 
    pattern2 = r"linalg.pooling_nhwc_sum "
    if pattern1 in sub1:
        pattern = pattern1
    elif pattern2 in sub1:
        pattern = pattern2
    else:
        pattern = pattern1
    
    if pattern in sub1:
        match = re.search(pattern, sub1)
        start = match.start()
        end = match.end()
        random_code = random.choice([
        lambda: re.sub(pattern, 'linalg.pooling_nchw_max ', sub1, count=1),
        lambda: re.sub(pattern, 'linalg.pooling_nchw_sum ', sub1, count=1)
        ])
        sub1 = random_code()
        
        start0 = end
        for _ in range(4):
            pattern = r'(tensor<)(\d+x)(\d+x)(\d+x)(\d+x)(f32|f64)(>)'
            match = re.search(pattern,sub1[start0:])
            if match:
                sub1 = sub1[:start0+match.start()] + match.group(1) + match.group(2) + match.group(5) + match.group(3) + match.group(4) + match.group(6) + match.group(7) + sub1[start0+match.end():]
                start0 = start0+match.end()
        
        substrings = re.findall( r'%(.*?)[,:]', sub1[end:])
        input =  "%"+substrings[0]
        output = "%"+substrings[2]


        for _ in range(20):
            start = sub1.rfind("\n", 0, start - 1)
            if start == -1:
                break
        start += 1
        

        pattern = input+r'(: tensor<)(\d+x)(\d+x)(\d+x)(\d+x)(f32|f64)(>)'
        match = re.search(pattern,sub1[start:end])
        #nhwc->nchw
        if match:
            sub1 = sub1[:match.start()] + input + match.group(1) + match.group(2) + match.group(5) + match.group(3) + match.group(4) + match.group(6) + match.group(7) + sub1[match.end():]
        

        pattern = output+r'(: tensor<)(\d+x)(\d+x)(\d+x)(\d+x)(f32|f64)(>)'
        match = re.search(pattern,sub1[start:end])
        #nhwc->nchw
        if match:
            sub1 = sub1[:match.start()] + output + match.group(1) + match.group(2) + match.group(5) + match.group(3) + match.group(4) + match.group(6) + match.group(7) + sub1[match.end():]
        
            if sub1[match.end() + 1:match.end() + len(" -> ") + 1] == " -> ":
                start1 = match.end() + len(" -> ")
                pattern = r'(tensor<)(\d+x)(\d+x)(\d+x)(\d+x)(f32|f64)(>)'
                match = re.search(pattern,sub1[start1:end])
                #fhwc->fschw
                if match:
                    sub1 = sub1[:start1+match.start()] + match.group(1) + match.group(2) + match.group(5) + match.group(3) + match.group(4) + match.group(6) + match.group(7) + sub1[start1 + match.end():]
    return sub1       



def RepMut(mlir):
    sub1 = mlir
    output = mlir

    if "affine.load" in sub1 or "affine.store" in sub1:
        sub1 = re.sub(r'affine.load', 'memref.load', sub1)
        sub1 = re.sub(r'affine.store', 'memref.store', sub1)
        print("affine mutation")

    if "linalg" in sub1:

        sub1 = re.sub(r'linalg.batch_matmul', 'linalg.batch_matmul_transpose_b', sub1)
        sub1 = re.sub(r'linalg.conv_3d_ndhwc_dhwcf', 'linalg.conv_3d_ncdhw_fcdhw', sub1)
        sub1 = re.sub(r'linalg.fill', 'linalg.copy', sub1)
        

        random_code = random.choice([
        lambda: re.sub(r'linalg.pooling_nhwc_max', 'linalg.pooling_nhwc_max_unsigned', sub1),
        lambda: re.sub(r'linalg.pooling_nhwc_max', 'linalg.pooling_nhwc_min', sub1),
        lambda: re.sub(r'linalg.pooling_nhwc_max', 'linalg.pooling_nhwc_min_unsigned', sub1)
        ])
        sub1 = random_code()


        random_code = random.choice([
        lambda: re.sub(r'linalg.pooling_nhwc_sum', 'linalg.pooling_nhwc_max_unsigned', sub1),
        lambda: re.sub(r'linalg.pooling_nhwc_sum', 'linalg.pooling_nhwc_min', sub1),
        lambda: re.sub(r'linalg.pooling_nhwc_sum', 'linalg.pooling_nhwc_min_unsigned', sub1)
        ])
        sub1 = random_code()

        sub1 = conv_2d_nhwc_fhwc(sub1)
        sub1 = linalg_pooling_nchw(sub1)
        
        print("linalg mutation")

    output = sub1
    # print(output)
    return output
    

