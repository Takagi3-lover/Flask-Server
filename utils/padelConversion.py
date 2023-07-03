import os

from padelpy import from_mdl, padeldescriptor
import pandas as pd
from utils.globalVar import *


# Args:
# mdl_file(str): path to MDL file
# output_csv(str): if supplied, saves descriptors / fingerprints here
# descriptors(bool): if `True`, calculates descriptors
# fingerprints(bool): if `True`, calculates fingerprints
# timeout(int): maximum time, in seconds, for conversion
# maxruntime(int): maximum running timeper molecule in seconds.default = -1.

# 定义模块，将转换后的mdl文件，通过padel处理，转换为csv
def mdlToCsv(mdl_file, output_csv):
    if os.path.exists(output_csv):
        os.remove(output_csv)
    # ！！！注意：此处的timeout参数，需要根据计算机的性能进行调整，超时后会报错
    # 当前电脑处理8个分子的时间为150s
    from_mdl(mdl_file, output_csv=output_csv, timeout=6000)

    # 输出后的csv文件，需要进行一些处理，才能被后续的模块使用
    df = pd.read_csv(output_csv)
    # 仅保留从列名为SsLi到SssssPb的列
    df1 = df.loc[:, "SsLi":"SssssPb"]
    # 仅保留从suml到末尾的列
    df2 = df.loc[:, "sumI":"DELS2"]
    # 将两个DataFrame进行拼接
    df = pd.concat([df1, df2], axis=1)
    # 保存处理后的csv文件
    df.to_csv(output_csv, index=False)


# descriptorType = 'PaDEL/descriptorTypes.xml'
configType = 'PaDEL/AutoConfig'
descriptorType = 'PaDEL/AutoDescriptor.xml'


# 利用padeldescriptor提供的接口，读取配置文件，减少不必要的分析。但是config配置文件中的内容必须单独指定
# 直接读取配置文件，不会生成csv文件
def getFeaturesByPadel(mdl_file, output_csv):
    input = tempPath + mdl_file
    output = tempPath + output_csv
    print("padel正在转换...")
    # 利用命令行，执行
    padeldescriptor(mol_dir=input, d_file=output, descriptortypes=descriptorType, d_2d=True)
    print("padel转换完成!")

    # 输出后的csv文件，需要进行一些处理，才能被后续的模块使用
    df = pd.read_csv(output)
    # 仅保留从列名为SsLi到SssssPb的列
    df1 = df.loc[:, "SsLi":"SssssPb"]
    # 仅保留从suml到末尾的列
    df2 = df.loc[:, "sumI":"DELS2"]
    # 将两个DataFrame进行拼接
    df = pd.concat([df1, df2], axis=1)
    # 保存处理后的csv文件
    df.to_csv(output, index=False)
    print("csv文件处理完成!")


# getFeaturesByPadel("sample.mdl", "sample2.csv")
