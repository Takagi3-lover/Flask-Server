import os
from openbabel import openbabel

import pandas as pd

from utils.cdkDesc import getChemistryFeaturesByCDK
from utils.globalVar import *
from utils.padelConversion import getFeaturesByPadel


# 利用open babel实现从smiles文件转换为mdl文件
def StringFormatConversion(inputFile, outputFile, input_format="smiles", output_format="mdl"):
    # 清空tempFiles文件夹
    for file in os.listdir(tempPath):
        os.remove(tempPath + file)
    conv = openbabel.OBConversion()  # 使用open babel的OBConversion函数，用于文件格式转换
    conv.OpenInAndOutFiles(inputFile, outputFile)  # 定义需要转换的文件名，以及转换后的文件名
    conv.SetInAndOutFormats(input_format, output_format)  # 定义转换文件前后的格式
    conv.Convert()  # 执行转换操作
    conv.CloseOutFile()  # 关闭转换后的文件


# 将两个csv文件进行拼接
def concatCsv(file1, file2, output, moleNames):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2], axis=1)
    # 删除Title列
    df.drop("Title", axis=1, inplace=True)
    df.insert(0, "MolName", moleNames)
    df.to_csv(resultPath + output + "_Result.csv", index=False)


# 根据用户提交的txt文件，对csv的特征进行筛选,txt文件中，每一行为一个特征值
def selectFeaturesByTXT(features, csvFile, output):
    type = ["HOMO", "LUMO"]
    i = 0
    for txtFile in features:
        postFix = type[i]
        i = i+1
        with open(txtFile, "r") as f:
            lines = f.readlines()
            features = ["MolName"]
            for line in lines:
                features.append(line.strip())

        df = pd.read_csv(csvFile)
        try:
            df = df.loc[:, features]
        except KeyError:
            print("特征值文件中存在不合法的特征值！")

        df.to_csv(resultPath + output + "_" + postFix + ".csv", index=False)


def formatConvert(molecule, filename):
    # 获取当前时间
    loca = filename
    # 将molecule存入.smiles文件
    with open(initPath + loca+".smiles", "w") as f:
        f.write(molecule)
    # 将.smiles文件转换为.mdl文件
    StringFormatConversion(initPath + loca + ".smiles", tempPath + loca + ".mdl")

    # # 利用CDKDesc获取特征值
    getChemistryFeaturesByCDK(loca + ".mdl", loca + "1.txt")

    # 调用函数，利用padel将mdl文件转换为csv文件
    getFeaturesByPadel(loca + ".mdl", loca + "2.csv")

    # 将csv文件进行拼接
    concatCsv(tempPath + loca + "1.csv", tempPath + loca + "2.csv", loca, molecule)

    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # 将|转换为/
    root = root.replace("\\", "/")
    txtFile_HOM0 = root + "/files/features_HOMO.txt"
    txtFile_LUMO = root + "/files/features_LUMO.txt"
    features = [txtFile_HOM0, txtFile_LUMO]

    selectFeaturesByTXT(features, resultPath + loca + "_Result.csv", loca)

