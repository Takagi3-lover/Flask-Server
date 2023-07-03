import os
import pandas as pd
from utils.globalVar import *
# 选项配置文件
selectionFile = "CDKDesc/defaultSelect.xml"


def getChemistryFeaturesByCDK(inputFile, outputFile):
    input = tempPath + inputFile
    output = tempPath + outputFile
    # 在命令行中,运行CDKDescUI-1.4.8.jar
    cmd = "java -jar CDKDesc/CdkDescUI-1.4.8.jar" + " -b " + input + " -o " + output + " -s " + selectionFile
    os.system(cmd)
    fileName = os.path.splitext(output)[0]
    # 逐行读取生成的txt文件
    with open(output, "r") as f:
        lines = f.readlines()
        # 每行以空格为分割符，将第一行作为特征名，后面的行作为特征值
        features = lines[0].split()
        values = []
        for line in lines[1:]:
            values.append(line.split())
        # 将特征名和特征值转换为DataFrame
        df = pd.DataFrame(values, columns=features)
        # 删除从列名为PPSA-1到RPSA的列
        df.drop(df.loc[:, "PPSA-1":"RPSA"].columns, axis=1, inplace=True)
        # 保存为csv文件
        df.to_csv(fileName + ".csv", index=False)
