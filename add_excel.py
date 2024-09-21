import pandas as pd

# Excel文件路径
file_path = '附件一（训练集）.xlsx'

# 使用pandas读取Excel文件
xls = pd.ExcelFile(file_path)

# 初始化一个列表，用于存储所有sheet的DataFrame
dfs = []

# 遍历Excel文件中的所有sheet
for sheet_name in xls.sheet_names:
    # 读取每个sheet的数据
    df = pd.read_excel(xls, sheet_name)
    # 将DataFrame添加到列表中
    dfs.append(df)

# 使用concat一次性合并所有DataFrame
merged_data = pd.concat(dfs, ignore_index=True)

# 保存合并后的数据到一个新的Excel文件
merged_data.to_excel('add_1.xlsx', index=False)

# 关闭Excel文件
xls.close()
