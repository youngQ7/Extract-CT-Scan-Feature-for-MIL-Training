# from pathlib import Path
# from tqdm import tqdm
# import os
# import pandas as pd
# from pypinyin import pinyin, Style
#
#
# # 定义函数将汉字转换为拼音
# def chinese_to_pinyin(name):
#     pinyin_list = pinyin(name, style=Style.NORMAL)
#     return ''.join([p[0] for p in pinyin_list])
#
# if __name__ == '__main__':
#     # 合并多个数据目录下的子目录
#     patients = []
#     for data_dir in [r'E:\1_1\after', r'E:\2_1\after', r'G:\三峡\2020\术后CT', r'G:\三峡\2021\术后CT', r'G:\三峡\2022\术后CT', r'G:\三峡\2023\术后CT']:
#         patients.extend([p for p in Path(data_dir).iterdir() if p.is_dir()])
#     total = len(patients)
#
#     # 读取 label.csv 文件
#     df = pd.read_csv(r'D:\Code\pythonProject\BrianCT\label.csv')
#     # 初始化统计变量
#     label_0_patient_count = 0
#     label_0_file_count = 0
#     label_1_patient_count = 0
#     label_1_file_count = 0
#     # 若不存在 Path 列，创建该列
#     if 'Path' not in df.columns:
#         df['Path'] = None
#
#     for patient in tqdm(patients, total=total):
#         # print(patient)
#         patient_name = os.path.basename(patient)
#         label = None
#         # 先直接查找
#         match = df[df['name'] == patient_name]
#         if len(match) == 1:
#             index = match.index[0]
#             label = match['label'].values[0]
#             df.at[index, 'Path'] = patient
#         elif len(match) > 1:
#             print(f"找到多个匹配项的病人目录: {patient}")
#         if label is None:
#             # 若未找到，转换为拼音再查找
#             pinyin_name = chinese_to_pinyin(patient_name)
#             match = df[df['name'] == pinyin_name]
#             if len(match) == 1:
#                 index = match.index[0]
#                 label = match['label'].values[0]
#                 df.at[index, 'Path'] = patient
#             elif len(match) > 1:
#                 print(f"找到多个匹配项的病人目录（拼音查找）: {patient}")
#         if label is None:
#             print(f"未找到对应内容的病人目录: {patient}")
#             continue
#
#         file_count = 0
#         # 深度搜索统计文件数量
#         for root, dirs, files in os.walk(patient):
#             file_count += len(files)
#
#         if label == 0:
#             label_0_patient_count += 1
#             label_0_file_count += file_count
#         else:
#             label_1_patient_count += 1
#             label_1_file_count += file_count
#     # 将更新后的 DataFrame 保存回 CSV 文件
#     df.to_csv(r'D:\Code\pythonProject\BrianCT\label.csv', index=False)
#     print(f"标签为 0 的病人数量: {label_0_patient_count}，相应的文件总数: {label_0_file_count}")
#     print(f"标签为 1 的病人数量: {label_1_patient_count}，相应的文件总数: {label_1_file_count}")



# import pandas as pd
#
# # 读取 CSV 文件
# csv_file_path = r'D:\Code\pythonProject\BrianCT\label.csv'
# csv_df = pd.read_csv(csv_file_path)
#
# # 读取 XLSX 文件
# xlsx_file_path = r"E:\2-MCE-2.xlsx"
# xlsx_df = pd.read_excel(xlsx_file_path)
#
# # 初始化新列
# csv_df['Chinese'] = None
#
# # 遍历 CSV 中的 name 字段
# for index, row in csv_df.iterrows():
#     name = row['name']
#
#     # 提取 XLSX 中文件夹名称 _ 之后的内容
#     xlsx_df['folder_name_suffix'] = xlsx_df['文件夹名称'].apply(lambda x: x.split('_')[-1] if '_' in x else x)
#     # 查找 XLSX 中文件夹名称 _ 之后内容匹配的行
#     matches = xlsx_df[xlsx_df['folder_name_suffix'] == name]
#
#     # # 查找 XLSX 中文件夹名称匹配的行
#     # matches = xlsx_df[xlsx_df['文件夹名称'] == name]
#
#     # 检查匹配情况
#     if len(matches) == 1:
#         csv_df.at[index, 'Chinese'] = matches.iloc[0]['姓名']
#
# # 将更新后的 CSV 数据保存到新文件
# new_csv_file_path = 'updated_csv_file.csv'
# csv_df.to_csv(new_csv_file_path, index=False)



import pandas as pd

# 读取文件
df = pd.read_csv(r"D:\Code\pythonProject\BrianCT\updated_csv_file.csv", encoding='GB2312')

# 定义函数来提取姓名
def extract_name(path):
    parts = path.split('\\')
    return parts[-1] if parts else None

# 应用函数到 Path 列，并将结果存入 Chinese 列
df['Chinese'] = df['Path'].apply(extract_name)

# 将结果保存为新的 CSV 文件
new_csv_path = r"D:\Code\pythonProject\BrianCT\updated_csv_file.csv"
df.to_csv(new_csv_path, index=False)