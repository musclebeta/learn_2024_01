import matplotlib.pyplot as plt
from collections import Counter

# 文件路径
file_name1 = r'/home/musclebeta/learn_2024_01/CCL/CFN2.1/cfn-train.json_target.txt'
file_name2 = r'/home/musclebeta/learn_2024_01/CCL/CFN2.1/cfn-test-A.json_target.txt'
file_name3 = r'/home/musclebeta/learn_2024_01/CCL/CFN2.1/cfn-dev.json_target.txt'
file_names = [file_name1, file_name2, file_name3]

plt.rcParams['font.sans-serif'] = ['AR PL UKai CN'] 
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


for file_name in file_names:

    # 读取文件并计算词频
    with open(file_name, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()  # 读取所有行并分割
        word_counts = Counter(words)  # 计算词频

    # 获取最常见的词及其频率
    common_words = word_counts.most_common(30)
    words, frequencies = zip(*common_words)

    # 可视化
    plt.figure(figsize=(30, 8))
    plt.bar(words, frequencies, color='skyblue')
    plt.xlabel('词')
    plt.ylabel('频率')
    plt.title('前10个词频率')
    plt.xticks(rotation=45)
    plt.show()
