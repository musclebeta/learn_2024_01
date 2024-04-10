import json


file_name1 = r'/home/musclebeta/learn_2024_01/CCL/CFN2.1/cfn-train.json'
file_name2 = r'/home/musclebeta/learn_2024_01/CCL/CFN2.1/cfn-test-A.json'
file_name3 = r'/home/musclebeta/learn_2024_01/CCL/CFN2.1/cfn-dev.json'
file_names = [file_name1, file_name2, file_name3]



for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as file:
        
        datas = json.load(file)
        
        
        fe_count = len(datas) 
        count = 0
        
        for data in datas:
        
            targets = data['target']
            with open(f'{file_name}_target.txt', 'a') as file:
                    for i in range(len(targets)):
                        print(targets[i])
                        start = targets[i]['start']
                        end = targets[i]['end']
                        text = data['text'][start:end+1]
                        if len(targets) > 1:
                            if i == 0:
                                file.write(text)
                            else:
                                file.write(' ' + text +'\n')
                        else:
                            file.write(text + '\n')
            if len(targets) > 1:
                count += 1
                # if len(targets) > 2:
                # print(targets)
                # print(len(targets))
                # print(targets)
                # print(len(targets))
        print(f'{file_name}中例句的数量是：{fe_count}')
        print(f'{file_name}有构式的数量：{count}。')
        

