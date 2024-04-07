import json


file_name1 = r'D:\CCL\CFN2.1\cfn-train.json'
file_name2 = r'D:\CCL\CFN2.1\cfn-test-A.json'
file_name3 = r'D:\CCL\CFN2.1\cfn-dev.json'
file_names = [file_name1, file_name2, file_name3]



for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as file:
        
        datas = json.load(file)
        
        
        fe_count = len(datas) 
        count = 0
        
        for data in datas:
        
            targets = data['target']
            # print(targets)
            # print(len(targets))
            if len(targets) > 1:
                count += 1
                if len(targets) > 2:
                    print(targets)
                    print(len(targets))
                # print(targets)
                # print(len(targets))
        print(f'{file_name}中例句的数量是：{fe_count}')
        print(f'{file_name}有构式的数量：{count}。')
        

