
with open('train_data_list_full.txt') as f:
    filenames = f.read().split()

with open('train_data_list.txt', 'w') as f:
    for fn in filenames:
        frame_num = int(fn.split('/')[-1])
        if('syn' in fn or frame_num % 7 == 1):
            f.write(fn + '\n')

with open('test_data_list_full.txt') as f:
    filenames = f.read().split()

with open('test_data_list.txt', 'w') as f:
    for fn in filenames:
        frame_num = int(fn.split('/')[-1])
        if('syn' in fn or frame_num % 7 == 1):
            f.write(fn + '\n')




