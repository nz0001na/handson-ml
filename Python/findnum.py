import re

file_name = 'regex_sum_1836031.txt'
# file_name = 'regex_sum_42.txt'
try:
    fh = open(file_name)
except:
    print("File not exists!")
    quit()

# all_nums = []
sum = 0
for line in fh:
    line = line.rstrip()
    nums = re.findall('[0-9]+', line)
    for n in nums:
        sum += int(n)

print('Sum=', sum)
