import copy

old_list = [['b','c'], [2, 2, 2], [3, 3, 3]]
new_list = copy.deepcopy(old_list)

old_list.append([4, 4, 4])
new_list.append(999)
# new_list[1][1] = 'a'
# old_list[0] = ['a', 'b']
# old_list[2] = 5
# new_list[1] = 'bbbb'
old_list[1].append('7')
new_list[0].append('f')
new_list[0][0] = 'B'
old_list[1][0] = 'ZZ'

old_list[1] = 'NNNN'


old_list[3].append(888)
new_list[3] = [999, 888]
print("Old list:", old_list)
print("New list:", new_list)

print(new_list.count([2,2]))