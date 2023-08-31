import re

# pattern = r'\d+'
# string = 'I have 42 apples and 7 oranges'
# string2 = 'sss 56 bbbbbbbb 78 nnnnnf456...???'
#
# common_pattern = re.compile(pattern, flags=0)
# print(common_pattern.findall(string))
# print(common_pattern.findall(string2))

# txt = "The rain in Spain"
# y = re.findall('aip', txt)
# x = re.search("aip", txt)
# print(x)
# print(y)


s = 'From stephen.marquard@uct.ac.za Sat Jan  5 09:14:16 2008'
l = re.findall('\S+?@\S+?',s)
print(l)