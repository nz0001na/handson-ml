# d = dict()
# print(dir(d))

student_scores = {"John": 85, "Ryan": 78, "Emma": 92}
for ppp in student_scores:
    print(ppp)




# scores_copy = student_scores.copy()
items = student_scores.items()
# for k, v in items:
#     print(k)
#     print(v)

# print(list(items))  # Out

va = student_scores.popitem()
print(va)


student_scores.setdefault('Apple', 98)
p = student_scores.setdefault('John', 33)
print(p)

print(student_scores)
# dd = {'Apple': 10000, 'Banana': 999999}
dd = [('Apple', 10000),('Banana', 999999)]
student_scores.update(dd)
print(student_scores)

print(list(student_scores))
print(list(student_scores.keys()))
print(list(student_scores.values()))