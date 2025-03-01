#print(165 // 12, 165 % 12)
#print(13 * 12 + 9)

list_1 = ['red', 'green', 'blue', 'black', 'white']
list_1.append('yellow')

for element in list_1:
    print(element)

s = ''
for i in range(1, 11):
    for j in range(1, 11):
        s += str(i*j) + '\t'
    s += '\n'
print(s)

