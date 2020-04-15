print('This is the first value')
test = data_df.iloc[[0],[0]]
print(test)

print(str('#'+test))
nu = str('#'+test)
print('this is the end')


# workbook = load_workbook(filename="Hashtags.xlsx")
# workbook.sheetnames
# sheet = workbook.active

# print(sheet.cell(1,1))

v = np.array([1, 'test', 4, 'test', 2, 0, 1.5, 1, 4, 3])
v[v == 'test'] = 'poep'
print(v)