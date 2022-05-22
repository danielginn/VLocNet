import json

for i in range(1230):
    filepath = '.\\BlenderNUbotsField\\Location25\\' + str(i) + '.json'
    with open(filepath, 'r') as f:
        data = json.load(f)
        data2 = {'position':data['position'], 'orientation':data['heading']}

    with open(filepath, 'w') as f:
        json.dump(data2, f)
