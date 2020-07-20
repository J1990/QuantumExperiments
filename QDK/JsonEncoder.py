#############################################################################################################################
######### https://stackoverflow.com/questions/54370322/how-to-limit-the-number-of-float-digits-jsonencoder-produces #########
#############################################################################################################################

import json

class MyJsonEncoder(json.JSONEncoder):
    def iterencode(self, obj):
        if isinstance(obj, float):
            yield format(obj, '.3f')
        elif isinstance(obj, dict):
            last_index = len(obj) - 1
            yield '{'
            i = 0
            for key, value in obj.items():
                yield '"' + key + '": '
                for chunk in MyJsonEncoder.iterencode(self, value):
                    yield chunk
                if i != last_index:
                    yield ", "
                i+=1
            yield '}'
        elif isinstance(obj, list):
            last_index = len(obj) - 1
            yield "["
            for i, o in enumerate(obj):
                for chunk in MyJsonEncoder.iterencode(self, o):
                    yield chunk
                if i != last_index: 
                    yield ", "
            yield "]"
        else:
            for chunk in json.JSONEncoder.iterencode(self, obj):
                yield chunk

def dump(data, json_path, indent):
    # write d using custom encoder
    with open(json_path, 'w') as f:
        json.dump(data, f, cls = MyJsonEncoder)

    # load output into new_d
    with open(json_path, 'r') as f:
        new_d = json.load(f)

    # write new_d out using default encoder
    with open(json_path, 'w') as f:
        json.dump(new_d, f, indent=indent)
