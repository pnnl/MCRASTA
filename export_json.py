
import json
import pprint


def export_to_json():

    path = 'foo.json'



    payload = {'sample': path,
               'section_ID': 'asdfasoasdf',
               'thislist' ['foo0', 'bar', ['another', 'list']]}


    with open(path, 'w') as wfile:
        json.dump(payload, wfile)



def read_from_json():
    with open('foo.json', 'r') as rfile:
        js = json.load(rfile)
        pprint.pprint(js)
        print(type(js))
        print('section id read from file', js.get('section_ID'))
        print('output path read from file: ', js.get('output_path', 'No output_path defined'))
        print('its the same as this ', js['section_ID'])

if __name__ == '__main__':
    export_to_json()
    read_from_json()