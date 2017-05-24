import argparse
import json
import re


def main():
    parser = argparse.ArgumentParser(description='Create a JSON from an transformix output points file')
    parser.add_argument('--in', metavar='IN', dest='input', type=str, required=True, help='The output points to parse')
    args = parser.parse_args()

    output = {'inputindex': {},
              'inputpoint': {},
              'outputindex': {},
              'outputpoint': {},
              'deformation': {}}

    with open(args.input) as fin:
        for line in fin:
            match = re.match('Point\s+(?P<id>\d+)\s+; InputIndex = \[(?P<inindex>[\s\d]+)\]\s+; InputPoint = \[(?P<inpoint>[\s\d\.-]+)\]\s+; OutputIndexFixed = \[(?P<outindex>[-\s\d]+)\]\s+; OutputPoint = \[(?P<outpoint>[\s\d\.-]+)\]\s+; Deformation = \[(?P<deform>[\s\d\.-]+)\]', line)
            if match:
                sample_id = match.group('id')
                output['inputindex'][sample_id] = [int(x) for x in match.group('inindex').split()]
                output['inputpoint'][sample_id] = [float(x) for x in match.group('inpoint').split()]
                output['outputindex'][sample_id] = [int(x) for x in match.group('outindex').split()]
                output['outputpoint'][sample_id] = [float(x) for x in match.group('outpoint').split()]
                output['deformation'][sample_id] = [float(x) for x in match.group('deform').split()]
            else:
                raise ValueError('Could not parse line: {}'.format(line))

    print('__INPUTINDEX__ = {}'.format(json.dumps(output['inputindex'])))
    print('__INPUTPOINT__ = {}'.format(json.dumps(output['inputpoint'])))
    print('__OUTPUTINDEX__ = {}'.format(json.dumps(output['outputindex'])))
    print('__OUTPUTPOINT__ = {}'.format(json.dumps(output['outputpoint'])))
    print('__DEFORMATION__ = {}'.format(json.dumps(output['deformation'])))


if __name__ == '__main__':
    main()
