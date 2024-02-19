from argparse import ArgumentParser
from os import listdir

def listdir_ordered(dir):
    filenames = listdir(dir)
    filenames_split = [x.split('-') for x in filenames]
    code, _ = zip(*filenames_split)
    code = [int(x) for x in code]
    filenames = zip(code, filenames)
    filenames = sorted(filenames)
    code, filenames = zip(*filenames)
    return filenames


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-n', type=int, dest='index', help='index of environment dataset')
    args = parser.parse_args()