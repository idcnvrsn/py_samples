import argparse
from pprint import pprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明', formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@')
    parser.add_argument('filepath', default="snippet.txt", help='vscodeのスニペット化したいコードが記載されたテキストファイル')
    parser.add_argument('name', default='snippet name', help='snippetのname')
    parser.add_argument('prefix', default='snippet prefix', help='snippetのprefix(検索に使われる)')
    parser.add_argument('-d', '--description', default='snippet description', help='snippetの説明')
    args = parser.parse_args()
#    pprint(args.__dict__)

    with open(args.filepath, 'r') as f:
        lines = f.read()
        lines=lines.splitlines()
        
        new_string ="\""+args.name+"\": {"
        print(new_string)
        new_string="     \"prefix\":\""+args.prefix+"\","
        print(new_string)
        new_string="     \"body\": ["
        print(new_string)

        for line in  lines:
            new_string="          \""+line+"\","
            print(new_string)

        new_string="     ],"
        print(new_string)
        new_string="     \"description\":\""+args.description+"\","
        print(new_string)
        new_string="},"
        print(new_string)