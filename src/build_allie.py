import gzip
import sys
import leveldb
import re

if __name__ == '__main__':
    # build bi-gram for a dictionary

    if len(sys.argv) < 3:
        print('specify allie file and database')
        sys.exit(0)

    dict_file = sys.argv[1]
    db_file = sys.argv[2]

    pattern = re.compile('[a-zA-Z]')

    with gzip.open(dict_file, 'rt', encoding='utf-8') as handler:

        db = leveldb.LevelDB(db_file)
        count = 0
        for line in handler:
            count += 1
            if count % 100000 == 0:
                print(count)
            tokens = line.strip().lower().split('\t')
            try:
                longform = tokens[5].lower()
                shortform = tokens[6].lower()
            except IndexError:
                continue
            
            try:
                shortforms = db.Get(longform.encode('utf-8')).decode('utf-8').split('|')
                if shortform in shortforms:
                    continue
                else:
                    shortforms.append(shortform)
                    shortforms_string = '|'.join(shortforms)
                    db.Put(longform.encode('utf-8'), shortforms_string.encode('utf-8'))
            except KeyError:
                db.Put(longform.encode('utf-8'), shortform.encode('utf-8'))
            

                


