from annotation.readers import AnnReader
import sys

class Evaluation(object):
    def __init__(self):
        pass

    @classmethod
    def get_containing_sentence(self, entity, sentences):
        for sentence in sentences:
            if (entity.start >= sentence.start and
                        entity.end <= sentence.end):
                return sentence.property.get('id')

    @classmethod
    def calculate(cls, user_set, golden_set):
        golden_num = len(golden_set)
        tp = len(golden_set & user_set)
        fp = len(user_set - golden_set)
        fn = len(golden_set - user_set)

        precision = tp * 1.0 / (tp + fp)
        recall = tp * 1.0 / (tp + fn)
        fscore = 2 * precision * recall / (precision + recall)

        print('all entities:', golden_num)
        print('precision:', precision)
        print('recall:', recall)
        print('f-score:', fscore)

        # print()
        # print('TP:')
        # for t in sorted(user_set & golden_set, key=lambda a: a[0]):
        #     print(t, relation_map[t])
        #
        # print()
        # print('FP:')
        # for t in sorted(user_set - golden_set, key=lambda a: a[0]):
        #     print(t, relation_map[t])
        #
        # print()
        # print('FN:')
        # for t in sorted(golden_set - user_set, key=lambda a: a[0]):
        #     print(t)
        #
        # print()
        # print('FN:')
        # print('\n'.join(set([str(t[0]) for t in golden_set - user_set])))


    @classmethod
    def evaluate(cls, user_data_path, golden_data_path, level):

        reader = AnnReader()

        user_data = reader.parse_folder(user_data_path, '.ann')
        golden_data = reader.parse_folder(golden_data_path, '.ann')

        user_keys = set(user_data.keys())
        golden_keys = set(golden_data.keys())
        user_entities = set()
        golden_entities = set()

        if user_keys != golden_keys:
            print('unmached keys between data sets', file=sys.stderr)
            # sys.exit(0)

        for pmid in golden_keys:
            gold_anno = golden_data[pmid]

            for entity in gold_anno.entities:
                if level == 'mention' or level == 'ending':
                    golden_entities.add((pmid,
                                         entity.category,
                                         entity.start,
                                         entity.end,
                                         entity.text))

                elif level == 'abstract':
                    golden_entities.add((pmid,
                                         entity.category,
                                         entity.text.lower()))
                    
        for pmid in user_keys:

            user_anno = user_data[pmid]

            for entity in user_anno.entities:
                if level == 'ending':
                    for gold_entity in golden_entities:
                        if entity.end == gold_entity[3]:
                            user_entities.add(gold_entity)
                            break
                            
                elif level == 'mention':
                    user_entities.add((pmid,
                                         entity.category,
                                         entity.start,
                                         entity.end,
                                         entity.text))

                elif level == 'abstract':
                    user_entities.add((pmid,
                                       entity.category,
                                       entity.text.lower()))


        cls.calculate(user_entities, golden_entities)
        
if __name__ == '__main__':
    user_data = sys.argv[1]
    gold_data = sys.argv[2]
    # Evaluation.evaluate(user_data, gold_data, 'mention')
    Evaluation.evaluate(user_data, gold_data, 'mention')