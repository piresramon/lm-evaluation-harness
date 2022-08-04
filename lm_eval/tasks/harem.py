"""
FaQuAD: Reading Comprehension Dataset in the Domain of Brazilian Higher Education
https://ieeexplore.ieee.org/document/8923668

The FaQuAD is a Portuguese reading comprehension dataset which follows the format 
of the Stanford Question Answering Dataset (SQuAD). As far as we know, FaQuAD is 
a pioneer Portuguese reading comprehension dataset with the SQuAD's challenging format.
"""
import re
from lm_eval import utils
import numpy as np
from typing import List
import collections
import logging
import os
import json
import datasets
from best_download import download_file
from math import exp
from lm_eval.base import rf, Task
from functools import partial
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import GPT2TokenizerFast

from tokenization import (
    Token,
    TokenizerWithAlignment,
    reconstruct_text_from_tokens,
)


_CITATION = """
@inproceedings{santos-etal-2006-harem,
    title = "{HAREM}: An Advanced {NER} Evaluation Contest for {P}ortuguese",
    author = "Santos, Diana  and
      Seco, Nuno  and
      Cardoso, Nuno  and
      Vilela, Rui",
    booktitle = "Proceedings of the Fifth International Conference on Language Resources and Evaluation ({LREC}{'}06)",
    month = may,
    year = "2006",
    address = "Genoa, Italy",
    publisher = "European Language Resources Association (ELRA)",
    url = "http://www.lrec-conf.org/proceedings/lrec2006/pdf/59_pdf.pdf",
    abstract = "In this paper we provide an overview of the first evaluation contest for named entity recognition in Portuguese, HAREM, which features several original traits and provided the first state of the art for the field in Portuguese, as well as a public-domain evaluation architecture.",
}
"""

ENTITIES_TOKENS = {
    'PESSOA': 'pessoa',
    'ORGANIZACAO': 'organização',
    'LOCAL': 'local',
    'TEMPO': 'tempo',
    'VALOR': 'valor',
    'OUTRO': 'outro',
}
# ENTITIES_TOKENS = {
#     'PESSOA': 'PESSOA',
#     'ORGANIZACAO': 'ORGANIZAÇÃO',
#     'LOCAL': 'LOCAL',
#     'TEMPO': 'TEMPO',
#     'VALOR': 'VALOR',
#     'OUTRO': 'OUTRO',
# }


PATTERNS_REPLACES = [
    #(r'\s*\n+\s*', r'\n'),
    (r'\s*\n+\s*', r' '), ## changing \n to space
    #(r'(\s)\1+', r'\1'),
    (r'(\s)\1+', r' '), ## changing \n to space
    (r'^\s+', r''),
]


apply_regex = lambda pattern, replace, text: re.sub(pattern, replace, text)


LOGGER = logging.getLogger(__name__)


NETag = collections.namedtuple("NETag", ['doc_id',
                                         'entity_id',
                                         'text',
                                         'type',
                                         'start_position',
                                         'end_position',
                                         'start_offset', # ramon
                                         'end_offset', # ramon
                                         ])

                                         
class Example(object):
    """
    A single training/test example for NER training.
    """

    def __init__(self,
                 doc_id: int,
                 orig_text: str,
                 doc_tokens: List[Token],
                 tags: List[NETag],
                 labels: List[str],
                 ):
        self.doc_id = doc_id
        self.orig_text = orig_text
        self.doc_tokens = doc_tokens
        self.tags = tags
        self.labels = labels
        self.explanation = ''  ## ramon

        for token in doc_tokens:
            token._example = self

    def __str__(self):
        return repr(self)

    def __repr__(self):
        s = ('doc_id: {}\n'
             'orig_text:{}\n'
             'doc_tokens: {}\n'
             'labels: {}\n'
             'tags: {}\n'
             'explanation: {}\n').format(self.doc_id, self.orig_text, self.doc_tokens,  ## ramon: add explanation
                                  self.labels, self.tags, self.explanation)
        return s

class HAREM(Task):
    VERSION = 1
    DATASET_PATH = "data/harem/"
        
    tokenizer_with_alignment = TokenizerWithAlignment()

    use_explanation = False
    html_based = True

    # temporary
    tokenizer = GPT2TokenizerFast.from_pretrained('/home/ramon.pires/git/gptimbau/tokenizer')

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        if os.path.exists(self.DATASET_PATH):
            print(f"Reusing dataset faquad ({self.DATASET_PATH})")
        else:
            download_file('https://raw.githubusercontent.com/neuralmind-ai/portuguese-bert/6ef2e318e4b625326fbd2406adeca39527134607/ner_evaluation/data/FirstHAREM-selective-train.json', local_directory=self.DATASET_PATH)
            download_file('https://raw.githubusercontent.com/neuralmind-ai/portuguese-bert/6ef2e318e4b625326fbd2406adeca39527134607/ner_evaluation/data/MiniHAREM-selective.json', local_directory=self.DATASET_PATH)

            self._generate_clean_dataset(self.DATASET_PATH + 'FirstHAREM-selective-train.json', self.DATASET_PATH + 'train.json')
            self._generate_clean_dataset(self.DATASET_PATH + 'MiniHAREM-selective.json', self.DATASET_PATH + 'dev.json')

        self.dataset = {}
        self.dataset['train'] = self.read_examples(self.DATASET_PATH + 'train.json', True)
        self.dataset['validation'] = self.read_examples(self.DATASET_PATH + 'dev.json', True)

        # temporary code to sort the validation set by number of tokens
        indexes = [100, 45, 63, 96, 125, 40, 90, 109, 112, 119, 88, 69, 32, 24, 101, 34, 5, 89, 46, 43, 39, 51, 60, 118, 93, 13, 105, 59, 28, 31, 21, 14, 35, 61, 111, 36, 83, 52, 74, 103, 72, 22, 57, 121, 50, 126, 3, 27, 113, 98, 91, 12, 16, 66, 86, 41, 44, 67, 107, 23, 9, 115, 47, 6, 68, 71, 64, 106, 77, 95, 0, 108, 114, 11, 37, 82, 56, 123, 18, 75, 76, 92, 127, 17, 26, 25, 29, 54, 122, 49, 15, 104, 62, 124, 53, 94, 33, 30, 120, 2, 87, 70, 110, 102, 42, 7, 55, 97, 19, 4, 84, 1, 79, 73, 10, 48, 81, 78, 65, 116, 58, 38, 99, 117, 85, 80, 20, 8]
        self.dataset['validation'] = np.array(self.dataset['validation'])[indexes].tolist()
        self.dataset['validation'] = self.dataset['validation'][:10] # only the first 10
        # for s in self.dataset['validation']:
        #     # print(len(s.orig_text))
        #     print(len(self.tokenizer.tokenize(s.orig_text)))

    def _clean_text(self, input_text: str):
        """This function applies the regex and generates a map that will be used to update 
        the start and end offsets.

        First we extract all the spans (e.g, <re.Match object; span=(396, 398), match='\n\n'>),
        adding them in a list. After, we get each span and update the map. As the patterns 
        can have intersections, it is fundamental to remove the repeated and intersected 
        spans (keep the largest one); otherwise the map will be incorrect.
        """
        input_text = input_text.replace('\t', ' ')
        ocr_map = np.zeros(len(input_text), dtype=np.int64)
        output_text = input_text

        spans = []
        for pattern, replace in PATTERNS_REPLACES:

            output_text = apply_regex(pattern, replace, output_text)

            gen = re.finditer(pattern, input_text, flags=0)
            for g in gen:
                span = g.span()
                spans.append((span, replace))

        last_span = (0, 0)
        last_diff = 0
        for span_tuple in sorted(list(set(spans))):
            span, replace = span_tuple

            replace_size = len(replace.encode().decode('unicode_escape'))  # raw string
            diff = (span[1] - span[0]) - replace_size
            if diff <= 0:
                continue

            # If the spans have intersection, we ignore the last one
            # the current one is contained into the last one: skip it
            if span[0] >= last_span[0] and span[1] <= last_span[1]:
                #print(f'Intersection: {last_span} and {span}')
                continue
            # there is an interception, but the current one is larger
            if span[0] < last_span[1] and span[1] > last_span[1]:
                #print(f'Intersection: {last_span} and {span}')
                diff = abs(diff - last_diff)
            last_span = span
            last_diff = diff

            # between span[0] and span[i], each original position will move back span[0] positions
            for i in range(span[0], span[1]):
                ocr_map[i] -= i - span[0]
            # after span[1], each original position will move back the difference of found pattern size and replace size
            ocr_map[span[1]:] -= diff

        return output_text, ocr_map.tolist()

    def _generate_clean_dataset(self, input_file: str, output_file: str):
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        all_data = []

        for document in input_data:

            # clean the document and generate the map
            text, map = self._clean_text(document['doc_text'])

            entities = []

            for entity in document['entities']:
                entity['start_offset'] += map[entity['start_offset']]
                entity['end_offset'] += map[entity['end_offset']]
                entities.append(entity)

            document['doc_text'] = text
            document['entities'] = entities
            all_data.append(document)

        with open(output_file, 'w') as f:
            json.dump(all_data, f)

    ### POR ENQUANTO O CÓDIGO ESTÁ SENDO COPIADO DE PORTUGUESE-BERT/NER_EVALUATION/PREPROCESSING.PY
    ## IF NO CHANGE BECOME NECESSARY, CONSIDER THE POSSIBILITY OF INCLUDING THE REPO IN REQUIREMENTS
    def read_examples(self, input_file: str,
                  is_training: bool,
                  scheme: str = 'BIO',
                  ) -> List[Example]:
        """Read a JSON file into a list of Examples.

        The JSON file should contain a list of dictionaries, one dict per input
        document. Each dict should have the following entries:

        doc_id: an example unique identifier (for debugging).
        doc_text: the document text.
        entities: a list of dicts of named entities contained in `doc_text`.
            Each entity dict should have the following entries:

                entity_id: an identifier for the entity (debugging purposes).
                label: the named entity gold label.
                start_offset: start char offset of the entity in `doc_text`.
                end_offset: **exclusive** end char offset of the entity in
                    `doc_text`.
                text: the named entity text. It should be equal to the slice of the
                    document text using `start_offset` and `end_offset`, e.g.,
                    `doc_text[start_offset:end_offset]`.
        """
        scheme = scheme.upper()
        if scheme not in ['BIO', 'BILUO']:
            raise ValueError("Invalid tagging scheme `{}`.".format(scheme))

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        examples = []

        for document in input_data:
            doc_text = document["doc_text"]
            doc_id = document["doc_id"]

            # Perform whitespace and punctuation tokenization keeping track of char
            # alignment (char_to_word_offset)
            doc_tokens, char_to_word_offset = self.tokenizer_with_alignment(doc_text)
            labels = ["O"] * len(doc_tokens)
            tags = []

            def set_label(index, tag):
                if labels[index] != 'O':
                    LOGGER.warning('Overwriting tag %s at position %s to %s',
                                labels[index], index, tag)
                labels[index] = tag

            if is_training:
                for entity in document["entities"]:
                    entity_id = entity["entity_id"]
                    entity_text = entity["text"]
                    entity_type = entity["label"]
                    start_token = None
                    end_token = None

                    entity_start_offset = entity["start_offset"]
                    entity_end_offset = entity["end_offset"]
                    start_token = char_to_word_offset[entity_start_offset]
                    # end_offset is NOT inclusive to the text, e.g.,
                    # entity_text == doc_text[start_offset:end_offset]
                    end_token = char_to_word_offset[entity_end_offset - 1]

                    assert start_token <= end_token, \
                        "End token cannot come before start token."
                    reconstructed_text = reconstruct_text_from_tokens(
                        doc_tokens[start_token:(end_token + 1)])
                    # assert entity_text.strip() == reconstructed_text, \
                    #     "Entity text and reconstructed text are not equal: %s != %s" % (
                    #         entity_text, reconstructed_text)
                    if entity_text.strip() != reconstructed_text:
                        print("Entity text and reconstructed text are not equal: %s != %s" % (
                            entity_text, reconstructed_text))

                    if scheme == 'BILUO':
                        # BILUO scheme
                        if start_token == end_token:
                            tag = 'U-' + entity_type
                            set_label(start_token, tag)
                        else:
                            for token_index in range(start_token, end_token + 1):
                                if token_index == start_token:
                                    tag = 'B-' + entity_type
                                elif token_index == end_token:
                                    tag = 'L-' + entity_type
                                else:
                                    tag = 'I-' + entity_type

                                set_label(token_index, tag)

                    elif scheme == 'BIO':
                        # BIO scheme
                        for token_index in range(start_token, end_token + 1):
                            if token_index == start_token:
                                tag = 'B-' + entity_type
                            else:
                                tag = 'I-' + entity_type
                            set_label(token_index, tag)

                    entity = NETag(
                        doc_id,
                        entity_id,
                        entity_text,
                        entity_type,
                        start_token,
                        end_token,
                        entity_start_offset,  # ramon
                        entity_end_offset, # ramon
                    )
                    tags.append(entity)

            example = Example(
                doc_id=doc_id,
                orig_text=doc_text,
                doc_tokens=doc_tokens,
                tags=tags,
                labels=labels)
            examples.append(example)

        return examples

    ### POR ENQUANTO O CÓDIGO ESTÁ SENDO COPIADO DE T5-for-new/src/models/evaluate.py
    ## IF NO CHANGE BECOME NECESSARY, CONSIDER THE POSSIBILITY OF INCLUDING THE REPO IN REQUIREMENTS
    def get_entities_from_tokens(self, tokens: List[str], entities_tokens: List[str],
                                length: int = 0, fill_token: str = 'O') -> List[str]:

        end_seq_tokens = ['<|endoftext|>']

        inverse_entities_tokens = { v: k for k, v in entities_tokens.items() }

        sequence_entities = []  # will save all the entities
        current_entity = []  # will save current entity
        open_entity = None  # entity class that is open

        for token in tokens:
            token = token.text

            if self.html_based:
                ##### SOLUTION THAT USES HTML-FORMAT
                if token[0] == '<' and token[-1] == '>':
                    entity = token[1:-1]  # remove <,>
                    if entity in inverse_entities_tokens or entity[1:] in inverse_entities_tokens:
                        blabel = ilabel = 'O'
                        if entity.startswith('/'):  # closing entity
                            entity = inverse_entities_tokens[entity[1:]]
                            # the closing entity is matching with the open entity?
                            if entity == open_entity:  
                                blabel = f'B-{entity}'
                                ilabel = f'I-{entity}'
                        else:  # opening entity
                            open_entity = inverse_entities_tokens[entity]
                        _len = len(current_entity)
                        if _len > 0:
                            sequence_entities += [blabel] + [ilabel] * (_len - 1)
                        current_entity.clear()
                    else:
                        current_entity.append(token)
                elif token in end_seq_tokens:
                    break
                else:
                    current_entity.append(token)
            else:
                #### SOLUTION THAT USES <outro>
                if token[0] == '<' and token[-1] == '>' and token[1:-1] in inverse_entities_tokens:
                    entity = token[1:-1]  # remove <,>
                    # put the entity back to upper letter
                    entity = inverse_entities_tokens[entity]
                    if entity == 'OUTRO':
                        blabel = ilabel = 'O'
                    else:
                        blabel = f'B-{entity}'
                        ilabel = f'I-{entity}'
                    _len = len(current_entity)
                    if _len > 0:  # RAMON FIXING: must include the entity only if there are tokens in current entity
                        sequence_entities += [blabel] + [ilabel] * (_len - 1)
                    current_entity.clear()
                elif token in end_seq_tokens:
                    break
                else:
                    current_entity.append(token)

        if length > 0:
            seq_len = len(sequence_entities)
            if seq_len > length:
                sequence_entities = sequence_entities[:length]
            elif seq_len < length:
                sequence_entities = sequence_entities + \
                    [fill_token] * (length - seq_len)
        return sequence_entities

    def _add_entities_into_text(self, text):
        pass

    def _text_to_target(self, text):
        return self._add_entities_into_text(text)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        #return 'Título: ' + doc['title'] + '\n\n' + 'Contexto: ' + doc['context'] + '\n\n' + 'Pergunta: ' + doc['question'] + '\n\n' + 'Resposta:'
        #return f"Entrada:\n{doc.orig_text}\n\nSaída:"
        #return f"Entrada:\n{doc.orig_text.strip()}\n\nSaída:"  # move the strip to preprocessing??
        # return f"Entrada: {doc.orig_text.strip()}\n\nExplicação: {doc.explanation}\n\nSaída:"  # move the strip to preprocessing??  # add explanation
        if not self.use_explanation:
            return f"Entrada: {doc.orig_text.strip()}\n\nSaída:"  # move the strip to preprocessing??

        if doc.explanation == '':
            # test set
            return f"Entrada: {doc.orig_text.strip()}\n\nExplicação:"
        else:
            # few-shot context
            return f"Entrada: {doc.orig_text.strip()}\n\nExplicação: {doc.explanation} Saída:"

    def doc_to_target(self, doc):
        # TODO: implement here the code that adds the entities ([OUTRO], [TEMPO], ...)
        #answer = self._add_entities_into_text(doc)

        # #####
        # # Solução para contruir texto com entidades a partir do texto original e anotações
        # # de entidades (offsets)
        # #####
        # answer = doc.orig_text
        # offset = 0
        # last_end_offset = 0
        # for tag in doc.tags: #zip(doc.doc_tokens, doc.tags):
        #     start_offset = tag.start_offset
        #     end_offset = tag.end_offset
        #     label = tag.type

        #     # include [Outro] before start_offset
        #     #if start_offset != last_end_offset:
        #     if start_offset > last_end_offset + 1:
        #         tagg = f"<OUTRO> "
        #         answer = f"{answer[:start_offset + offset]}{tagg}{answer[start_offset + offset :]}"
        #         offset += len(tagg) 

        #     tagg = f" <{label}>"
        #     answer = f"{answer[:end_offset + offset]}{tagg}{answer[end_offset + offset :]}"
        #     offset += len(tagg) 

        #     last_end_offset = end_offset


        #########
        ## Solução nova para contruir answer a partir do doc_tokens e tag.
        ## Com esta solução, não teremos problemas porque estamos seguindo 
        ## os mesmos critérios de tokenização que será aplicado no pós-processamento,
        ## ao converter a sequencia gerada pelo modelo em uma lista de entidades.
        #########
        answer = ''

        # if the text does not have entities, return the original text
        if doc.tags == []:
            #return '\n' + doc.orig_text
            return ' ' + doc.orig_text

        tags = iter(doc.tags)
        tag = next(tags)
        recent_entity = True
        has_tags = True

        for idx, token in enumerate(doc.doc_tokens):

            # The current token is the start of a new entity.
            # If a entity was not included recently, add <OUTRO>
            if has_tags and idx == tag.start_position:
                if self.html_based:
                    answer += f'<{ENTITIES_TOKENS[tag.type]}> '
                elif not recent_entity:
                    # answer += '<OUTRO> '
                    answer += '<outro> '

            # Concatenate the token text
            answer += token.text
            recent_entity = False

            # The current token is the end of the entity.
            # The <TYPE> must concatenated here.
            if has_tags and idx == tag.end_position:
                if self.html_based:
                    answer += f' </{ENTITIES_TOKENS[tag.type]}>'  # new html-based format
                else:
                    answer += f' <{ENTITIES_TOKENS[tag.type]}>'  # original format
                recent_entity = True
                try:
                    tag = next(tags)
                except:
                    has_tags = False

            answer += token.tail

        # print(doc.orig_text)
        # print('------------------')
        # print(answer)
        # print('------------------')
        # print('------------------')
        # print(doc)

        # ### sanity test
        doc_tokens, _ = self.tokenizer_with_alignment(answer)
        labels = self.get_entities_from_tokens(doc_tokens, entities_tokens=ENTITIES_TOKENS, length=len(doc.labels))
        if labels != doc.labels: #  and doc_tokens[0].text == 'Mosteiro': # and doc_tokens[0].text == 'confrontos':
            idx = 0
            for a, b in zip(labels, doc.labels):
                if a != b:
                    break
                idx += 1

            if True: #len(doc_tokens) < 2000:
                print(f'decoded entities: \n{labels[idx-10:]}\ntarget entities: \n{doc.labels[idx-10:]}\n\n')
                #print(doc_tokens[idx-10:idx+10],'\n')
                print(doc.doc_tokens[idx-10:idx+10],'\n')
                print(doc_tokens)
                print(doc)
                #print(answer, '\n\n', doc.orig_text, '\n\n')

        #return "\n" + answer
        return " " + answer

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        n = len(self.tokenizer.tokenize(ctx))
        #print(f'--------\nINPUT ({n} tokens):\n--------\n...{ctx}...\n\n')
        continuation = rf.greedy_until(ctx, ['\n'])
        #continuation = rf.greedy_until(ctx, ['<|endoftext|>'])
        return continuation

    @classmethod
    def precision(cls, items):
        preds, golds = zip(*items)
        return precision_score(golds, preds, average='micro')

    @classmethod
    def recall(cls, items):
        preds, golds = zip(*items)
        return recall_score(golds, preds, average='micro')

    @classmethod
    def f1(cls, items):
        preds, golds = zip(*items)
        return f1_score(golds, preds, average='micro')
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """        
        #doc_tokens, _ = self.tokenizer_with_alignment(results[0])
        response = results[0]
        if self.use_explanation:
            sub = response.split('Saída: ')
            explanation = sub[0]
            response = sub[1]
        doc_tokens, _ = self.tokenizer_with_alignment(response) #results[0].split('Saída: ')[1]) ##### in case of explanation, get only the response itself.
        predictions = self.get_entities_from_tokens(doc_tokens, entities_tokens=ENTITIES_TOKENS, length=len(doc.labels))

        references = doc.labels

        print(f'[EXAMPLE]:\ngold: {self.doc_to_target(doc)}\n-----------\npred: {response}')
        if self.use_explanation:
            print(f'-----------\nexplanation: {explanation}')
        print(f'[EXAMPLE]:\ngold: {doc.labels}\n-----------\npred: {predictions}\n')

        return { 
            'precision': (predictions, references),
            'recall': (predictions, references),
            'f1': (predictions, references),
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        return { 
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return { 
            'precision': True,
            'recall': True,
            'f1': True,
        }

    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        # # Visualizing lenght (tokens) and entities distribution
        # #with open(self.DATASET_PATH + 'train.json', "r", encoding='utf-8') as reader:
        # with open(self.DATASET_PATH + 'dev.json', "r", encoding='utf-8') as reader:
        #     input_data = json.load(reader)
        # stats = []
        # for i, data in enumerate(input_data):
        #     text = data['doc_text']

        #     from collections import Counter
        #     c = Counter([entity['label'] for entity in data['entities']])

        #     c['n_entidades'] = sum(c.values())
        #     c['size'] = len(text.split())
        #     c['n_tokens'] = len(self.tokenizer.tokenize(text))
        #     c['index'] = i
        #     stats.append(c)

        # # print(indexes)
        # stats.sort(key=(lambda x: x['n_tokens']))
        # for s in stats:
        #     print(s)
        # print([ s['index'] for s in stats ])
        # sizes = [s['n_tokens'] for s in stats]
        # entidades = [s['n_entidades'] for s in stats]
        # print(f'MEAN OF NUM TOKENS: {np.mean(sizes)}')
        # print(f'MEAN OF NUM TOKENS (first 5): {np.mean(sizes[:5])}')
        # print(f'MEAN OF ENTITIES: {np.mean(entidades)}')

        # Indexes of all examples sorted by number of tokens
        indexes = [17, 120, 47, 20, 86, 66, 78, 11, 103, 82, 25, 46, 81, 87, 
            114, 94, 36, 35, 56, 55, 119, 57, 19, 40, 0, 42, 100, 67, 26, 13, 
            90, 43, 32, 88, 75, 14, 44, 92, 48, 50, 1, 99, 12, 2, 107, 61, 74, 
            108, 58, 73, 62, 7, 16, 28, 60, 72, 76, 91, 104, 109, 89, 41, 27, 
            18, 65, 105, 30, 4, 59, 63, 106, 29, 15, 38, 102, 45, 117, 118, 
            80, 95, 5, 79, 9, 34, 51, 110, 53, 68, 83, 54, 85, 98, 3, 96, 111, 
            97, 112, 113, 23, 21, 77, 115, 49, 93, 70, 64, 31, 33, 69, 24, 71, 
            84, 8, 6, 10, 52, 37, 22, 39, 101, 116]
        
        # ADD MANUALLY THE EXPLANATION
        for i, idx in enumerate(indexes[:k]):
            document = self._training_docs[idx]
            if i == 0:
                explanation = 'Clive Cussler é uma pessoa. Dirk Pitt é uma pessoa. US$ 14 milhões é um valor. Simon & Schuster é uma organização.'
            if i == 1:
                explanation = 'PCP é uma organização. Braga é um local. Comissão Concelhia de Braga do PCP é uma organização. Centro de Trabalho do PCP é um local. 78 anos é um valor. 6 é tempo.'
            if i == 2:
                explanation = '54 é um valor. Cerca de 40 é um valor. Rio é um local. 30 é um valor. Guarani é uma organização. Corinthians é uma organização. Campinas é um local. M.F.B é uma pessoa. 16 é um valor. Wagner Silva é uma pessoa. 23 é um valor.'
            if i == 3:
                explanation = 'R$ 399,00 é um valor. 6 é um valor. R$ 33,00 é um valor.'
            if i == 4:
                explanation = 'Foz é um local. Porto é um local. 02h30 horas é tempo. 05h20 é tempo. Porto Fino é uma organização. Padrão é um local. 44 é um valor.'
            
            #print(self.doc_to_target(document), explanation)
            document.explanation = f'{explanation}'
            self._training_docs[idx] = document


        # return rnd.sample(self._training_docs, k)
        return [ self._training_docs[i] for i in indexes[:k] ]
        
    @utils.positional_deprecated
    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.
        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = description + "\n\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = ''
            for i, doc_ex in enumerate(fewshotex):
                labeled_examples += f'Exemplo {i+1}:\n\n'
                labeled_examples += self.doc_to_text(doc_ex) + self.doc_to_target(doc_ex)
                labeled_examples += '\n\n\n'
            labeled_examples += f'Exemplo {len(fewshotex) + 1}:\n\n'

        example = self.doc_to_text(doc)
        return description + labeled_examples + example
