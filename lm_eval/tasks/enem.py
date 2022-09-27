"""
University Entrance Exam as a Guiding Test for Artificial Intelligence
https://www.ime.usp.br/~ddm/project/enem/ENEM-GuidingTest.pdf

The ENEM Challenge consists in designing an autonomous system that matches the 
performance of a human students on the exam. The overall goal is to foster and 
evaluate the development of Artificial Intelligence techniques that have good 
performance on complex cognitive tasks, not particularly designed for AI systems. 
In addition, this challenge aims to promote and give more visiblity to the 
development of NLP tools for Brazilian Portuguese.

Homepage: https://www.ime.usp.br/~ddm/project/enem
"""
import collections
from io import BytesIO
import os
import re
from urllib.request import urlopen
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@InProceedings{ ENEM-Challenge,
    author={Silveira, Igor Cataneo and Mau\'a, Denis Deratani},
    booktitle={Proceedings of the 6th Brazilian Conference on Intelligent Systems},
    series={BRACIS},
    title={University Entrance Exam as a Guiding Test for Artificial Intelligence},
    pages={426--431},
    year={2017}
}
"""


PATTERNS_REPLACES = [
    (r'\s*\n+\s*', r' '), ## changing \n to space
    (r'(\s)\1+', r' '), ## changing \n to space
    (r'^\s+', r''),
]

apply_regex = lambda pattern, replace, text: re.sub(pattern, replace, text)


class ENEM(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = 'data/enem'
    DATASET_NAME = None

    os.makedirs(DATASET_PATH, exist_ok=True)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):

        # download and unpack the dataset
        URL = "https://www.ime.usp.br/~ddm/project/enem/ENEMdataset.zip"
        http_response = urlopen(URL)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=self.DATASET_PATH)

        enem_stats = {
            '2009-1.xml':    {'TC_only': 0, 'total': 45}, #
            '2009-2.xml':    {'TC_only': 0, 'total': 40}, #
            '2010-1.xml':    {'TC_only': 16, 'total': 45},
            '2010-2.xml':    {'TC_only': 25, 'total': 40},
            '2011-1.xml':    {'TC_only': 12, 'total': 45},
            '2011-2.xml':    {'TC_only': 21, 'total': 40},
            '2012-1.xml':    {'TC_only': 21, 'total': 45},
            '2012-2.xml':    {'TC_only': 23, 'total': 40},
            '2013-1.xml':    {'TC_only': 19, 'total': 45},
            '2013-2.xml':    {'TC_only': 23, 'total': 40},
            '2014-1.xml':    {'TC_only': 13, 'total': 45},
            '2014-2.xml':    {'TC_only': 22, 'total': 40},
            '2015-1.xml':    {'TC_only': 22, 'total': 45},
            '2015-2.xml':    {'TC_only': 23, 'total': 40},
            '2016-1.xml':    {'TC_only': 0, 'total': 45}, #
            '2016-2.xml':    {'TC_only': 0, 'total': 40}, #
            '2016_2_-1.xml': {'TC_only': 0, 'total': 45}, #
            '2016_2_-2.xml': {'TC_only': 0, 'total': 40}, #
            '2017-1.xml':    {'TC_only': 0, 'total': 45}, #
            '2017-2.xml':    {'TC_only': 0, 'total': 40}, #
        }

        self.dataset = collections.defaultdict(list)

        for data_name in enem_stats:
            _, total = enem_stats[data_name].values()

            # get the documents
            documents = self._parse_json(os.path.join(self.DATASET_PATH, data_name), first_n=total)

            if data_name in ['2009-1.xml', '2009-2.xml', '2016-1.xml', '2016-2.xml', '2006_2_-1.xml', '2006_2_-2.xml', '2017-1.xml', '2017-2.xml']:
                self.dataset['train'] += documents
            else:
                self.dataset['test'] += documents

    def _parse_json(self, path, filters=None, first_n=45, verbose=True):
        tree = ET.parse(path)
        root = tree.getroot()

        if filters is None:
            filters = {
                'EK': "No",
                'IC': "No",
                'TC': "Yes",
                'IMG': None,
            }

        def ignore_question(filters):
            for k,v in filters.items():
                if child.get(k) != v:
                    return True
            return False

        documents = []

        for idx, child in enumerate(root):

            if idx == first_n:
                break

            if ignore_question(filters):
                continue

            header = child.find('header').text
            statement = child.find('statement').text

            for pattern, replace in PATTERNS_REPLACES:
                header = apply_regex(pattern, replace, header)
                statement = apply_regex(pattern, replace, statement)

            options = []

            answers = child.find('answers')
            for option in answers.iter('option'):
                id = option.get('id')
                text = option.text
                for pattern, replace in PATTERNS_REPLACES:
                    id = apply_regex(pattern, replace, id)
                    if text is not None:
                        text = apply_regex(pattern, replace, text)
                options.append(text)

                if option.get('correct') == 'Yes':
                    correct = option.get('id')

            document = {
                'context': header,
                'question': statement,
                'options': options,
                'label': correct.lower(),
            }
            assert len(document['options']) == 5, print('The document does not have 5 options')
            documents.append(document)

        return documents

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        print(len(self.dataset['train']))
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def test_docs(self):
        print(len(self.dataset['test']))
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, choices):
            """
                Passagem: <passage>
                Pergunta: <question>
                Choices:
                A. <choice1>
                B. <choice2>
                C. <choice3>
                D. <choice4>
                Answer:
            """
            prompt = "Cabeçalho: " + doc["context"] + "\n"
            prompt += "Enunciado: " + doc["question"] + "\nAlternativas:\n"
            for choice, option in zip(choices, doc["options"]):
                prompt += f"{choice.upper()}. {option}\n"
            prompt += "Resposta:"
            return prompt
        choices = ['a', 'b', 'c', 'd', 'e']
        return {
            "query": format_example(doc, choices),
            "choices": doc["options"],
            "gold": choices.index(doc["label"])
        }

    def doc_to_text(self, doc):
        return doc["query"]
