# -*- coding:utf-8 -*-

import thulac
import pke
from pke.unsupervised import (
    TopicRank, TopicalPageRank, SingleRank, PositionRank, MultipartiteRank
)
from jieba.analyse import extract_tags
from unittest import TestCase, main


pos = ["N", "NS", "NP", "A", "V"]
thul = thulac.thulac()
stopwords = []
with open('/home/songwenbing/nltk_data/corpora/stopwords/chinese', 'r') as sf:
    while True:
        word = sf.readline()
        if not word:
            break
        stopwords.append(word.strip())
model_funcs = {
        'topic': (
            TopicRank,
            ({'pos': pos, 'stoplist': stopwords, 'compound': False},
             {}
             )
        ),
        'topical_page': (
            TopicalPageRank,
            ({},
             {}
             )
        ),
        'single': (
            SingleRank,
            ({},
             {}
             )
        ),
        'position': (
            PositionRank,
            ({},
             {}
             )
        ),
        'multipartite': (
            MultipartiteRank,
            ({},
             {}
             )
        )
    }


def get_file(s):
    s = thul.cut(s, text=True)
    s = s.replace('\n', ' ._w ')
    file_path = '/home/songwenbing/Data/pke/test.txt'
    with open(file_path, 'w') as f:
        f.write(s)
    return file_path


def pke_keyphrases(s, k, model):
    model_func = model_funcs[model]
    file_path = get_file(s)
    extractor = model_func(input_file=file_path, language='chinese')
    extractor.read_document(format='preprocessed', sep='_', stemmer=None)
    extractor.candidate_selection(pos=pos, stoplist=stopwords, compound=False)
    print("after stops: ", extractor.candidates)
    extractor.candidate_weighting(pos=pos)
    return extractor.get_n_best(k)


def jieba_keyphrases(s, k):
    return extract_tags(s, topK=k, withWeight=True)


class TestPke(TestCase):

    def setUp(self):
        self.long_str = (
        '''改革开放以来，中国取得一切成绩和进步的根本原因，就是开辟了中国特色社会主义道路，形成了中国特色社会主义理论体系，
        确立了中国特色社会主义制度，发展了中国特色社会主义文化。习近平总书记指出：“当今世界，要说哪个政党、哪个国家、哪个民族能
        够自信的话，那中国共产党、中华人民共和国、中华民族是最有理由自信的。”所以说，中国特色社会主义道路自信、理论自信、制度自
        信、文化自信是最有理由的自信，是理由最充分的自信。坚定中国特色社会主义“四个自信”需要我们深入理解、全面把握其重大意义、
        精神实质和丰富内涵。\n中国特色社会主义道路是实现社会主义现代化、最终实现共同富裕的必由之路，是实现中华民族伟大复兴、
        屹立于世界民族之林的必由之路。中国特色社会主义道路坚持“一个中心、两个基本点”，统筹推进“五位一体”总体布局，协调推进“
        四个全面”战略布局，不断解放和发展社会生产力，分阶段实现全体人民共同富裕，最终促进人的全面发展。无论是改革开放以来的各
        种风险挑战，还是十八大以来，具有许多新的历史特点的伟大斗争，其结果都无可争辩地证明，中国特色社会主义这条道路前途光明、
        方向正确、成就卓著。这条道路能否走得好、走得远，取决于我们的清醒头脑和战略定力。我们有960万平方公里的广阔舞台，有5000
        多年的深厚历史底蕴，有近9000万中共党员和450万基层党组织的坚强执政力量。这就是中国特色社会主义能够成为人间正道、通往复
        兴梦想的最有力理由。\n中国特色社会主义理论体系是实现社会主义现代化和中华民族复兴的正确指导理论，是科学社会主义基本原理
        与中国实际相结合的科学理论。问题是时代的声音，理论是实践的升华。它凝聚了几代中国共产党人不懈探索实践的智慧心血，反映了
        改革开放以来马克思主义中国化的理论创新成果，创造了党最可宝贵的政治财富和精神财富，奠定了全国各族人民团结奋斗的共同思想
        基础。理论的说服力来自于理论的彻底与科学。中国特色社会主义理论体系包括邓小平理论、“三个代表”重要思想、科学发展观、习近
        平新时代中国特色社会主义思想，创造性探索和回答了什么是马克思主义、怎样对待马克思主义，什么是社会主义、怎样建设社会主义，
        建设什么样的党、怎样建设党，实现什么样的发展、怎样发展，新时代坚持和发展什么样的中国特色社会主义、怎样坚持和发展中国特
        色社会主义等重大问题。\n中国特色社会主义制度是当代中国发展进步和民族复兴的根本制度保障。这一制度体现在“五位一体”各个方
        面，如人民代表大会制度的根本政治制度，中国共产党领导的多党合作和政治协商制度、民族区域自治制度和基层群众自治制度等基本
        政治制度，中国特色社会主义法律体系，公有制为主体、多种所有制经济共同发展的基本经济制度，以及建立在这些制度基础上的各项
        具体制度等。这一制度具有鲜明的中国特色，拥有强大的自我完善能力，集中体现了中国特色社会主义的制度优势。“橘生淮南则为橘，
        生于淮北则为枳”。习近平总书记指出，政治制度“不可能脱离特定社会政治条件来抽象评判，不可能千篇一律、归于一尊”。应该看到，
        中国特色社会主义制度不是理想完美、成熟定型的。要从实际出发，构建系统完备、科学规范、运行有效的制度体系，使各方面制度更
        加成熟、更加定型。中国特色社会主义文化是激励全党全国各族人民奋勇前进的强大精神力量。发展中国特色社会主义文化就是发展面
        向现代化、面向世界、面向未来的，民族的科学的大众的社会主义文化。中国特色社会主义文化源自于中华传统优秀文化、近现代中国
        革命文化、社会主义先进文化，积淀着中华民族最深沉的精神追求，代表着中华民族最独特的精神标识。习近平总书记强调，“文化自信
        是更基础、更广泛、更深厚的自信”，“文化自信是一个国家、一个民族发展中更基本、更深沉、更持久的力量”。坚定中国特色社会主
        义道路自信、理论自信、制度自信，说到底是要坚持文化自信。一枝独放不是春，万紫千红春满园。坚定文化自信，推进文化繁荣兴盛，
        要心态更加自信、胸怀更加宽广、胆略更加过人、方式更加多样，广泛参与世界文化交流对话，大胆借鉴一切文明成果，更好打造中国
        精神、中国价值、中国力量。\n在中国特色社会主义伟大实践中，中国特色社会主义道路是实现途径，中国特色社会主义理论体系是行
        动指南，中国特色社会主义制度是根本保障，中国特色社会主义文化是精神力量。这是中国特色社会主义区别于其他主义的根本标志，
        是中国特色社会主义得以成立的最充分理由。中国特色社会主义“四个自信”来源于“敢教日月换新天”的伟大实践、伟大人民、伟大真理，
        必将继续激励全国各族人民开辟历史新天地、创造人间新奇迹。''')
        self.short_str1 = '中国取得一切成绩和进步的根本原因'
        self.short_str2 = '我们明天去北京出差'
        self.short_str3 = '政治制度不能脱离特定社会政治条件来抽象批判，不能千篇一律、归于一尊'
        self.short_str4 = '武汉市长昨日考察武汉市长江大桥'
        self.tags_long = ['中国特色社会主义', '四个自信', '习近平']

    def test_long(self):
        # print(pke_keyphrases(self.long_str, 5, 'single'))
        # print(pke_keyphrases(self.long_str, 5, 'topic'))
        # print(pke_keyphrases(self.long_str, 5, 'topical_page'))
        # print(pke_keyphrases(self.long_str, 5, 'position'))
        # print(pke_keyphrases(self.long_str, 5, 'multipartite'))
        # print(jieba_keyphrases(self.long_str, 5))
        pass

    def test_short(self):
        # print("\nshort1: ", self.short_str1)
        print(pke_keyphrases(self.short_str1, 5, 'single'))
        # print(pke_keyphrases(self.short_str1, 5, 'topic'))
        # print(pke_keyphrases(self.short_str1, 5, 'topical_page'))
        # print(pke_keyphrases(self.short_str1, 5, 'position'))
        print(pke_keyphrases(self.short_str1, 5, 'multipartite'))
        print(jieba_keyphrases(self.short_str1, 5))
        #
        # print("\nshort2: ", self.short_str2)
        # print(pke_keyphrases(self.short_str2, 5, 'single'))
        # print(pke_keyphrases(self.short_str2, 5, 'topic'))
        # print(pke_keyphrases(self.short_str2, 5, 'topical_page'))
        # print(pke_keyphrases(self.short_str2, 5, 'position'))
        # print(pke_keyphrases(self.short_str2, 5, 'multipartite'))
        # print(jieba_keyphrases(self.short_str2, 5))
        #
        # print("\nshort3: ", self.short_str3)
        # print(pke_keyphrases(self.short_str3, 5, 'single'))
        # print(pke_keyphrases(self.short_str3, 5, 'topic'))
        # print(pke_keyphrases(self.short_str3, 5, 'topical_page'))
        # print(pke_keyphrases(self.short_str3, 5, 'position'))
        # print(pke_keyphrases(self.short_str3, 5, 'multipartite'))
        # print(jieba_keyphrases(self.short_str3, 5))
        #
        # print("\nshort4: ", self.short_str4)
        # print(pke_keyphrases(self.short_str4, 5, 'single'))
        # print(pke_keyphrases(self.short_str4, 5, 'topic'))
        # print(pke_keyphrases(self.short_str4, 5, 'topical_page'))
        # print(pke_keyphrases(self.short_str4, 5, 'position'))
        # print(pke_keyphrases(self.short_str4, 5, 'multipartite'))
        # print(jieba_keyphrases(self.short_str4, 5))
        pass


if __name__ == '__main__':
    main()
