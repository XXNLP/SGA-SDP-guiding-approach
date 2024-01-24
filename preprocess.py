import os, json, pickle, sys

class KBP37():

    def __init__(self,base_path):
        self.file_base_path = os.path.join(base_path,"kbp37","original_dataset")
        self.dataset_name = "kbp37"
        self.train_data = self.load("train")
        self.dev_data = self.load("dev")
        self.test_data =self.load("test")
        #self.json_data = self.convert2pickle()

    def load(self, dataType):

        entities = []  # the entity name
        entity_ind = []  # the index of the elements
        sentence_label = []
        tokens = []

        sentences = open(os.path.join(self.file_base_path, dataType+".txt"), 'r')
        line = sentences.readline()
        while True:
            # 空行或者数字行跳出循环
            id = -1

            while (line != '' and not line[0].isdigit()):
                line = sentences.readline()
            if line == '':
                break
            id = line.split('\t')[0]
            line = self.check_error(line)
            line = line[line.find('"') + 1:len(line) - 3]  # 保留到句号前一位

            format_output = self.locate_entities(id,line)
            words, e1, e2 = format_output[0], format_output[1], format_output[2]
            line = words
            line.append('.')
            entities.append([e1,e2])
            tokens.append(line)

            line = sentences.readline()
            if (line != '' and not line[0].isdigit()):
                category = line.rstrip()
                # use dictionary to get sentence label
                if category != "no_relation":
                    head = int(category[-5]) if category[-5].isdigit() else 1
                    tail = int(category[-2]) if category[-2].isdigit() else 2
                sentence_label.append({'h':head,'t':tail,'r':category})
                line = sentences.readline()  # 跳过comment这一行
        sentences.close()
        return {'tokens': tokens, 'entities': entities, "label": sentence_label}

        #for i in range(0, len(lines),4):
        #    item = {
        #        "token":self.load_tokens(),
        #        "h":self.load_head(),
        #        "t":self.load_tail(),
        #        "relation":lines[i+1]
        #    }


    def load_tokens(self, data):
        return data["tokens"]

    def load_head(self, data):
        return [e_name[e_id["h"]-1] for e_id,e_name in zip(data['label'],data["entities"])]

    def load_tail(self, data):
        return [e_name[e_id["t"]-1] for e_id,e_name in zip(data['label'],data["entities"])]

    def load_relation(self, data):
        return [h['r'] for h in data['label']]


    def convert2NewTxt(self, data, dataType):
        tokens = self.load_tokens(data)
        head = self.load_head(data)
        tail = self.load_tail(data)
        relation = self.load_relation(data)
        j_file = []
        for token, h, t, r in zip(tokens,head,tail,relation):
            item = {
                "token":token,
                "h":h,
                "t":t,
                "relation":r
            }
            j_file.append(item)
        file_path = os.path.join('./benchmark', self.dataset_name, dataType + ".txt")
        with open(file_path, 'w', encoding='UTF-8') as f:
            for ele in j_file:
                f.write(json.dumps(ele) + "\n")
    def convert2relId(self):
        relation_set = set([r["r"] for r in self.train_data["label"]])
        relation_set = relation_set.union(set([r["r"] for r in self.dev_data["label"]]))
        relation_set = relation_set.union(set([r["r"] for r in self.test_data["label"]]))
        rel2id = {name:id for id, name in enumerate(relation_set)}
        file_path = os.path.join('./benchmark', self.dataset_name, "rel2id.json")
        if os.path.exists(file_path):
            pass
        else:
            with open(file_path,'w', encoding="UTF-8") as f:
                f.write(json.dumps(rel2id))

    def dump(self):
        self.convert2NewTxt(self.train_data,"train")
        self.convert2NewTxt(self.dev_data, "dev")
        self.convert2NewTxt(self.test_data, "test")
        self.convert2relId()

    def check_error(self, line):
        #file = ["train.txt","dev.txt","test.txt"]
        #dataset_name="kbp37"
        #for name in file:
            #with open(os.path.join('./benchmark', dataset_name, name)) as sentences:
                #line = sentences.readline()
                #while True:
                    # 空行或者数字行跳出循环
                    #while (line != '' and not line[0].isdigit()):
                    #    line = sentences.readline()
        #if line == '':
        #    break
        #id = line.split('\t')[0]
        #print(id)
        if line[-3] == " ":  # the last word is null.
            temp = list(line)
            del temp[-3]
            line = "".join(temp)
        if line[-4] == " ":
            temp = list(line)
            del temp[-4]
            line = "".join(temp)
        if line[-3] ==">":
            temp = list(line)
            temp.insert(-2,".")
            line = "".join(temp)
        if line[-3].isalpha():
            temp = list(line)
            temp.insert(-2,".")
            line = "".join(temp)
        if line[-3] not in [".","!","?"]: #the last
            temp = list(line)
            temp.insert(-2,".")
        return line
                    #line = line[line.find('"') + 1:len(line) - 3]  #


    def format_line(self,line):
        # This will take in a raw input sentence and return [[element strings], [indicies of elements], [sentence with elements removed]]
        print(line)
        words = line.split(' ')

        e1detagged = []
        e2detagged = []
        rebuilt_line = ''
        count = 1  # 对单词的位置进行标记，从1开始
        for word in words:
            # if tagged at all
            # print(word)
            if word.find('<e') != -1 or word.find('</e') != -1:
                # e1 or e2
                if word[2] == '1' or word[word.find('>') - 1] == '1':
                    # remove tags from word for
                    e1detagged = self.get_word(words, word)
                    e1detagged.append(count)
                    # replace and tac back on . at end if needed
                    word = self.replace_word(word)
                else:
                    e2detagged = self.get_word(words, word)
                    e2detagged.append(count)
                    word = self.replace_word(word)
            rebuilt_line += ' ' + word
            # 处理关系的数据
            print(word)
            first_char = word[0]
            last_char = word[len(word) - 1]
            if first_char == '(' or last_char == ')' or last_char == ',':
                count += 1
            count += 1
        rebuilt_line = rebuilt_line[1:len(rebuilt_line)]
        rebuilt_line += '\n'
        return [[e1detagged[0], e2detagged[0]], [e1detagged[1], e2detagged[1]], [e1detagged[2], e2detagged[2]],
                rebuilt_line]

    def format_line_2(self,line):
        # This will take in a raw input sentence and return [[element strings], [indicies of elements], [sentence with elements removed]]
        print(line)
        words = line.split(' ')

        e1detagged = []
        e2detagged = []
        rebuilt_line = ''
        count = 1  # 对单词的位置进行标记，从1开始
        for word in words:
            # if tagged at all
            # print(word)
            if word.find('<e') != -1 or word.find('</e') != -1:
                # e1 or e2
                if word[2] == '1' or word[word.find('>') - 1] == '1':
                    # remove tags from word for
                    e1detagged = self.get_word(words, word)
                    e1detagged.append(count)
                    # replace and tac back on . at end if needed
                    word = self.replace_word(word)
                else:
                    e2detagged = self.get_word(words, word)
                    e2detagged.append(count)
                    word = self.replace_word(word)
            rebuilt_line += ' ' + word
            # 处理关系的数据
            #print(word)
            first_char = word[0]
            last_char = word[len(word) - 1]
            if first_char == '(' or last_char == ')' or last_char == ',':
                count += 1
            count += 1
        rebuilt_line = rebuilt_line[1:len(rebuilt_line)]
        rebuilt_line += '\n'
        return [[e1detagged[0], e2detagged[0]], [e1detagged[1], e2detagged[1]], [e1detagged[2], e2detagged[2]],
                rebuilt_line]
    def get_word(self, words, word):
        if self.end_two_words(word):
            return [self.replace_word(word, False), 1]
        else:
            return [self.replace_word(word, False), 0]

    def replace_word(self, word, should_end_sentence=True):
        word_list = word.split('</')
        end_sentence = ''
        if len(word_list) == 2 and len(word_list[len(word_list) - 1]) != 3:
            end = word_list[len(word_list) - 1]
            end_sentence += end[end.find('>') + 1:len(end)]
        word_list = word_list[0].split('>')
        new_word = word_list[len(word_list) - 1]
        if should_end_sentence:
            new_word += end_sentence
        return new_word

    # if this has a two or more words ex. <e2>fast cars</e2>
    def end_two_words(self, word):
        return word.find('<e') == -1

    def pre_process_KBP37(self, file):
        """
        preprocess KBP37
        :param file:
        :param cat_map:
        :return:
        """
        # parameters to return
        entity_strings = []  # the entity name
        entity_ind = []  # the index of the elements
        sentence_label = []
        raw_sentences = []

        sentences = open(file, 'r')
        line = sentences.readline()
        while True:
            # 空行或者数字行跳出循环
            if line[0].isdigit(): print(line[0])
            while (line != '' and not line[0].isdigit()):
                line = sentences.readline()

            if line == '':
                break
            line = line[line.find('"') + 1:len(line) - 3]  # 保留到句号前一位

            format_output = self.format_line(line)
            line = format_output[3]
            entity_strings.append(format_output[0])
            entity_ind.append(format_output[2])
            raw_sentences.append(line[:-1] + '.')

            line = sentences.readline()
            if (line != '' and not line[0].isdigit()):
                category = line
                # use dictionary to get sentence label
                sentence_label.append(category)
                line = sentences.readline()  # 跳过comment这一行
        sentences.close()

        return entity_strings, entity_ind, sentence_label, raw_sentences

    def locate_entities(self, sentence_id, line):
        """

        """
        words = line.split(' ')
        e1tag = {}  # {"pos":[start,end], "name":words}
        e2tag = {}  # {"pos":[start,end], "name":words}
        rebuilt_line = ''
        count = 0  # 对单词的位置进行标记，从1开始
        for id, word in enumerate(words):
            # if tagged at all
            # print(word)
            e_start_exist = True if word.find('<e') != -1 else False
            e_end_exist = True if word.find('</e') != -1 else False
            if e_start_exist or e_end_exist:
                e1_start_exist = True if word.find('<e1>') != -1 else False
                e1_end_exist = True if word.find('</e1>') != -1 else False
                e2_start_exist = True if word.find('<e2>') != -1 else False
                e2_end_exist = True if word.find('</e2>') != -1 else False
                if e1_start_exist or e1_end_exist:
                    if e1_start_exist is True and e1_end_exist is False:
                        e1tag['start'] = id
                        words[id] = word.replace('<e1>',"")
                    elif e1_start_exist is True and e1_end_exist is True:
                        e1tag['start'] = id
                        e1tag['end'] = id + 1
                        words[id] = word.replace('<e1>', "").replace('</e1>', "")
                    elif e1_start_exist is False and e1_end_exist is True:
                        e1tag['end'] = id + 1
                        words[id] = word.replace('</e1>', "")
                else:
                    if e2_start_exist is True and e2_end_exist is False:
                        e2tag['start'] = id
                        words[id] = word.replace('<e2>', "")
                    elif e2_start_exist is True and e2_end_exist is True:
                        e2tag['start'] = id
                        e2tag['end'] = id + 1
                        words[id] = word.replace('<e2>', "").replace('</e2>', "")
                    elif e2_start_exist is False and e2_end_exist is True:
                        e2tag['end'] = id + 1
                        words[id] = word.replace('</e2>', "")
        e1tag["pos"] = [e1tag["start"],e1tag["end"]]
        e2tag["pos"] = [e2tag["start"], e2tag["end"]]
        e1tag["name"] = " ".join(words[e1tag['start']:e1tag['end']])
        e2tag["name"] = " ".join(words[e2tag['start']:e2tag['end']])
        del e1tag["start"]
        del e1tag["end"]
        del e2tag["start"]
        del e2tag["end"]
        return words, e1tag, e2tag

    def remove_tag(self, word):
        pass
        #string.replace(substring, "")

class WEBNLG_v2_0():# webnlg2.0

    def __init__(self, base_path):
        self.file_base_path = os.path.join(base_path, "webnlg", "original")
        self.dataset_name = "webnlg"
        self.train_data = self.load("train")
        self.dev_data = self.load("dev")
        self.test_data = self.load("test")

    def load(self, dataType):

        entities = []  # the entity name
        entity_ind = []  # the index of the elements
        sentence_label = []
        tokens = []

        sentences = json.load(open(os.path.join(self.file_base_path, dataType+".json"), 'r'))
        entries = sentences['entries']
        for entry in entries:
            # 空行或者数字行跳出循环
            id = int(list(entry.keys())[0])


            #line = self.check_error(line)
            #line = line[line.find('"') + 1:len(line) - 3]  # 保留到句号前一位
            sent = entry['id']['lexicalisations']['lex']
            entity = entry['id']['modifietripleset'][0]

            format_output = self.locate_entities(id,line)
            words, e1, e2 = format_output[0], format_output[1], format_output[2]
            line = words
            line.append('.')
            entities.append([e1,e2])
            tokens.append(line)

            line = sentences.readline()
            if (line != '' and not line[0].isdigit()):
                category = line.rstrip()
                # use dictionary to get sentence label
                if category != "no_relation":
                    head = int(category[-5]) if category[-5].isdigit() else 1
                    tail = int(category[-2]) if category[-2].isdigit() else 2
                sentence_label.append({'h':head,'t':tail,'r':category})
                line = sentences.readline()  # 跳过comment这一行
        sentences.close()
        return {'tokens': tokens, 'entities': entities, "label": sentence_label}

        #for i in range(0, len(lines),4):
        #    item = {
        #        "token":self.load_tokens(),
        #        "h":self.load_head(),
        #        "t":self.load_tail(),
        #        "relation":lines[i+1]
        #    }


    def load_tokens(self, data):
        return data["tokens"]

    def load_head(self, data):
        return [e_name[e_id["h"]-1] for e_id,e_name in zip(data['label'],data["entities"])]

    def load_tail(self, data):
        return [e_name[e_id["t"]-1] for e_id,e_name in zip(data['label'],data["entities"])]

    def load_relation(self, data):
        return [h['r'] for h in data['label']]


    def convert2NewTxt(self, data, dataType):
        tokens = self.load_tokens(data)
        head = self.load_head(data)
        tail = self.load_tail(data)
        relation = self.load_relation(data)
        j_file = []
        for token, h, t, r in zip(tokens,head,tail,relation):
            item = {
                "token":token,
                "h":h,
                "t":t,
                "relation":r
            }
            j_file.append(item)
        file_path = os.path.join('./benchmark', self.dataset_name, dataType + ".txt")
        with open(file_path, 'w', encoding='UTF-8') as f:
            for ele in j_file:
                f.write(json.dumps(ele) + "\n")
    def convert2relId(self):
        relation_set = set([r["r"] for r in self.train_data["label"]])
        relation_set = relation_set.union(set([r["r"] for r in self.dev_data["label"]]))
        relation_set = relation_set.union(set([r["r"] for r in self.test_data["label"]]))
        rel2id = {name:id for id, name in enumerate(relation_set)}
        file_path = os.path.join('./benchmark', self.dataset_name, "rel2id.json")
        if os.path.exists(file_path):
            pass
        else:
            with open(file_path,'w', encoding="UTF-8") as f:
                f.write(json.dumps(rel2id))

    def dump(self):
        self.convert2NewTxt(self.train_data, "train")
        self.convert2NewTxt(self.dev_data, "dev")
        self.convert2NewTxt(self.test_data, "test")
        self.convert2relId()

    def check_error(self, line):
        #file = ["train.txt","dev.txt","test.txt"]
        #dataset_name="kbp37"
        #for name in file:
            #with open(os.path.join('./benchmark', dataset_name, name)) as sentences:
                #line = sentences.readline()
                #while True:
                    # 空行或者数字行跳出循环
                    #while (line != '' and not line[0].isdigit()):
                    #    line = sentences.readline()
        #if line == '':
        #    break
        #id = line.split('\t')[0]
        #print(id)
        if line[-3] == " ":  # the last word is null.
            temp = list(line)
            del temp[-3]
            line = "".join(temp)
        if line[-4] == " ":
            temp = list(line)
            del temp[-4]
            line = "".join(temp)
        if line[-3] ==">":
            temp = list(line)
            temp.insert(-2,".")
            line = "".join(temp)
        if line[-3].isalpha():
            temp = list(line)
            temp.insert(-2,".")
            line = "".join(temp)
        if line[-3] not in [".","!","?"]: #the last
            temp = list(line)
            temp.insert(-2,".")
        return line
                    #line = line[line.find('"') + 1:len(line) - 3]  #

    def locate_entities(self, sentence_id, line):
        words = line.split(' ')
        e1tag = {}  # {"pos":[start,end], "name":words}
        e2tag = {}  # {"pos":[start,end], "name":words}

        return words, e1tag, e2tag

class CONLL04_ADE_SCIERC():

    def __init__(self, base_path, dataset_name):        
        self.dataset_name = dataset_name
        self.file_base_path = os.path.join(base_path, dataset_name)

        self.train_data = self.load("train_triples")
        self.dev_data = self.load("dev_triples")
        self.test_data = self.load("test_triples")

    def load(self, dataType):
        sentences = json.load(open(os.path.join(self.file_base_path,"original" ,dataType + ".json"), 'r'))
        tokens=[]
        entities = []
        sentence_label=[]
        j_file = []
        for item in sentences:
            #tokens.append(item['tokens'])
            for rel_tri in item['relations']:
                h_id = rel_tri["head"]
                h_pos = [item['entities'][h_id]['start'],item['entities'][h_id]['end']]
                #h = {"name": item['tokens'][h[0]:h[1]],"pos": h_pos}
                t_id = rel_tri["tail"]
                t_pos = [item['entities'][t_id]['start'], item['entities'][t_id]['end']]
                #t = {"name": item['tokens'][t[0]:t[1]], "pos": t_pos}
                #entities.append([item['relation']['h']])
                res = {
                    "token":item["tokens"],
                    "h":{"name": item['tokens'][h_pos[0]:h_pos[1]],"pos": h_pos},
                    "t":{"name": item['tokens'][t_pos[0]:t_pos[1]], "pos": t_pos},
                    "relation": rel_tri['type']
                }
                j_file.append(res)
        return j_file

    def convert2relId(self):
        relation_set = set([r["relation"] for r in self.train_data])
        relation_set = relation_set.union(set([r["relation"] for r in self.dev_data]))
        relation_set = relation_set.union(set([r["relation"] for r in self.test_data]))
        rel2id = {name:id for id, name in enumerate(relation_set)}
        file_path = os.path.join('./benchmark', self.dataset_name, "rel2id.json")
        if os.path.exists(file_path):
            pass
        else:
            with open(file_path,'w', encoding="UTF-8") as f:
                f.write(json.dumps(rel2id))

    def dump(self):
        for dataType,data in zip(["train","test","dev"],[self.train_data,self.test_data,self.dev_data]):
            file_path = os.path.join('./benchmark', self.dataset_name, dataType + ".txt")
            with open(file_path, 'w', encoding='UTF-8') as f:
                for ele in data:
                    f.write(json.dumps(ele) + "\n")
        self.convert2relId()


class webnlg_5019():  #use nyt10 test webnlg cod

    def __init__(self, base_path):
        self.file_base_path = os.path.join(base_path, "webnlg_5019","original")
        self.dataset_name = "webnlg_5019"
        self.train_data = self.load("train_data")
        self.dev_data = self.load("valid_data")
        self.test_data = self.load("test_data")

    def load(self,dataType):
        sentences = json.load(open(os.path.join(self.file_base_path, dataType + ".json"), 'r'))
        tokens=[]
        entities = []
        sentence_label=[]
        j_file = []

        for idx, item in enumerate(sentences):
            token = item['text'].split(" ")
            for triplet in item["relation_list"]:
                #h_pos = triplet['subj_tok_span']

                h_pos, t_pos, h_name, t_name = self.locate_pos(item['text'],triplet)

                #assert h_name == token[h_pos[0]:h_pos[1]]

                # assert h_name == token[h_pos[0]:h_pos[1]]
                #assert t_name == token[t_pos[0]:t_pos[1]]
                relation = triplet["predicate"]

                res = {
                    "token": token,
                    "h": {"name": h_name, "pos": h_pos},
                    "t": {"name": t_name, "pos": t_pos},
                    "relation": relation
                }
                j_file.append(res)
                #t_pos = [token.index(t_str[0]), token.index(t_str[-1])+1]
                #item['token'] = token
                #item['h']['pos'] = h_pos
                #item['t']['pos'] = t_pos

        return j_file

    def locate_pos(self,text_str, triplet):
        token = text_str.split(" ")
        index = [len(t) for t in token]
        index = [(i+sum(index[0:i]),i+sum(index[0:i+1])) for i in range(len(index))]
        index = {t:id for id, t in enumerate(index)}
        rev_index = {id:t for id, t in enumerate(index)}

        h_name = triplet['subject'].split(" ")
        token_start, token_end = token.index(h_name[0]), token.index(h_name[-1])
        char_start = triplet['subj_char_span'][0]
        char_end = triplet['subj_char_span'][1]
        if char_start not in rev_index[token_start]:
            for (start, end) in index.keys():
                if char_start in [start, end]:
                    token_start = index[(start, end)]
        if char_end not in rev_index[token_end]:
            for (start, end) in index.keys():
                if char_end in [start, end]:
                    token_end = index[(start, end)]
        h_pos = [token_start,token_end+1]
        h_name = " ".join(token[h_pos[0]:h_pos[1]])
        assert h_name == triplet['subject']

        t_name = triplet['object'].split(" ")
        token_start, token_end = token.index(t_name[0]), token.index(t_name[-1])
        char_start = triplet['obj_char_span'][0]
        char_end = triplet['obj_char_span'][1]
        if char_start not in rev_index[token_start]:
            for (start, end) in index.keys():
                if char_start in [start, end]:
                    token_start = index[(start, end)]
        if char_end not in rev_index[token_end]:
            for (start, end) in index.keys():
                if char_end in [start, end]:
                    token_end = index[(start, end)]
        t_pos = [token_start,token_end+1]
        t_name = " ".join(token[t_pos[0]:t_pos[1]])
        assert t_name == triplet['object']

        """
        h_name = triplet["subject"]
        char_start = triplet['subj_char_span'][0]
        char_end = triplet['subj_char_span'][1]
        token_start, token_end = 0, 0
        for (start,end) in index.keys():
            if char_start in [start, end]:
                token_start = index[(start,end)]
            if char_end in [start, end]:
                token_end = index[(start,end)]
        h_pos=[token_start, token_end+1]
        assert " ".join(token[h_pos[0]:h_pos[1]])==h_name

        t_name = triplet["object"]
        char_start = triplet['obj_char_span'][0]
        char_end = triplet['obj_char_span'][1]
        token_start, token_end = 0, 0
        for (start, end) in index.keys():
            if char_start in [start, end]:
                token_start = index[(start, end)]
            if char_end in [start, end]:
                token_end = index[(start, end)]
        t_pos = [token_start, token_end + 1]
        assert " ".join(token[t_pos[0]:t_pos[1]]) == t_name
        """
        return h_pos, t_pos, h_name, t_name
    def convert2relId(self):
        relation_set = set([r["relation"] for r in self.train_data])
        relation_set = relation_set.union(set([r["relation"] for r in self.dev_data]))
        relation_set = relation_set.union(set([r["relation"] for r in self.test_data]))
        rel2id = {name:id for id, name in enumerate(relation_set)}
        file_path = os.path.join('./benchmark', self.dataset_name, "rel2id.json")
        if os.path.exists(file_path):
            pass
        else:
            with open(file_path,'w', encoding="UTF-8") as f:
                f.write(json.dumps(rel2id))


    def dump(self):
        for dataType,data in zip(["train","test","dev"],[self.train_data,self.test_data,self.dev_data]):
            file_path = os.path.join('./benchmark', self.dataset_name, dataType + ".txt")
            with open(file_path, 'w', encoding='UTF-8') as f:
                for ele in data:
                    f.write(json.dumps(ele) + "\n")
        self.convert2relId()

class nyt10():

    def __init__(self, base_path):
        self.file_base_path = os.path.join(base_path, "nyt","original")
        self.dataset_name = "nyt"
        self.train_data = self.load("train_data")
        self.dev_data = self.load("valid_data")
        self.test_data = self.load("test_data")

    def load(self,dataType):
        sentences = json.load(open(os.path.join(self.file_base_path, dataType + ".json"), 'r'))
        tokens=[]
        entities = []
        sentence_label=[]
        j_file = []

        for idx, item in enumerate(sentences):
            token = item['text'].split(" ")
            for triplet in item["relation_list"]:
                #h_pos = triplet['subj_tok_span']

                h_pos, t_pos, h_name, t_name = self.locate_pos(item['text'],triplet)

                #assert h_name == token[h_pos[0]:h_pos[1]]

                # assert h_name == token[h_pos[0]:h_pos[1]]
                #assert t_name == token[t_pos[0]:t_pos[1]]
                relation = triplet["predicate"]

                res = {
                    "token": token,
                    "h": {"name": h_name, "pos": h_pos},
                    "t": {"name": t_name, "pos": t_pos},
                    "relation": relation
                }
                j_file.append(res)
                #t_pos = [token.index(t_str[0]), token.index(t_str[-1])+1]
                #item['token'] = token
                #item['h']['pos'] = h_pos
                #item['t']['pos'] = t_pos

        return j_file

    def locate_pos(self,text_str, triplet):
        token = text_str.split(" ")
        index = [len(t) for t in token]
        index = [(i+sum(index[0:i]),i+sum(index[0:i+1])) for i in range(len(index))]
        index = {t:id for id, t in enumerate(index)}
        rev_index = {id:t for id, t in enumerate(index)}

        h_name = triplet['subject'].split(" ")
        token_start, token_end = token.index(h_name[0]), token.index(h_name[-1])
        char_start = triplet['subj_char_span'][0]
        char_end = triplet['subj_char_span'][1]
        if char_start not in rev_index[token_start]:
            for (start, end) in index.keys():
                if char_start in [start, end]:
                    token_start = index[(start, end)]
        if char_end not in rev_index[token_end]:
            for (start, end) in index.keys():
                if char_end in [start, end]:
                    token_end = index[(start, end)]
        h_pos = [token_start,token_end+1]
        h_name = " ".join(token[h_pos[0]:h_pos[1]])
        assert h_name == triplet['subject']

        t_name = triplet['object'].split(" ")
        token_start, token_end = token.index(t_name[0]), token.index(t_name[-1])
        char_start = triplet['obj_char_span'][0]
        char_end = triplet['obj_char_span'][1]
        if char_start not in rev_index[token_start]:
            for (start, end) in index.keys():
                if char_start in [start, end]:
                    token_start = index[(start, end)]
        if char_end not in rev_index[token_end]:
            for (start, end) in index.keys():
                if char_end in [start, end]:
                    token_end = index[(start, end)]
        t_pos = [token_start,token_end+1]
        t_name = " ".join(token[t_pos[0]:t_pos[1]])
        assert t_name == triplet['object']

        """
        h_name = triplet["subject"]
        char_start = triplet['subj_char_span'][0]
        char_end = triplet['subj_char_span'][1]
        token_start, token_end = 0, 0
        for (start,end) in index.keys():
            if char_start in [start, end]:
                token_start = index[(start,end)]
            if char_end in [start, end]:
                token_end = index[(start,end)]
        h_pos=[token_start, token_end+1]
        assert " ".join(token[h_pos[0]:h_pos[1]])==h_name

        t_name = triplet["object"]
        char_start = triplet['obj_char_span'][0]
        char_end = triplet['obj_char_span'][1]
        token_start, token_end = 0, 0
        for (start, end) in index.keys():
            if char_start in [start, end]:
                token_start = index[(start, end)]
            if char_end in [start, end]:
                token_end = index[(start, end)]
        t_pos = [token_start, token_end + 1]
        assert " ".join(token[t_pos[0]:t_pos[1]]) == t_name
        """
        return h_pos, t_pos, h_name, t_name
    def convert2relId(self):
        relation_set = set([r["relation"] for r in self.train_data])
        relation_set = relation_set.union(set([r["relation"] for r in self.dev_data]))
        relation_set = relation_set.union(set([r["relation"] for r in self.test_data]))
        rel2id = {name:id for id, name in enumerate(relation_set)}
        file_path = os.path.join('./benchmark', self.dataset_name, "rel2id.json")
        if os.path.exists(file_path):
            pass
        else:
            with open(file_path,'w', encoding="UTF-8") as f:
                f.write(json.dumps(rel2id))


    def dump(self):
        for dataType,data in zip(["train","test","dev"],[self.train_data,self.test_data,self.dev_data]):
            file_path = os.path.join('./benchmark', self.dataset_name, dataType + ".txt")
            with open(file_path, 'w', encoding='UTF-8') as f:
                for ele in data:
                    f.write(json.dumps(ele) + "\n")
        self.convert2relId()

class nyt24():

    def __init__(self, base_path):
        self.file_base_path = os.path.join(base_path, "nyt24","original")
        self.dataset_name = "nyt24"
        self.train_data = self.load("train_data")
        self.dev_data = self.load("valid_data")
        self.test_data = self.load("test_data")

    def load(self,dataType):
        sentences = json.load(open(os.path.join(self.file_base_path, dataType + ".json"), 'r'))
        tokens=[]
        entities = []
        sentence_label=[]
        j_file = []

        for idx, item in enumerate(sentences):
            token = item['text'].split(" ")
            for triplet in item["relation_list"]:
                #h_pos = triplet['subj_tok_span']

                h_pos, t_pos, h_name, t_name = self.locate_pos(item['text'],triplet)

                #assert h_name == token[h_pos[0]:h_pos[1]]

                # assert h_name == token[h_pos[0]:h_pos[1]]
                #assert t_name == token[t_pos[0]:t_pos[1]]
                relation = triplet["predicate"]

                res = {
                    "token": token,
                    "h": {"name": h_name, "pos": h_pos},
                    "t": {"name": t_name, "pos": t_pos},
                    "relation": relation
                }
                j_file.append(res)
                #t_pos = [token.index(t_str[0]), token.index(t_str[-1])+1]
                #item['token'] = token
                #item['h']['pos'] = h_pos
                #item['t']['pos'] = t_pos

        return j_file

    def locate_pos(self,text_str, triplet):
        token = text_str.split(" ")
        index = [len(t) for t in token]
        index = [(i+sum(index[0:i]),i+sum(index[0:i+1])) for i in range(len(index))]
        index = {t:id for id, t in enumerate(index)}
        rev_index = {id:t for id, t in enumerate(index)}

        h_name = triplet['subject'].split(" ")
        token_start, token_end = token.index(h_name[0]), token.index(h_name[-1])
        char_start = triplet['subj_char_span'][0]
        char_end = triplet['subj_char_span'][1]
        if char_start not in rev_index[token_start]:
            for (start, end) in index.keys():
                if char_start in [start, end]:
                    token_start = index[(start, end)]
        if char_end not in rev_index[token_end]:
            for (start, end) in index.keys():
                if char_end in [start, end]:
                    token_end = index[(start, end)]
        h_pos = [token_start,token_end+1]
        h_name = " ".join(token[h_pos[0]:h_pos[1]])
        assert h_name == triplet['subject']

        t_name = triplet['object'].split(" ")
        token_start, token_end = token.index(t_name[0]), token.index(t_name[-1])
        char_start = triplet['obj_char_span'][0]
        char_end = triplet['obj_char_span'][1]
        if char_start not in rev_index[token_start]:
            for (start, end) in index.keys():
                if char_start in [start, end]:
                    token_start = index[(start, end)]
        if char_end not in rev_index[token_end]:
            for (start, end) in index.keys():
                if char_end in [start, end]:
                    token_end = index[(start, end)]
        t_pos = [token_start,token_end+1]
        t_name = " ".join(token[t_pos[0]:t_pos[1]])
        assert t_name == triplet['object']

        """
        h_name = triplet["subject"]
        char_start = triplet['subj_char_span'][0]
        char_end = triplet['subj_char_span'][1]
        token_start, token_end = 0, 0
        for (start,end) in index.keys():
            if char_start in [start, end]:
                token_start = index[(start,end)]
            if char_end in [start, end]:
                token_end = index[(start,end)]
        h_pos=[token_start, token_end+1]
        assert " ".join(token[h_pos[0]:h_pos[1]])==h_name

        t_name = triplet["object"]
        char_start = triplet['obj_char_span'][0]
        char_end = triplet['obj_char_span'][1]
        token_start, token_end = 0, 0
        for (start, end) in index.keys():
            if char_start in [start, end]:
                token_start = index[(start, end)]
            if char_end in [start, end]:
                token_end = index[(start, end)]
        t_pos = [token_start, token_end + 1]
        assert " ".join(token[t_pos[0]:t_pos[1]]) == t_name
        """
        return h_pos, t_pos, h_name, t_name
    def convert2relId(self):
        relation_set = set([r["relation"] for r in self.train_data])
        relation_set = relation_set.union(set([r["relation"] for r in self.dev_data]))
        relation_set = relation_set.union(set([r["relation"] for r in self.test_data]))
        rel2id = {name:id for id, name in enumerate(relation_set)}
        file_path = os.path.join('./benchmark', self.dataset_name, "rel2id.json")
        if os.path.exists(file_path):
            pass
        else:
            with open(file_path,'w', encoding="UTF-8") as f:
                f.write(json.dumps(rel2id))


    def dump(self):
        for dataType,data in zip(["train","test","dev"],[self.train_data,self.test_data,self.dev_data]):
            file_path = os.path.join('./benchmark', self.dataset_name, dataType + ".txt")
            with open(file_path, 'w', encoding='UTF-8') as f:
                for ele in data:
                    f.write(json.dumps(ele) + "\n")
        self.convert2relId()
class Nyt24_star():  #use nyt24 

    q = '[{"text": "Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug ." ,"id": "train_0", "relation_list": [{"subject": "Annandale-on-Hudson", "object": "College", "subj_char_span": [68, 87], "obj_char_span": [58, 65], "predicate": "/location/location/contains", "subj_tok_span": [17, 24], "obj_tok_span": [15, 16]}],"entity_list": [{"text": "Annandale-on-Hudson", "type": "DEFAULT", "char_span": [68, 87], "tok_span": [17, 24]}]}]'

    def __init__(self, base_path):
        self.dataset_name = "nyt24_star"
        self.file_base_path = os.path.join(base_path, self.dataset_name,"original")
        self.train_data = self.load("train_data")
        self.dev_data = self.load("valid_data")
        self.test_data = self.load("test_data")
        self.test_triples_epo = self.load("test_triples_epo")
        self.test_triples_normal = self.load("test_triples_normal")
        self.test_triples_seo = self.load("test_triples_seo")
        self.test_triples_1 = self.load("test_triples_1")
        self.test_triples_2 = self.load("test_triples_2")
        self.test_triples_3 = self.load("test_triples_3")
        self.test_triples_4 = self.load("test_triples_4")
        self.test_triples_5 = self.load("test_triples_5")


    def load(self,dataType):
        sentences = json.load(open(os.path.join(self.file_base_path, dataType + ".json"), 'r'))
        tokens=[]
        entities = []
        sentence_label=[]
        j_file = []

        for idx, item in enumerate(sentences):
            token = item['text'].split(" ")
            for triplet in item["relation_list"]:
                #h_pos = triplet['subj_tok_span']

                h_pos, t_pos, h_name, t_name = self.locate_pos(item['text'],triplet)

                #assert h_name == token[h_pos[0]:h_pos[1]]

                # assert h_name == token[h_pos[0]:h_pos[1]]
                #assert t_name == token[t_pos[0]:t_pos[1]]
                relation = triplet["predicate"]

                res = {
                    "token": token,
                    "h": {"name": h_name, "pos": h_pos},
                    "t": {"name": t_name, "pos": t_pos},
                    "relation": relation
                }
                j_file.append(res)
                #t_pos = [token.index(t_str[0]), token.index(t_str[-1])+1]
                #item['token'] = token
                #item['h']['pos'] = h_pos
                #item['t']['pos'] = t_pos

        return j_file

    def locate_pos(self,text_str, triplet):
        token = text_str.split(" ")
        index = [len(t) for t in token]
        index = [(i+sum(index[0:i]),i+sum(index[0:i+1])) for i in range(len(index))]
        index = {t:id for id, t in enumerate(index)}
        rev_index = {id:t for id, t in enumerate(index)}

        h_name = triplet['subject'].split(" ")
        token_start, token_end = token.index(h_name[0]), token.index(h_name[-1])
        char_start = triplet['subj_char_span'][0]
        char_end = triplet['subj_char_span'][1]
        if char_start not in rev_index[token_start]:
            for (start, end) in index.keys():
                if char_start in [start, end]:
                    token_start = index[(start, end)]
        if char_end not in rev_index[token_end]:
            for (start, end) in index.keys():
                if char_end in [start, end]:
                    token_end = index[(start, end)]
        h_pos = [token_start,token_end+1]
        h_name = " ".join(token[h_pos[0]:h_pos[1]])
        assert h_name == triplet['subject']

        t_name = triplet['object'].split(" ")
        token_start, token_end = token.index(t_name[0]), token.index(t_name[-1])
        char_start = triplet['obj_char_span'][0]
        char_end = triplet['obj_char_span'][1]
        if char_start not in rev_index[token_start]:
            for (start, end) in index.keys():
                if char_start in [start, end]:
                    token_start = index[(start, end)]
        if char_end not in rev_index[token_end]:
            for (start, end) in index.keys():
                if char_end in [start, end]:
                    token_end = index[(start, end)]
        t_pos = [token_start,token_end+1]
        t_name = " ".join(token[t_pos[0]:t_pos[1]])
        assert t_name == triplet['object']

        """
        h_name = triplet["subject"]
        char_start = triplet['subj_char_span'][0]
        char_end = triplet['subj_char_span'][1]
        token_start, token_end = 0, 0
        for (start,end) in index.keys():
            if char_start in [start, end]:
                token_start = index[(start,end)]
            if char_end in [start, end]:
                token_end = index[(start,end)]
        h_pos=[token_start, token_end+1]
        assert " ".join(token[h_pos[0]:h_pos[1]])==h_name

        t_name = triplet["object"]
        char_start = triplet['obj_char_span'][0]
        char_end = triplet['obj_char_span'][1]
        token_start, token_end = 0, 0
        for (start, end) in index.keys():
            if char_start in [start, end]:
                token_start = index[(start, end)]
            if char_end in [start, end]:
                token_end = index[(start, end)]
        t_pos = [token_start, token_end + 1]
        assert " ".join(token[t_pos[0]:t_pos[1]]) == t_name
        """
        return h_pos, t_pos, h_name, t_name
    def convert2relId(self):
        relation_set = set([r["relation"] for r in self.train_data])
        relation_set = relation_set.union(set([r["relation"] for r in self.dev_data]))
        relation_set = relation_set.union(set([r["relation"] for r in self.test_data]))
        rel2id = {name:id for id, name in enumerate(relation_set)}
        file_path = os.path.join('./benchmark', self.dataset_name, "rel2id.json")
        if os.path.exists(file_path):
            pass
        else:
            with open(file_path,'w', encoding="UTF-8") as f:
                f.write(json.dumps(rel2id))

    def dump(self):
        split_file_name = ["train","test","dev",
                            "test_triples_1",
                            "test_triples_2",
                            "test_triples_3",
                            "test_triples_4",
                            "test_triples_5",
                            "test_triples_normal",
                            "test_triples_epo",
                            "test_triples_seo"]
        split_file = [self.train_data,
                      self.test_data,
                      self.dev_data,
                      self.test_triples_1,
                      self.test_triples_2,
                      self.test_triples_3, 
                      self.test_triples_4,
                      self.test_triples_5,
                      self.test_triples_normal,
                      self.test_triples_epo,
                      self.test_triples_seo ]
        
        for dataType,data in zip(split_file_name,split_file):
            file_path = os.path.join('./benchmark', self.dataset_name, dataType + ".txt")
            with open(file_path, 'w', encoding='UTF-8') as f:
                for ele in data:
                    f.write(json.dumps(ele) + "\n")
        self.convert2relId()

class webnlg_start():  #use nyt10 test webnlg cod

    def __init__(self, base_path):
        self.file_base_path = os.path.join(base_path, "webnlg_star","original")
        self.dataset_name = "webnlg_star"
        self.train_data = self.load("train_data")
        self.dev_data = self.load("valid_data")
        self.test_data = self.load("test_data")
        self.test_triples_epo = self.load("test_triples_epo")
        self.test_triples_normal = self.load("test_triples_normal")
        self.test_triples_seo = self.load("test_triples_seo")
        self.test_triples_1 = self.load("test_triples_1")
        self.test_triples_2 = self.load("test_triples_2")
        self.test_triples_3 = self.load("test_triples_3")
        self.test_triples_4 = self.load("test_triples_4")
        self.test_triples_5 = self.load("test_triples_5")


    def load(self,dataType):
        sentences = json.load(open(os.path.join(self.file_base_path, dataType + ".json"), 'r'))
        tokens=[]
        entities = []
        sentence_label=[]
        j_file = []

        for idx, item in enumerate(sentences):
            token = item['text'].split(" ")
            for triplet in item["relation_list"]:
                #h_pos = triplet['subj_tok_span']

                h_pos, t_pos, h_name, t_name = self.locate_pos(item['text'],triplet)

                #assert h_name == token[h_pos[0]:h_pos[1]]

                # assert h_name == token[h_pos[0]:h_pos[1]]
                #assert t_name == token[t_pos[0]:t_pos[1]]
                relation = triplet["predicate"]

                res = {
                    "token": token,
                    "h": {"name": h_name, "pos": h_pos},
                    "t": {"name": t_name, "pos": t_pos},
                    "relation": relation
                }
                j_file.append(res)
                #t_pos = [token.index(t_str[0]), token.index(t_str[-1])+1]
                #item['token'] = token
                #item['h']['pos'] = h_pos
                #item['t']['pos'] = t_pos

        return j_file

    def locate_pos(self,text_str, triplet):
        token = text_str.split(" ")
        index = [len(t) for t in token]
        index = [(i+sum(index[0:i]),i+sum(index[0:i+1])) for i in range(len(index))]
        index = {t:id for id, t in enumerate(index)}
        rev_index = {id:t for id, t in enumerate(index)}

        h_name = triplet['subject'].split(" ")
        token_start, token_end = token.index(h_name[0]), token.index(h_name[-1])
        char_start = triplet['subj_char_span'][0]
        char_end = triplet['subj_char_span'][1]
        if char_start not in rev_index[token_start]:
            for (start, end) in index.keys():
                if char_start in [start, end]:
                    token_start = index[(start, end)]
        if char_end not in rev_index[token_end]:
            for (start, end) in index.keys():
                if char_end in [start, end]:
                    token_end = index[(start, end)]
        h_pos = [token_start,token_end+1]
        h_name = " ".join(token[h_pos[0]:h_pos[1]])
        assert h_name == triplet['subject']

        t_name = triplet['object'].split(" ")
        token_start, token_end = token.index(t_name[0]), token.index(t_name[-1])
        char_start = triplet['obj_char_span'][0]
        char_end = triplet['obj_char_span'][1]
        if char_start not in rev_index[token_start]:
            for (start, end) in index.keys():
                if char_start in [start, end]:
                    token_start = index[(start, end)]
        if char_end not in rev_index[token_end]:
            for (start, end) in index.keys():
                if char_end in [start, end]:
                    token_end = index[(start, end)]
        t_pos = [token_start,token_end+1]
        t_name = " ".join(token[t_pos[0]:t_pos[1]])
        assert t_name == triplet['object']

        """
        h_name = triplet["subject"]
        char_start = triplet['subj_char_span'][0]
        char_end = triplet['subj_char_span'][1]
        token_start, token_end = 0, 0
        for (start,end) in index.keys():
            if char_start in [start, end]:
                token_start = index[(start,end)]
            if char_end in [start, end]:
                token_end = index[(start,end)]
        h_pos=[token_start, token_end+1]
        assert " ".join(token[h_pos[0]:h_pos[1]])==h_name

        t_name = triplet["object"]
        char_start = triplet['obj_char_span'][0]
        char_end = triplet['obj_char_span'][1]
        token_start, token_end = 0, 0
        for (start, end) in index.keys():
            if char_start in [start, end]:
                token_start = index[(start, end)]
            if char_end in [start, end]:
                token_end = index[(start, end)]
        t_pos = [token_start, token_end + 1]
        assert " ".join(token[t_pos[0]:t_pos[1]]) == t_name
        """
        return h_pos, t_pos, h_name, t_name
    def convert2relId(self):
        relation_set = set([r["relation"] for r in self.train_data])
        relation_set = relation_set.union(set([r["relation"] for r in self.dev_data]))
        relation_set = relation_set.union(set([r["relation"] for r in self.test_data]))
        rel2id = {name:id for id, name in enumerate(relation_set)}
        file_path = os.path.join('./benchmark', self.dataset_name, "rel2id.json")
        if os.path.exists(file_path):
            pass
        else:
            with open(file_path,'w', encoding="UTF-8") as f:
                f.write(json.dumps(rel2id))

    def dump(self):
        split_file_name = ["train","test","dev",
                            "test_triples_1",
                            "test_triples_2",
                            "test_triples_3",
                            "test_triples_4",
                            "test_triples_5",
                            "test_triples_normal",
                            "test_triples_epo",
                            "test_triples_seo"]
        split_file = [self.train_data,
                      self.test_data,
                      self.dev_data,
                      self.test_triples_1,
                      self.test_triples_2,
                      self.test_triples_3, 
                      self.test_triples_4,
                      self.test_triples_5,
                      self.test_triples_normal,
                      self.test_triples_epo,
                      self.test_triples_seo ]
        
        for dataType,data in zip(split_file_name,split_file):
            file_path = os.path.join('./benchmark', self.dataset_name, dataType + ".txt")
            with open(file_path, 'w', encoding='UTF-8') as f:
                for ele in data:
                    f.write(json.dumps(ele) + "\n")
        self.convert2relId()

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    #file_path = "./benchmark/semeval/train.json"

    file_path = os.path.join('./benchmark')
    #python preprocess.py kbp37 ./benchmark

    #file_path = os.path.join("./benchmark","kbp37","train.txt")
    #with open(file_path,'r')as f:
    #    lines = f.readlines()
    #print(len(lines))

    #file_path =
    c=CONLL04(file_path)
    c.dump()

    w = webnlg_5019(file_path)
    w.dump()

    w_star = webnlg_start(file_path)
    w_star.dump()
    
    nyt24_star = Nyt24_star(file_path)
    nyt24_star.dump()
    
    #c = CONLL04_ADE_SCIERC(file_path,"ADE")
    #c.dump()
    #d = nyt24(file_path)
    #d.dump()
    
    #file_path = sys.argv[2]
    #k = KBP37(file_path)
    #k.dump()

    #w = WEBNLG_v2_0(file_path)