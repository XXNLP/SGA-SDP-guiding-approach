import os.path, math

import en_core_web_sm, spacy,sys


import json
from tqdm import tqdm
from interval import Interval
from pyecharts.charts import Graph, Page
from pyecharts import options as opts
from example import configs as cf
from spacy.tokens import Doc
import networkx as nx


class Paser:
    """
    How to install spacy?
    setup1
    steup2: python -m spacy download en_core_web_sm
    more details reference https://github.com/explosion/spacy-models
    """
    def __init__(self):
        self.paser = spacy.load('en_core_web_sm')

    def paser_print(self, sent):
        nlp = self.paser
        doc = nlp(sent)
        for token in doc:
            print(
                '{0}({1}) <-- {2} -- {3}({4})'.format(token.text, token.tag_, token.dep_, token.head.text,
                                                      token.head.tag_))
    def paser_sentence(self, sent):
        assert sent is not None
        nlp = self.paser
        dep_one = nlp(sent)
        return dep_one

    def paser_sentences(self, sentences):
        """
        input:sentences['this is a sentence.','this is the second sentence.',...]
        output:deps:[Object of dep file,.....]
        this version is no tokenization
        """
        deps = []
        for sent in tqdm(sentences, desc="Parsing file..."):
            #sentence = (' '.join(sent))
            if ''in sent: print(sent)
            doc = Doc(self.paser.vocab, words=sent)
            dep = self.paser(doc)
            deps.append(dep)
        return deps


class Text2Pychart:
    def __init__(self,deps,features):
        self.graph_info_py=self.convert_dep2pygraph(deps)
        self.graph_info_nx=self.convert_dep2graph_info_nx(deps,features)
        pass

    def convert_dep2pygraph(self, doc_deps):
        graph_info_py = []
        for dep in doc_deps:
            #This part creats elements for generating a graph based on pyecharts.
            py_nodes = []
            py_edges = []
            #sent_len = 0
            sent_id = 0
            categories =[{"name": "id"+str(i)} for i in range(0, len(dep))]
            for token in dep:
                symbolSize = 20 if token.i == token.head.i else 10   #root
                py_nodes.append({"name": token.text,
                                 "symbolSize": symbolSize,
                                 "id": token.i,#+sent_len,
                                 "category": sent_id})

                #print({"name": token.text, "symbolSize": 10, "id": token.i+sent_len})

                py_edges.append({"source": token.i,#+sent_len,
                                 "target": token.head.i,#+sent_len,
                                 "value": token.dep_})
                #print({"source": token.head.i+sent_len, "source ": token.head.text, "target": token.i+sent_len, "target": token.text, "value": token.dep_})
                #sent_len = len(py_nodes)
                sent_id += 1
            graph_info_py.append({'nodes': py_nodes, 'edges': py_edges, 'categories': categories})
        return graph_info_py

    def convert_dep2graph_info_nx(self, doc_deps, features, parser='spacy'):
        """
        graph_info_nx = {'nodes':[(word_id,{'name':str, 'symbolSize':int, 'category':int})],
                        'edges':[(source_node_id, target_node_id,{'value':int})],
                        'categories':
                        'edu_id':[(edu_id, {'name':str,})]
                            }
        :param doc_deps:
        :return:
        """
        graph_info_nx = []

        for dep, f in zip(doc_deps, features):
            # This part creats elements for generating a graph based on pyecharts.
            nx_nodes = []
            nx_edges = []
            sent_len = 0
            sent_id = 0
            categories = [{"name": "sent" + str(i)} for i in range(0, len(dep))]
            entity_pos = [Interval(e[0], e[1]) for e in [f['h']['pos'],f['t']['pos']]]  # 用于判断字符是否是entity，如果是entity，后续在可视化时，将字符变大

            for token in dep:
                symbolSize = 20 if token.i == token.head.i else 10
                symbol = "rect" if sum(
                    [token.i + sent_len in i for i in entity_pos]) else "circle"  # 画图时，实体节点形状不一样
                symbolSize = 20 if sum([token.i + sent_len in i for i in entity_pos]) else symbolSize

                #   the format of nx node is like (id,dict(att=)):      (id,{'name':str, 'symbolSize:'int, 'category':sent_id})
                nx_nodes.append((token.i + sent_len,
                                 {'name': token.text, 'symbolSize': symbolSize, 'category': sent_id,
                                  'symbol': symbol, 'ori_id': token.i + sent_len})
                                )
                # print({"name": token.text, "symbolSize": 10, "id": token.i+sent_len})

                # the format of nx edges is (source, target, dict(att=)) (id, {'value':token.dep_})
                nx_edges.append((token.head.i + sent_len,
                                 token.i + sent_len,
                                 {'value': token.dep_})
                                )
                # print({"source": token.head.i+sent_len, "source ": token.head.text, "target": token.i+sent_len, "target": token.text, "value": token.dep_})

            sent_len = len(nx_nodes)
            sent_id += 1
            graph_info_nx.append({'nodes': nx_nodes, 'edges': nx_edges, 'categories': categories})
        graph_info_nx = self.check_graph_info_nx(graph_info_nx)
        return graph_info_nx

    def check_graph_info_nx(self, graph_info_nx):
        """
        check errors in the data,such as:
         1.data out-of-order
         2....
         if needed, more steps could be added.
        :param graph_infp_nx:
        :return:
        """
        for i in range(len(graph_info_nx)):
            graph_info_nx[i]['nodes'] = sorted(graph_info_nx[i]['nodes'])

        return graph_info_nx



    def convert_nx2py_format(self, graph_info_nx):  # [{'nodes':[]}, 'edges':[], ]
        graph_info_py = []
        for one_doc in graph_info_nx:
            py_nodes = []
            py_edges = []
            for item in one_doc['nodes']:
                py_nodes.append({"name": item[1]['name'],
                                 "symbolSize": item[1]['symbolSize'],
                                 "id": item[0],
                                 "category": item[1]['category'],
                                 "symbol": item[1]['symbol']}
                                )
            for item in one_doc['edges']:
                py_edges.append({'source': item[1],  # source node
                                 'target': item[0],  # target node
                                 'value': item[2]['value']}
                                )
            graph_info_py.append({'nodes': py_nodes, 'edges': py_edges, 'categories': one_doc['categories']})

        return graph_info_py

    def construct_nx_structure(self, graph_info_nx):
        graph_nx_for_computation = []
        for item in graph_info_nx:
            g = nx.DiGraph()
            g.add_nodes_from(item['nodes'])
            g.add_edges_from(item['edges'])
            graph_nx_for_computation.append(g)
        return graph_nx_for_computation

    def get_shortest_path(self, features, plan=1):
        """
        out put word path between two entities along the graph
        plan 1: find the shortest path between head(word_1,...word_n) to tail(word_1, word_i)
        in the graph, it shows the path between word_1 to word_i

        :param graph_info_nx: graph_info_nx
        :param entity_pos:[(start, end, entity str)]
        :return:
        """
        graph_nx_for_computation = self.construct_nx_structure(self.graph_info_nx)
        entity_pos = [[f['h']['pos'], f['t']['pos']] for f in features]
        #entity_pos = [item['entity_pos'] for item in features]
        #htr = [item["hts"] for item in features]
        sdp_paths = []
        sdp_info_nx=[]
        roots = []
        count = 0
        unconnected=[]
        for idx, (graph, pos, info_nx) in enumerate(
                tqdm(zip(graph_nx_for_computation, entity_pos, self.graph_info_nx), total=len(features), desc="Generating sdp...")):
            connectivity = nx.is_connected(graph.to_undirected())
            if connectivity is False:
                count = count + 1
                unconnected.append(idx)
                # if ths graph is unconnected, the adj sentence will be connected using each sentence's root node
                isolated_nodes = []
                for edge in graph.edges.data():
                    if edge[0]==edge[1] and edge[2]['value']=='ROOT':
                        isolated_nodes.append(edge[0])
                adj_sent = []
                for i in range(len(isolated_nodes)-1):
                    adj_sent.append((isolated_nodes[i], isolated_nodes[i + 1], {"value": 'adj_sen'}))
                graph.add_edges_from(adj_sent)
                # if the graph is unconnected the original graph will be seen as the sdp

                # for id, value in graph.nodes.data():
                #    if value['ori_id'] in sdp:
                #        nodes.append((maps[id], value))
                """
                ori_nodes = list(graph.nodes)
                sdp_paths.append(ori_nodes)
                for id, value in enumerate(ori_nodes):
                    if value == graph.nodes[value]['ori_id']:
                        nodes.append((value, graph.nodes[value]))

                for start, end, value in graph.edges.data():
                    if start in ori_nodes and end in ori_nodes:
                        edges.append((start, end, value))

                categories = info_nx["categories"]
                sdp_info_nx.append({'nodes': nodes, 'edges': edges, "categories": categories})
                roots.append({i: info_nx["nodes"][i][1]["name"] for i, n in enumerate(info_nx["edges"]) if
                              n[2]["value"] == "ROOT"})  # 抽取每个DEP的root结点输出为{i:word}
                """
            if len(graph.nodes) != len(info_nx["nodes"]):
                print('{0} The graph nodes are not the same to the original file {1}...'.format(count, idx))
            sdp = nx.shortest_path(graph.to_undirected(), source=pos[0][0], target=pos[1][1] - 1)
            sdp_paths.append(sdp)
            nodes, edges = [], []

            # 新的 sdp 的新图需要 重新排序其顺序号用于画图
            maps = {n: i for i, n in enumerate(sdp)}
            #for id, value in graph.nodes.data():
            #    if value['ori_id'] in sdp:
            #        nodes.append((maps[id], value))

            for id, value in enumerate(sdp):
                if value == graph.nodes[value]['ori_id']:
                    nodes.append((maps[value], graph.nodes[value]))

            for start, end, value in graph.edges.data():
                if start in sdp and end in sdp:
                    edges.append((maps[start], maps[end], value))

            categories=info_nx["categories"]
            sdp_info_nx.append({'nodes':nodes,'edges':edges,"categories":categories})
            roots.append({i:info_nx["nodes"][i][1]["name"] for i, n in enumerate(info_nx["edges"]) if n[2]["value"]=="ROOT"})  #抽取每个DEP的root结点输出为{i:word}
        sdp_info_py=self.convert_nx2py_format(sdp_info_nx)
        #print('{0} The graph is not connectivity {1}...'.format(count, idx))
        print("There are {0} sentences are not connected".format(count))
        print("The ids are these below:")
        print(unconnected)
        return sdp_paths, sdp_info_py, roots



    def draw_graph(self, graph_info, path, file_name):
        l = len(graph_info)
        for i in tqdm(range(l), desc="Drawing graphs"):
            nodes = graph_info[i]['nodes']
            edges = graph_info[i]['edges']
            categories = graph_info[i]['categories']
            graph = (
                Graph(init_opts=opts.InitOpts(width="1600px", height="1600px"))
                    .add("",
                         nodes,
                         edges,
                         categories=categories,
                         layout="force",
                         #layout='circular',
                         repulsion=8000,
                         #linestyle_opts=opts.LineStyleOpts(color="source", curve=0.3),
                         linestyle_opts=opts.LineStyleOpts(color="source"),
                         edge_label=opts.LabelOpts(is_show=True, font_size=5, position='middle', formatter="{c}"),
                         edge_symbol=['arrow'],
                         is_rotate_label=True,
                         is_draggable=True)
                    .set_global_opts(title_opts=opts.TitleOpts(title="Graph-基本示例"))
            )

            graph.render(path = path+"/graph_"+file_name+"_"+str(i)+".html")

    def draw_deps2graph(self, path,file_name):
        #self.graph_info_py = self.convert_dep2pygraph(deps)
        self.draw_graph(self.graph_info_py, path, file_name)

    def draw_sdps2graph(self, sdps_info_py, path,file_name):
        #self.graph_info_py = self.convert_dep2pygraph(deps)
        self.draw_graph(sdps_info_py, path, file_name)

def write_sdp_into_orifile_for_big_data(dataset_name, file_type):
    #datasetType = ["train", "test", "dev"] if dataset_name not in ["nyt10", "nyt10_1"] else ["train", "test"]
    file_path = os.path.join('./benchmark', dataset_name, file_type + ".txt")
    out_json_file_path = os.path.join('./benchmark', dataset_name, file_type + ".json")
    if os.path.exists(out_json_file_path):
        print("file exists:"+out_json_file_path)
    else:
        with open(file_path, 'r') as f:
            t = f.readlines()

        t = [json.loads(i) for i in t]
        if len(t)>=20000:
            divide = 100
            datasetType=[file_type+"_"+str(id) for id in range(0,divide)]
            index = [(start, start+1) for start in range(0,divide)]
            interval = math.ceil(len(t)/divide)
            split = [(start*interval, end*interval) if end*interval<len(t) else (start*interval, len(t)) for (start, end) in index]

            for type,i in zip(datasetType,split):
                output_path = os.path.join('./benchmark', dataset_name, type+ ".txt")
                with open(output_path,'w') as f:
                    s = t[i[0]:i[1]]
                    for ele in s:
                        f.write(json.dumps(ele)+"\n")
        
            # 上面的将train 文件分割成小段，后续再来解析，因为文件过大，系统会报错
        else:
            datasetType=[file_type]

        for type in datasetType:
            print("Dealing with {0} dataset {1} file...".format(dataset_name,type))
            txt_file_path = os.path.join('./benchmark', dataset_name, type+".txt")
            with open(txt_file_path) as f:
                j_file = [json.loads(i) for i in f]

            for item in j_file:
                if "token" in item:
                    pass
                else:
                    token = item['text'].split()
                    h = item['h']['pos']
                    h_str = item['text'][h[0]:h[1]].split(" ")
                    h_pos = [token.index(h_str[0]), token.index(h_str[-1]) + 1]

                    t= item['t']['pos']
                    t_str = item['text'][t[0]:t[1]].split(" ")
                    t_pos = [token.index(t_str[0]), token.index(t_str[-1])+1]
                    item['token'] = token
                    item['h']['pos'] = h_pos
                    item['t']['pos'] = t_pos

                    del item["text"]

            sentences = [i["token"] if "token" in i else i["text"].split(" ") for i in j_file ]

            P = Paser()
            deps = P.paser_sentences(sentences)
            G = Text2Pychart(deps, j_file)
            sdps, sdps_info_py, roots = G.get_shortest_path(j_file)
            for idx,(j, sdp, info, root) in enumerate(zip(j_file, sdps, sdps_info_py,roots)):
                if all([j['token'][id]==info['nodes'][i]['name'] for i,id in enumerate(sdp)]):
                    pass
                else:
                    print("wrong")
                j["sdp"] = {i: j["token"][i] for i in sdp}
                j["root"] = root

            
            with open(out_json_file_path,'a',encoding='UTF-8') as f:
                for ele in j_file:
                    f.write(json.dumps(ele)+"\n")
                #json.dump(j_file, f)
            if len(datasetType)>1:
                os.remove(txt_file_path) 




def write_sdp_into_orifile_for_nyt(dataset_name):
    #datasetType = ["train", "test", "dev"] if dataset_name not in ["nyt10", "nyt10_1"] else ["train", "test"]
    divide = 100
    datasetType=['train_'+str(id) for id in range(0,divide)]
    index = [(start, start+1) for start in range(0,divide)]

    file_path = os.path.join('./benchmark', dataset_name, "train" + ".txt")
    with open(file_path, 'r') as f:
        t = f.readlines()

    t = [json.loads(i) for i in t]
    interval = int(len(t)/divide)
    split = [(start*interval, end*interval) if end*interval<len(t) else (start*interval, len(t)) for (start, end) in index]

    for type,i in zip(datasetType,split):
        output_path = os.path.join('./benchmark', dataset_name, type+ ".txt")
        with open(output_path,'w') as f:
            s = t[i[0]:i[1]]
            for ele in s:
                f.write(json.dumps(ele)+"\n")
    
    # 上面的将train 文件分割成小段，后续再来解析，因为文件过大，系统会报错
    datasetType.extend(["dev","test"])
    for type in datasetType:
        print("Dealing with {0} dataset {1} file...".format(dataset_name,type))
        txt_file_path = os.path.join('./benchmark', dataset_name, type+".txt")
        with open(txt_file_path) as f:
            j_file = [json.loads(i) for i in f]

        for item in j_file:
            if "token" in item:
                pass
            else:
                token = item['text'].split()
                h = item['h']['pos']
                h_str = item['text'][h[0]:h[1]].split(" ")
                h_pos = [token.index(h_str[0]), token.index(h_str[-1]) + 1]

                t= item['t']['pos']
                t_str = item['text'][t[0]:t[1]].split(" ")
                t_pos = [token.index(t_str[0]), token.index(t_str[-1])+1]
                item['token'] = token
                item['h']['pos'] = h_pos
                item['t']['pos'] = t_pos

                del item["text"]

        sentences = [i["token"] if "token" in i else i["text"].split(" ") for i in j_file ]

        P = Paser()
        deps = P.paser_sentences(sentences)
        G = Text2Pychart(deps, j_file)
        sdps, sdps_info_py, roots = G.get_shortest_path(j_file)
        for idx,(j, sdp, info, root) in enumerate(zip(j_file, sdps, sdps_info_py,roots)):
            if all([j['token'][id]==info['nodes'][i]['name'] for i,id in enumerate(sdp)]):
                pass
            else:
                print("wrong")
            j["sdp"] = {i: j["token"][i] for i in sdp}
            j["root"] = root
        json_file_path = os.path.join('./benchmark', dataset_name, 'train' + ".json")
        with open(json_file_path,'a',encoding='UTF-8') as f:
            for ele in j_file:
                f.write(json.dumps(ele)+"\n")
            #json.dump(j_file, f)
        if type not in ["dev", "test"]:
            os.remove(txt_file_path) 



def write_sdp_into_orifile(dataset_name):
    datasetType = ["train", "test", "dev"] if dataset_name not in ["nyt10", "nyt10_1"] else ["train", "test"]

    for type in datasetType:
        print("Dealing with {0} dataset {1} file...".format(dataset_name,type))
        file_path = os.path.join('./benchmark', dataset_name, type+".txt")
        with open(file_path) as f:
            j_file = [json.loads(i) for i in f]

        for item in j_file:
            if "token" in item:
                pass
            else:
                item['token']=item["text"].split(" ")
                del item['text']

        sentences = [i["token"] for i in j_file ]
        P = Paser()
        deps = P.paser_sentences(sentences)
        G = Text2Pychart(deps, j_file)
        sdps, sdps_info_py, roots = G.get_shortest_path(j_file)
        for idx,(j, sdp, info, root) in enumerate(zip(j_file, sdps, sdps_info_py,roots)):
            if all([j['token'][id]==info['nodes'][i]['name'] for i,id in enumerate(sdp)]):
                pass
            else:
                print("wrong")
            j["sdp"] = {i: j["token"][i] for i in sdp}
            j["root"] = root
        file_path = os.path.join('./benchmark', dataset_name, type + ".json")
        with open(file_path,'w',encoding='UTF-8') as f:
            for ele in j_file:
                f.write(json.dumps(ele)+"\n")
            #json.dump(j_file, f)



def visualization(dataset_name, draw_tree=False, draw_sdp=False):
    datasetType = ["train", "test", "dev"] if dataset_name not in ["nyt10", "nyt10_1"] else ["train", "test"]
    if draw_tree and draw_sdp:
        for type in datasetType:
            print("Rendering {0} dataset {1} file...".format(dataset_name, type))
            file_path = os.path.join('./benchmark', dataset_name, type+".txt")
            with open(file_path) as f:
                j_file = [json.loads(i) for i in f]
            sentences = [i["token"] for i in j_file]
            P = Paser()
            deps = P.paser_sentences(sentences)
            G = Text2Pychart(deps, j_file)
            sdps, sdps_info_py, _ = G.get_shortest_path(j_file)
            for j, sdp, info in zip(j_file, sdps, sdps_info_py):
                j["sdp"] = {i: j["token"][i] for i in sdp}
            output_path = os.path.join("./visual", dataset_name, type)
            if os.path.exists(output_path):
                pass
            else:
                os.makedirs(output_path)
            G.draw_deps2graph(output_path, "ori")
            G.draw_sdps2graph(sdps_info_py, output_path, "sdp")

def whl_test():
    # dataset_name = sys.argv[1]
    # draw_tree = sys.argv[2] == "original"
    # draw_sdp = sys.argv[3] == "sdp"

    dataset_name = "kbp37"
    # draw_tree = True
    # draw_sdp = True

    # write_sdp_into_orifile(dataset_name)
    # visualization(dataset_name, draw_tree, draw_sdp)
    file_path = os.path.join('./benchmark', dataset_name, "test" + ".txt")
    with open(file_path) as f:
        j_file = [json.loads(i) for i in f]
    sentences = [i["token"] for i in j_file]
    P = Paser()
    deps = P.paser_sentences(sentences)
    G = Text2Pychart(deps, j_file)
    sdps, sdps_info_py = G.get_shortest_path(j_file)
    for idx, (j, sdp, info) in enumerate(zip(j_file, sdps, sdps_info_py)):
        if all([j["token"][id] == info["nodes"][i]["name"] for i, id in enumerate(sdp)]):
            pass
        else:
            print("wrong")
        j["sdp"] = {i: j["token"][i] for i in sdp}
    # G.draw_deps2graph("visual", "ori")
    # G.draw_sdps2graph(sdps_info_py,"visual","sdp")

if __name__ == "__main__":

    # python parse.py semeval original sdp

    dataset_name = sys.argv[1]
    #draw_tree = sys.argv[2] == "original" if len(sys.argv)>2 else False
    #draw_sdp = sys.argv[3] == "sdp" if len(sys.argv)>3 else False
    
    #dataset_name = "nyt24"
    #draw_tree = False
    #draw_sdp = False

    write_sdp_into_orifile_for_big_data(dataset_name,"train")
    write_sdp_into_orifile_for_big_data(dataset_name,"test")
    write_sdp_into_orifile_for_big_data(dataset_name,"dev")
    
    


    #write_sdp_into_orifile_for_nyt(dataset_name)
    #write_sdp_into_orifile(dataset_name)
    #visualization(dataset_name, draw_tree, draw_sdp)
    #test()

