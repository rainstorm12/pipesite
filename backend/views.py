from django.shortcuts import render

# Create your views here.
import json
from py2neo import Node
from django.core import serializers
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from backend.models import KnowledgeData
import uuid
def search_single_node(name,node_type):
    if node_type=='all':
        nodes = GRAPH.run("MATCH (n{name:'"+name+"'}) \
                           RETURN id(n) As id,head(labels(n)) As type,n.name As title").data()
    else:
        nodes = GRAPH.run("MATCH (n:"+node_type+"{name:'"+name+"'}) \
                           RETURN id(n) As id,head(labels(n)) As type,n.name As title").data()
    return nodes

def search_relation_nodes(name,node_type):
    if node_type=='all':
        relations = GRAPH.run("MATCH (n {name:'"+ name +"'})-[r]->(m)  \
                                    RETURN id(r) As id, type(r) As type, \
                                    n.name As title1 ,m.name As title2,\
                                    id(n) As id1 ,id(m) As id2,\
                                    head(labels(n)) As type1, head(labels(m)) As type2").data()
        relations += GRAPH.run("MATCH (n )-[r]->(m{name:'"+ name +"'})  \
                                    RETURN id(r) As id, type(r) As type, \
                                    n.name As title1 ,m.name As title2,\
                                    id(n) As id1 ,id(m) As id2,\
                                    head(labels(n)) As type1, head(labels(m)) As type2").data()
    else:
        relations = GRAPH.run("MATCH (n {name:'"+ name +"'})-[r]->(m)  \
                                    WHERE head(labels(n))= '"+node_type+"'\
                                    RETURN id(r) As id, type(r) As type, \
                                    n.name As title1 ,m.name As title2,\
                                    id(n) As id1 ,id(m) As id2,\
                                    head(labels(n)) As type1, head(labels(m)) As type2").data()
        relations += GRAPH.run("MATCH (n )-[r]->(m{name:'"+ name +"'})  \
                                    WHERE head(labels(m))= '"+node_type+"'\
                                    RETURN id(r) As id, type(r) As type, \
                                    n.name As title1 ,m.name As title2,\
                                    id(n) As id1 ,id(m) As id2,\
                                    head(labels(n)) As type1, head(labels(m)) As type2").data()        
    return relations
def search_relation_withid(id):
    relations = GRAPH.run("MATCH (n)-[r]->(m)  \
                                WHERE id(n)= "+id+"\
                                RETURN id(r) As id, type(r) As type, \
                                n.name As title1 ,m.name As title2,\
                                id(n) As id1 ,id(m) As id2,\
                                head(labels(n)) As type1, head(labels(m)) As type2").data()
    relations += GRAPH.run("MATCH (n )-[r]->(m)  \
                                WHERE id(m)= "+id+"\
                                RETURN id(r) As id, type(r) As type, \
                                n.name As title1 ,m.name As title2,\
                                id(n) As id1 ,id(m) As id2,\
                                head(labels(n)) As type1, head(labels(m)) As type2").data()      
    return relations
def nodes_according_to_relations(all_relations,all_nodes):
    #根据relation匹配nodes
    for item in all_relations:
        node = {}
        node["type"] = item["type1"]
        node["title"] = item["title1"]
        node["id"] = item["id1"]
        if node not in all_nodes:
            all_nodes.append(node)
        node = {}
        node["type"] = item["type2"]
        node["title"] = item["title2"]
        node["id"] = item["id2"]
        if node not in all_nodes:
            all_nodes.append(node)
    return all_nodes

from pipesite.settings import GRAPH
import time
#neo4j数据库
#查询节点
@require_http_methods(["GET"])
def show_neo_old(request):#返回全部数据，太耗内存，已经被弃用
    response = {}
    try:
        t1=time.time()
        #节点数据
        all_nodes = GRAPH.run("MATCH (p) RETURN head(labels(p)) As type, p.name As title, id(p) As id").data()
        #关系数据
        all_relations = GRAPH.run("MATCH (n)-[r]->(m) RETURN id(r) As id, type(r) As type, n.name As title1 ,m.name As title2,id(n) As id1 ,id(m) As id2").data()
        response['nodes'] = all_nodes
        response['links'] = all_relations
        response['respMsg'] = 'success'
        response['respCode'] = '100000'
        print('success!!!!',time.time()-t1)
    except Exception as e:
        response['respMsg'] = str(e)
        response['respCode'] = '899999'
    return JsonResponse(response)

def show_neo(request):
    response = {}
    try:
        t1=time.time()
        show_system = "火灾自动报警系统"
        # 关系数据
        all_relations = GRAPH.run("MATCH (n {name:'"+ show_system+"'})-[r]->(m)  \
                                  RETURN id(r) As id, type(r) As type, \
                                  n.name As title1 ,m.name As title2,\
                                  id(n) As id1 ,id(m) As id2,\
                                  head(labels(n)) As type1, head(labels(m)) As type2").data()
        # node_matcher = NodeMatcher(GRAPH)
        # node = node_matcher.match(show_type[0]).first()
        all_nodes = []
        all_nodes = nodes_according_to_relations(all_relations,all_nodes)
        response['nodes'] = all_nodes
        response['links'] = all_relations
        response['respMsg'] = 'success'
        response['respCode'] = '100000'
        print('success!!!!',time.time()-t1)
    except Exception as e:
        response['respMsg'] = str(e)
        response['respCode'] = '899999'
    return JsonResponse(response)


def search_node(request):
    response = {}
    try:
        t1=time.time()
        node_type = request.GET.get('node_type')
        show_node = request.GET.get('node_name')
        node_attribute = request.GET.get('node_attribute')
        all_nodes = []
        all_relations = []
        if node_attribute == 'fuzzy':
            # 关系数据
            if show_node!='' or show_node=='':#可以设置空值
                node_matcher = NodeMatcher(GRAPH)
                nodes = list(node_matcher.match().where("_.name =~ '.*"+show_node+".*'"))
                for n in nodes:
                    selectnodes = search_single_node(n.__name__,node_type)
                    for sn in selectnodes:
                        if sn not in all_nodes:
                            all_nodes.append(sn)
                    relations = search_relation_nodes(n.__name__,node_type)
                    all_relations += relations
        elif node_attribute == 'precise':
            if show_node!='':
                selectnodes = search_single_node(show_node,node_type)
                all_nodes += selectnodes
                all_relations = search_relation_nodes(show_node,node_type)
        #最后整合所有存在关系的节点
        all_nodes = nodes_according_to_relations(all_relations,all_nodes)
        #返回值
        response['nodes'] = all_nodes
        response['links'] = all_relations
        response['respMsg'] = 'success'
        response['respCode'] = '100000'
        print('success!!!!',time.time()-t1)
    except Exception as e:
        response['respMsg'] = str(e)
        response['respCode'] = '899999'
    return JsonResponse(response)

def search_id(request):
    response = {}
    try:
        t1=time.time()
        node_id = request.GET.get('node_id')
        all_relations = search_relation_withid(str(node_id))
        #根据relation匹配
        all_nodes = nodes_according_to_relations(all_relations,[])
        response['nodes'] = all_nodes
        response['links'] = all_relations
        response['respMsg'] = 'success'
        response['respCode'] = '100000'
        print('success!!!!',time.time()-t1)
    except Exception as e:
        response['respMsg'] = str(e)
        response['respCode'] = '899999'
    return JsonResponse(response)

from py2neo import NodeMatcher
@require_http_methods(["GET"])
def update_Node(request):
    response = {}
    try:
        node_id = request.GET.get('node_id')
        node_name = request.GET.get('node_name')
        node_attribute = request.GET.get('node_attribute')
        #查询节点信息
        node_matcher = NodeMatcher(GRAPH)
        node = node_matcher.get(int(node_id))
        #修改节点数据
        node['name'] = node_name
        #修改节点属性
        #...
        #更新节点
        GRAPH.push(node)
        response['respMsg'] = 'success'
        response['respCode'] = '000004'
    except Exception as e:
        response['respMsg'] = str(e)
        response['respCode'] = '999995'
    return JsonResponse(response)

@require_http_methods(["GET"])
def del_neo_Node(request):
    response = {}
    try:
        node_id = request.GET.get('node_id')
        GRAPH.run("MATCH (n) WHERE id(n)="+node_id+" DETACH DELETE (n)")
        response['respMsg'] = 'success'
        response['respCode'] = '000004'
    except Exception as e:
        response['respMsg'] = str(e)
        response['respCode'] = '999995'
    return JsonResponse(response)

#模型算法类函数
shemas = {"CON":"内容","PRO":"巡查项目","MAI":"维护方法","PAT":"巡检方法","PER":"巡检周期"}
@require_http_methods(["GET"])
def model_test(request):
    from GPLinker_torch.predict_sentence import model_test_GPLinker
    response = {}
    try:
        model_type = request.GET.get('model_type')
        input_sentence = request.GET.get('input_sentence')
        if model_type=='GPLinker':
            threetuple = model_test_GPLinker(input_sentence)
        else:
            threetuple = []
        all_relations = []
        all_nodes = []
        for item in threetuple:
            item['subject_type'] = shemas[item['subject_type']]
            item['object_type'] = shemas[item['object_type']]
        #第一步：整合node
        for item in threetuple:
            node={}
            node['title'] = item['subject']
            node['type'] = item['subject_type']
            if node not in all_nodes:
                all_nodes.append(node)
            node={}
            node['title'] = item['object']
            node['type'] = item['object_type']       
            if node not in all_nodes:
                all_nodes.append(node)
        #第二步：给node赋予id
        for node in all_nodes:
            node['id'] = uuid.uuid1().hex
        #第三步：给relation赋值
        for item in threetuple:
            relation={}
            relation['title1'] = item['subject']
            relation['type1'] = item['subject_type']
            for node in all_nodes:
                if node['type']==relation['type1'] and node['title']==relation['title1']:
                    relation['id1'] = node['id']
                    break
            relation['title2'] = item['object']
            relation['type2'] = item['object_type']
            for node in all_nodes:
                if node['type']==relation['type2'] and node['title']==relation['title2']:
                    relation['id2'] = node['id']
                    break
            #关系选择
            if item['predicate']=='contain':
                if relation['type2']=='巡查项目':
                    relation['type'] = '巡查子项目'
                else:
                    relation['type'] = '包含'
            if item['predicate']=='need':
                relation['type'] = relation['type2']
            relation['id'] = uuid.uuid1().hex
            all_relations.append(relation)
        response['nodes'] = all_nodes
        response['links'] = all_relations
        response['respMsg'] = 'success'
        response['respCode'] = '000010'
    except Exception as e:
        response['respMsg'] = str(e)
        response['respCode'] = '999989'
    return JsonResponse(response)

#创建类
@require_http_methods(["GET"])
def create_new_node(request):
    response = {}
    try:
        node_type = request.GET.get('node_type')
        node_name = request.GET.get('node_name')
        node_attribute = request.GET.get('node_arr')
        all_nodes = []
        if node_name!='':
            all_relations = search_relation_nodes(node_name,node_type)
            selectnodes = search_single_node(node_name,node_type)
            all_nodes += selectnodes
            all_nodes = nodes_according_to_relations(all_relations,all_nodes)
        if len(all_nodes)>0:#这里判定重复的条件仅仅为名字和type不同
            response['message'] = 'IsRepeat'
            response['nodes'] = all_nodes
            response['links'] = all_relations
        else:
            response['message'] = 'IsOK'
            node = {}
            node['type'] = node_type
            node['title'] = node_name
            node['attribute'] = node_attribute
            results=GRAPH.run("create(n:"+node_type+"{name:'"+node_name+"'})return id(n) As id").data()
            node['id'] = results[0]['id']
            response['nodes'] = node
        response['respMsg'] = 'success'
        response['respCode'] = '000005'
    except Exception as e:
        response['respMsg'] = str(e)
        response['respCode'] = '999994'
    return JsonResponse(response)

@require_http_methods(["GET"])
def create_new_link(request):
    response = {}
    response['message'] = ''
    node1_type = request.GET.get('node1_type')
    node1_name = request.GET.get('node1_name')
    node2_type = request.GET.get('node2_type')
    node2_name = request.GET.get('node2_name')
    link_type = request.GET.get('link_type')
    node1_all_nodes = []
    node2_all_nodes = []
    # node1的周边节点
    if node1_name!='':
        node1_relations = search_relation_nodes(node1_name,node1_type)
        selectnodes = search_single_node(node1_name,node1_type)
        node1_all_nodes += selectnodes
        node1_all_nodes = nodes_according_to_relations(node1_relations,node1_all_nodes)
    if node2_name!='':
        node2_relations = search_relation_nodes(node2_name,node2_type)
        selectnodes = search_single_node(node2_name,node2_type)
        node2_all_nodes += selectnodes
        node2_all_nodes = nodes_according_to_relations(node2_relations,node2_all_nodes)
    #没有节点直接返回
    if len(node1_all_nodes)==0:
        response['message'] = 'LackHead'
    if len(node2_all_nodes)==0:
        if response['message'] == 'LackHead':
            response['message'] == 'LackHT'
        else:
            response['message'] == 'LackTail'
    #有节点情况下
    if len(node1_all_nodes)>0 and len(node2_all_nodes)>0:
        #去重复
        for i in range(0,len(node1_all_nodes)):
            node = node1_all_nodes[i]
            if(node2_name==node['title'] and node2_type==node['type']):
                response['message'] = 'IsRepeat'
                node1_all_nodes.pop(i)#去掉node1周边节点中的node2
                for j in range(0,len(node1_relations)):
                    rel = node1_relations[j]
                    if(node2_name==rel['title1'] and node2_type==rel['type1']):
                        node1_relations.pop(j)#去掉node1有关node2的关系属性
                        break
                    if(node2_name==rel['title2'] and node2_type==rel['type2']):
                        node1_relations.pop(j)#去掉node1有关node2的关系属性
                        break
                break
        if(response['message'] == 'IsRepeat'):
            for i in range(0,len(node2_all_nodes)):
                node = node2_all_nodes[i]
                if(node1_name==node['title'] and node1_type==node['type']):
                    node2_all_nodes.pop(i)#去掉node2周边节点中的node1
                    break
        all_nodes = node1_all_nodes+ node2_all_nodes
        all_relations = node1_relations + node2_relations
        #重复情况直接返回
        if(response['message'] == 'IsRepeat'):
            response['nodes'] = all_nodes
            response['links'] = all_relations
        else:
            #无重复情况则同意创建关系请求
            response['message'] = 'IsOK'
            relation = GRAPH.run("MATCH(n:"+node1_type+"),(m:"+node2_type+")\
                        WHERE n.name='"+node1_name+"'AND m.name='"+node2_name+"' \
                        CREATE(n)-[r:"+link_type+"]->(m)\
                        RETURN id(r) As id, type(r) As type, \
                        n.name As title1 ,m.name As title2,\
                        id(n) As id1 ,id(m) As id2,\
                        head(labels(n)) As type1, head(labels(m)) As type2").data()
            #去重复
            new_all_relations = []
            for rel in all_relations:
                if rel not in new_all_relations:
                    new_all_relations.append(rel)
            new_all_nodes = []
            for node in all_nodes:
                if node not in new_all_nodes:
                    new_all_nodes.append(node)
            response['nodes'] = new_all_nodes
            response['links'] = new_all_relations+relation
    response['respMsg'] = 'success'
    response['respCode'] = '000005'
    return JsonResponse(response)

@require_http_methods(["GET"])
def create_new_tuple(request):
    response = {}
    response['message'] = ''
    node1_type = request.GET.get('node1_type')
    node1_name = request.GET.get('node1_name')
    node1_attribute = request.GET.get('node1_arr')
    node2_type = request.GET.get('node2_type')
    node2_name = request.GET.get('node2_name')
    node2_attribute = request.GET.get('node2_arr')
    link_type = request.GET.get('link_type')
    node1_all_nodes = []
    node2_all_nodes = []
    # node1的周边节点
    if node1_name!='':
        node1_relations = search_relation_nodes(node1_name,node1_type)
        selectnodes = search_single_node(node1_name,node1_type)
        node1_all_nodes += selectnodes
        node1_all_nodes = nodes_according_to_relations(node1_relations,node1_all_nodes)
    if node2_name!='':
        node2_relations = search_relation_nodes(node2_name,node2_type)
        selectnodes = search_single_node(node2_name,node2_type)
        node2_all_nodes += selectnodes
        node2_all_nodes = nodes_according_to_relations(node2_relations,node2_all_nodes)
    #有节点情况下
    if len(node1_all_nodes)>0 and len(node2_all_nodes)>0:
        #去重复
        for i in range(0,len(node1_all_nodes)):
            node = node1_all_nodes[i]
            if(node2_name==node['title'] and node2_type==node['type']):
                response['message'] = 'IsRepeat'
                node1_all_nodes.pop(i)#去掉node1周边节点中的node2
                for j in range(0,len(node1_relations)):
                    rel = node1_relations[j]
                    if(node2_name==rel['title1'] and node2_type==rel['type1']):
                        node1_relations.pop(j)#去掉node1有关node2的关系属性
                        break
                    if(node2_name==rel['title2'] and node2_type==rel['type2']):
                        node1_relations.pop(j)#去掉node1有关node2的关系属性
                        break
                break
        if(response['message'] == 'IsRepeat'):
            for i in range(0,len(node2_all_nodes)):
                node = node2_all_nodes[i]
                if(node1_name==node['title'] and node1_type==node['type']):
                    node2_all_nodes.pop(i)#去掉node2周边节点中的node1
                    break
        all_nodes = node1_all_nodes+ node2_all_nodes
        all_relations = node1_relations + node2_relations
        #重复情况直接返回
        if(response['message'] == 'IsRepeat'):
            response['nodes'] = all_nodes
            response['links'] = all_relations
        else:
            #无重复情况则同意创建关系请求
            response['message'] = 'IsOK'
            relation = GRAPH.run("MATCH(n:"+node1_type+"),(m:"+node2_type+")\
                        WHERE n.name='"+node1_name+"'AND m.name='"+node2_name+"' \
                        CREATE(n)-[r:"+link_type+"]->(m)\
                        RETURN id(r) As id, type(r) As type, \
                        n.name As title1 ,m.name As title2,\
                        id(n) As id1 ,id(m) As id2,\
                        head(labels(n)) As type1, head(labels(m)) As type2").data()
            #去重复
            new_all_relations = []
            for rel in all_relations:
                if rel not in new_all_relations:
                    new_all_relations.append(rel)
            new_all_nodes = []
            for node in all_nodes:
                if node not in new_all_nodes:
                    new_all_nodes.append(node)
            response['nodes'] = new_all_nodes
            response['links'] = new_all_relations+relation
    #头节点为新节点时
    if len(node1_all_nodes)==0 and len(node2_all_nodes)>0:
        response['message'] = 'NewHead'
        headnode = {}
        headnode['type'] = node1_type
        headnode['title'] = node1_name
        headnode['attribute'] = node1_attribute
        results=GRAPH.run("create(n:"+node1_type+"{name:'"+node1_name+"'})return id(n) As id").data()
        headnode['id'] = results[0]['id']
        new_relations = GRAPH.run("MATCH(n:"+node1_type+"),(m:"+node2_type+")\
            WHERE n.name='"+node1_name+"'AND m.name='"+node2_name+"' \
            CREATE(n)-[r:"+link_type+"]->(m)\
            RETURN id(r) As id, type(r) As type, \
            n.name As title1 ,m.name As title2,\
            id(n) As id1 ,id(m) As id2,\
            head(labels(n)) As type1, head(labels(m)) As type2").data()
        all_relations = new_relations + node2_relations
        #原先不管node2周边到底有几个节点，现在肯定可以通过allrelation查明
        all_nodes = nodes_according_to_relations(all_relations,[])
        response['nodes'] = all_nodes
        response['links'] = all_relations
    if len(node2_all_nodes)==0 and len(node1_all_nodes)>0:
        response['message'] = 'NewTail'
        tailnode = {}
        tailnode['type'] = node2_type
        tailnode['title'] = node2_name
        tailnode['attribute'] = node2_attribute
        results=GRAPH.run("create(n:"+node2_type+"{name:'"+node2_name+"'})return id(n) As id").data()
        tailnode['id'] = results[0]['id']
        new_relations = GRAPH.run("MATCH(n:"+node1_type+"),(m:"+node2_type+")\
            WHERE n.name='"+node1_name+"'AND m.name='"+node2_name+"' \
            CREATE(n)-[r:"+link_type+"]->(m)\
            RETURN id(r) As id, type(r) As type, \
            n.name As title1 ,m.name As title2,\
            id(n) As id1 ,id(m) As id2,\
            head(labels(n)) As type1, head(labels(m)) As type2").data()
        all_relations = new_relations + node1_relations
        #原先不管node1周边到底有几个节点，现在肯定可以通过allrelation查明
        all_nodes = nodes_according_to_relations(all_relations,[])
        response['nodes'] = all_nodes
        response['links'] = all_relations
    if len(node2_all_nodes)==0 and len(node1_all_nodes)==0:
        response['message'] = 'NewHT'
        headnode = {}
        headnode['type'] = node1_type
        headnode['title'] = node1_name
        headnode['attribute'] = node1_attribute
        results=GRAPH.run("create(n:"+node1_type+"{name:'"+node1_name+"'})return id(n) As id").data()
        headnode['id'] = results[0]['id']
        tailnode = {}
        tailnode['type'] = node2_type
        tailnode['title'] = node2_name
        tailnode['attribute'] = node2_attribute
        results=GRAPH.run("create(n:"+node2_type+"{name:'"+node2_name+"'})return id(n) As id").data()
        tailnode['id'] = results[0]['id']
        new_relations = GRAPH.run("MATCH(n:"+node1_type+"),(m:"+node2_type+")\
            WHERE n.name='"+node1_name+"'AND m.name='"+node2_name+"' \
            CREATE(n)-[r:"+link_type+"]->(m)\
            RETURN id(r) As id, type(r) As type, \
            n.name As title1 ,m.name As title2,\
            id(n) As id1 ,id(m) As id2,\
            head(labels(n)) As type1, head(labels(m)) As type2").data()
        all_relations = new_relations
        all_nodes = nodes_according_to_relations(all_relations,[])
        response['nodes'] = all_nodes
        response['links'] = all_relations
    response['respMsg'] = 'success'
    response['respCode'] = '000005'
    return JsonResponse(response)

#监测界面
@require_http_methods(["GET"])
def show_weather(request):
    response = {}
    try:
        response["weather_state"] = "阴天"
        response["temperature_state"] = 25
        response['respMsg'] = 'success'
        response['respCode'] = '000004'
    except Exception as e:
        response['respMsg'] = str(e)
        response['respCode'] = '999995'
    return JsonResponse(response)

#知识库界面
#增
@require_http_methods(["POST"]) 
@csrf_exempt
def AddKnowledgeData(request):
    response = {}
    try:
        data = json.loads(request.body.decode('utf-8'))
        knowledge = data.get('knowledge')
        content = data.get('content')
        knowledge_data = KnowledgeData(Knowledge=knowledge,Content=content)
        knowledge_data.save()
        response['respMsg'] = 'success'
    except Exception as e:
        response['respMsg'] = str(e)
    return JsonResponse(response)
#删
@require_http_methods(["POST"]) 
@csrf_exempt
def DeleteKnowledgeData(request):
    response = {}
    try:
        data = json.loads(request.body.decode('utf-8'))
        knowledge = data.get('knowledge')
        content = data.get('content')
        pk = data.get('pk')
        knowledge_data = KnowledgeData.objects.filter(id=pk)
        knowledge_data.delete()
        response['respMsg'] = 'success'
    except Exception as e:
        response['respMsg'] = str(e)
    return JsonResponse(response)
#查
@require_http_methods(["GET"])
def ShowKnowledgeData(request):
    response = {}
    try:
        knowledge_data = KnowledgeData.objects.filter()
        response['KnowledgeData'] = json.loads(serializers.serialize("json", knowledge_data))
        response['respMsg'] = 'success'
    except Exception as e:
        response['respMsg'] = str(e)
    return JsonResponse(response)
#改
@require_http_methods(["POST"]) 
@csrf_exempt
def ModifyKnowledgeData(request):
    response = {}
    try:
        data = json.loads(request.body.decode('utf-8'))
        knowledge = data.get('knowledge')
        content = data.get('content')
        pk = data.get('pk')
        knowledge_data = KnowledgeData.objects.filter(id=pk)
        knowledge_data.update(Knowledge=knowledge)
        knowledge_data.update(Content=content)
        response['respMsg'] = 'success'
    except Exception as e:
        response['respMsg'] = str(e)
    return JsonResponse(response)
