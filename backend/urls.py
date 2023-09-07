# -*- coding: utf-8 -*-
from django.conf.urls import url

from backend.views import *

urlpatterns = [
    # #添加类
    # url("add_Node", add_Node, ),
    # url("add_Link", add_Link, ),
    # #返回整个数据类
    # url("show_Nodes", show_Nodes, ),
    # url("show_Links", show_Links, ),
    # url("show_NL", show_NL, ),
    # #删除类
    # url("del_Links", del_Links,),
    # url("del_Nodes", del_Nodes,),
    #编辑类
    url("update_Node", update_Node,),
    #neo4j
    url("show_neo_old",show_neo_old),
    url("show_neo",show_neo),
    url("del_neo_Node",del_neo_Node),
    #搜索类
    url("search_node",search_node),
    url("search_id",search_id),
    #模型类
    url("model_test",model_test),
    #创建类
    url("create_new_node",create_new_node),
    url("create_new_link",create_new_link),
    url("create_new_tuple",create_new_tuple),
    #监测类
    url("show_weather",show_weather),
    #知识库类
    url("AddKnowledgeData",AddKnowledgeData),
    url("DeleteKnowledgeData",DeleteKnowledgeData),
    url("ShowKnowledgeData",ShowKnowledgeData),
    url("ModifyKnowledgeData",ModifyKnowledgeData),
]

