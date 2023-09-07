# old version views.py
# 采用mysql数据库来实现知识图谱的增删查改
# @require_http_methods(["GET"])
# def add_Node(request):
#     response = {}
#     try:
#         node_name = request.GET.get('node_name')
#         node = NodesData(nodeName=node_name)
#         node.save()
#         response['respMsg'] = 'success'
#         response['respCode'] = '000000'
#     except Exception as e:
#         response['respMsg'] = str(e)
#         response['respCode'] = '999999'
#     return JsonResponse(response)

# @require_http_methods(["GET"])
# def add_Link(request):
#     response = {}
#     try:
#         source_name = request.GET.get('source_name')
#         sourcenode = LinksData(nodeName=source_name)
#         sourcenode.save()
#         response['respMsg'] = 'success'
#         response['respCode'] = '000000'
#     except Exception as e:
#         response['respMsg'] = str(e)
#         response['respCode'] = '999999'
#     return JsonResponse(response)

# @require_http_methods(["GET"])
# def show_Nodes(request):
#     response = {}
#     try:
#         nodes = NodesData.objects.filter()
#         response['list'] = json.loads(serializers.serialize("json", nodes))
#         response['respMsg'] = 'success'
#         response['respCode'] = '000001'
#     except Exception as e:
#         response['respMsg'] = str(e)
#         response['respCode'] = '999998'
#     return JsonResponse(response)

# @require_http_methods(["GET"])
# def show_Links(request):
#     response = {}
#     try:
#         links = LinksData.objects.filter()
#         response['list'] = json.loads(serializers.serialize("json", links))
#         print(response)
#         response['respMsg'] = 'success'
#         response['respCode'] = '000001'
#     except Exception as e:
#         response['respMsg'] = str(e)
#         response['respCode'] = '999998'
#     return JsonResponse(response)

# @require_http_methods(["GET"])
# def show_NL(request):
#     response = {}
#     try:
#         nodes = NodesData.objects.filter()
#         response['nodeslist'] = json.loads(serializers.serialize("json", nodes))
#         links = LinksData.objects.filter()
#         response['linkslist'] = json.loads(serializers.serialize("json", links))
#         response['respMsg'] = 'success'
#         response['respCode'] = '000002'
#     except Exception as e:
#         response['respMsg'] = str(e)
#         response['respCode'] = '999997'
#     return JsonResponse(response)

# @require_http_methods(["GET"])
# def del_Nodes(request):
#     response = {}
#     try:
#         nodes = NodesData.objects.all()
#         nodes.delete()
#         response['respMsg'] = 'success'
#         response['respCode'] = '000003'
#     except Exception as e:
#         response['respMsg'] = str(e)
#         response['respCode'] = '999996'
#     return JsonResponse(response)

# @require_http_methods(["GET"])
# def del_Links(request):
#     response = {}
#     try:
#         links = LinksData.objects.all()
#         links.delete()
#         response['respMsg'] = 'success'
#         response['respCode'] = '000003'
#     except Exception as e:
#         response['respMsg'] = str(e)
#         response['respCode'] = '999996'
#     return JsonResponse(response)

# #旧版本（sql为基础数据）编辑服务指令
# @require_http_methods(["GET"])
# def update_Node_sql(request):
#     response = {}
#     try:
#         node_id = request.GET.get('node_id')
#         node_name = request.GET.get('node_name')
#         node_attribute = request.GET.get('node_attribute')
#         #获得节点信息
#         node = NodesData.objects.filter(id=node_id)
#         for n in node:
#             upnode = n.nodeName
#         #修改关系数据
#         links = LinksData.objects.filter(sourceNode=upnode)
#         links.update(sourceNode=node_name)
#         links = LinksData.objects.filter(targetNode=upnode)
#         links.update(targetNode=node_name)
#         #修改节点数据
#         node.update(nodeName=node_name)
#         node.update(attribute=node_attribute)
#         response['respMsg'] = 'success'
#         response['respCode'] = '000004'
#     except Exception as e:
#         response['respMsg'] = str(e)
#         response['respCode'] = '999995'
#     return JsonResponse(response)