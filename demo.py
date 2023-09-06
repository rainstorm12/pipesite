# 导入pymysql
import pymysql
import datetime

class Database():
    # **config是指连接数据库时需要的参数,这样只要参数传入正确，连哪个数据库都可以
    # 初始化时就连接数据库
    def __init__(self, **config):
        try:
            # 连接数据库的参数我不希望别人可以动，所以设置私有
            self.__conn = pymysql.connect(**config)
            self.__cursor = self.__conn.cursor()
        except Exception as e:
            print("数据库连接失败：\n", e)

    # 查询一条数据
    # 参数：表名table_name,条件factor_str,要查询的字段field 默认是查询所有字段*
    def select_one(self, table_name, factor_str='', field="*"):
        if factor_str == '':
            sql = f"select {field} from {table_name}"
        else:
            sql = f"select {field} from {table_name} where {factor_str}"
        self.__cursor.execute(sql)
        return self.__cursor.fetchone()

    # 查询多条数据
    # 参数：要查询数据的条数num,表名table_name,条件factor_str,要查询的字段field 默认是查询所有字段*
    def select_many(self, num, table_name, factor_str='', field="*"):
        if factor_str == '':
            sql = f"select {field} from {table_name}"
        else:
            sql = f"select {field} from {table_name} where {factor_str}"
        self.__cursor.execute(sql)
        return self.__cursor.fetchmany(num)

    # 查询全部数据
    # 参数：表名table_name,条件factor_str,要查询的字段field 默认是查询所有字段*
    def select_all(self, table_name, factor_str='', field="*"):
        if factor_str == '':
            sql = f"select {field} from {table_name}"
        else:
            sql = f"select {field} from {table_name} where {factor_str}"
        self.__cursor.execute(sql)
        return self.__cursor.fetchall()

    # 新增数据
    def insert(self,table_name, value):
        sql = f"insert into {table_name} values {value}"
        try:
            self.__cursor.execute(sql)
            self.__conn.commit()
            print("插入成功")
        except Exception as e:
            print("插入失败\n", e)
            self.__conn.rollback()

    # 修改数据
    # 参数：表名，set值(可能是一个，也可能是多个，所以用字典)，条件
    def update(self, table_name, val_obl,change_str):
        sql = f"update {table_name} set"
        # set后面应该是要修改的字段，但是可能会修改多个字段的值，所以遍历一下
        # key对应字段的名，val对应字段的值
        for key, val in val_obl.items():
            sql += f" {key} = {val},"
        # 遍历完的最后面会有一个逗号，所以给它切掉，然后再拼接条件
        # !!!空格很重要
        sql = sql[:-1]+" where "+change_str
        try:
            self.__cursor.execute(sql)
            self.__conn.commit()
            print("修改成功")
        except Exception as e:
            print("修改失败\n", e)
            self.__conn.rollback()

    # 删除数据
    def delete(self,table_name, item):
        sql = f"delete from {table_name} where {item}"
        try:
            self.__cursor.execute(sql)
            self.__conn.commit()
            print("删除成功")
        except Exception as e:
            print("删除失败\n", e)
            self.__conn.rollback()

# 定义一个函数
# 这个函数用来创建连接(连接数据库用）
def mysql_db():
    # 连接数据库肯定需要一些参数
    conn = pymysql.connect(
        host="127.0.0.1",
        port=3306,
        database="px01",
        charset="utf8",
        user="root",
        passwd=""
    )
    return conn
def read_sql(conn):
    # 打开数据库可能会有风险，所以添加异常捕捉
    try:
        with conn.cursor() as cursor:
            # 准备SQL语句
            sql = "select * from backend_node"
            # 执行SQL语句
            cursor.execute(sql)
            # 执行完SQL语句后的返回结果都是保存在cursor中
            # 所以要从cursor中获取全部数据
            datas = cursor.fetchall()
            print("获取的数据：\n", datas)
    except Exception as e:
        print("数据库操作异常：\n", e)
    finally:
        # 不管成功还是失败，都要关闭数据库连接
        conn.close()

config = {
    'host':"127.0.0.1",
    'port':3306,
    'database':"px01",
    'charset':"utf8",
    'user':"root",
    'passwd':"",
}


if __name__ == '__main__':
    # conn = mysql_db()
    # read_sql(conn)
    # 实例化时就直接传参数
    db = Database(**config)

    # 查询1条
    # select_one = db.select_one("backend_node")
    # print(select_one)

    # # 查询多条
    # select_many = db.select_many(3, "user")
    # print(select_many)

    # # 查询所有数据(根据条件)
    # select_all = db.select_all("user", "id>10")
    # print(select_all)
    
    nodesData = [
        { 'name': '管廊本体结构', 'type': '系统' ,'attribute':{'年龄':'150','资金':'280'}},
        { 'name': '混凝土管段', 'type': '巡查项目' },
        { 'name': '结构', 'type': '巡查子项目' },
        { 'name': '变形缝', 'type': '巡查子项目' },
        { 'name': '管线引入处', 'type': '巡查子项目' },
        { 'name': '预留孔', 'type': '巡查子项目' },
        { 'name': '支墩', 'type': '巡查子项目' },
        { 'name': '吊装口', 'type': '巡查子项目' },
        { 'name': '逃生口', 'type': '巡查子项目' },
        { 'name': '进（排）风口', 'type': '巡查子项目' },
        { 'name': '无变形、缺损、渗漏、腐蚀、漏筋等', 'type': '内容' },
        { 'name': '无裂缝', 'type': '内容' },
        { 'name': '无变形、渗漏水，止水带无破损', 'type': '内容' },
        { 'name': '无变形、缺损、腐蚀、渗漏等', 'type': '内容' },
        { 'name': '封堵完好，无渗漏', 'type': '内容' },
        { 'name': '无变形、缺损、裂缝、腐蚀等', 'type': '内容' },
        { 'name': '无变形、缺损、堵塞、污浊、覆盖异物，防盗设施完好、无异常进入特征，井口设施不影响交通，已打开井口有防护及警示措施', 'type': '内容' },
        { 'name': '无变形、缺损、堵塞、覆盖异物，通道通畅，无异常进入特征，格栅等金属构配件安装牢固，无受损、锈蚀', 'type': '内容' },
        { 'name': '尺量、观察', 'type': '方法' },
        { 'name': '观察', 'type': '方法' },
        { 'name': '1次/月', 'type': '巡检周期' },
        { 'name': '1次/半年', 'type': '巡检周期' },
        { 'name': '1次/周', 'type': '巡检周期' },
        { 'name': '1次/季', 'type': '巡检周期' },
    ]
    linksData = [
        { 'source': '管廊本体结构', 'target': '混凝土管段', 'type': '巡查项目' },
        { 'source': '混凝土管段', 'target': '结构', 'type': '巡查子内容' },
        { 'source': '混凝土管段', 'target': '变形缝', 'type': '巡查子内容' },
        { 'source': '混凝土管段', 'target': '管线引入处', 'type': '巡查子内容' },
        { 'source': '混凝土管段', 'target': '预留孔', 'type': '巡查子内容' },
        { 'source': '混凝土管段', 'target': '支墩', 'type': '巡查子内容' },
        { 'source': '混凝土管段', 'target': '吊装口', 'type': '巡查子内容' },
        { 'source': '混凝土管段', 'target': '逃生口', 'type': '巡查子内容' },
        { 'source': '混凝土管段', 'target': '进（排）风口', 'type': '巡查子内容' },
        { 'source': '结构', 'target': '无变形、缺损、渗漏、腐蚀、漏筋等','type': '巡查内容' },
        { 'source': '结构', 'target': '无裂缝','type': '巡查内容' },
        { 'source': '变形缝', 'target': '无变形、渗漏水，止水带无破损','type': '巡查内容'  },
        { 'source': '管线引入处', 'target': '无变形、缺损、腐蚀、渗漏等','type': '巡查内容'  },
        { 'source': '预留孔', 'target': '封堵完好，无渗漏','type': '巡查内容'  },
        { 'source': '支墩', 'target': '无变形、缺损、裂缝、腐蚀等','type': '巡查内容' },
        { 'source': '吊装口','target': '无变形、缺损、堵塞、污浊、覆盖异物，防盗设施完好、无异常进入特征，井口设施不影响交通，已打开井口有防护及警示措施','type': '巡查内容'  },
        { 'source': '逃生口','target': '无变形、缺损、堵塞、污浊、覆盖异物，防盗设施完好、无异常进入特征，井口设施不影响交通，已打开井口有防护及警示措施','type': '巡查内容'  },
        { 'source': '进（排）风口','target': '无变形、缺损、堵塞、覆盖异物，通道通畅，无异常进入特征，格栅等金属构配件安装牢固，无受损、锈蚀','type': '巡查内容' },
        { 'source': '无变形、缺损、渗漏、腐蚀、漏筋等', 'target': '尺量、观察','type': '维护方法' },
        { 'source': '无裂缝', 'target': '观察','type': '维护方法' },
        { 'source': '无变形、渗漏水，止水带无破损', 'target': '尺量、观察','type': '维护方法' },
        { 'source': '无变形、缺损、腐蚀、渗漏等', 'target': '尺量、观察','type': '维护方法' },
        { 'source': '封堵完好，无渗漏', 'target': '观察','type': '维护方法' },
        { 'source': '无变形、缺损、裂缝、腐蚀等', 'target': '观察','type': '维护方法' },
        { 'source': '无变形、缺损、堵塞、污浊、覆盖异物，防盗设施完好、无异常进入特征，井口设施不影响交通，已打开井口有防护及警示措施', 'target': '观察','type': '维护方法' },
        { 'source': '无变形、缺损、堵塞、覆盖异物，通道通畅，无异常进入特征，格栅等金属构配件安装牢固，无受损、锈蚀', 'target': '观察','type': '维护方法' },
        { 'source': '无变形、缺损、渗漏、腐蚀、漏筋等', 'target': '1次/月','type': '巡检周期' },
        { 'source': '无裂缝', 'target': '1次/半年','type': '巡检周期' },
        { 'source': '无变形、渗漏水，止水带无破损', 'target': '1次/周','type': '巡检周期' },
        { 'source': '无变形、缺损、腐蚀、渗漏等', 'target': '1次/周','type': '巡检周期' },
        { 'source': '封堵完好，无渗漏', 'target': '1次/周','type': '巡检周期' },
        { 'source': '无变形、缺损、裂缝、腐蚀等', 'target': '1次/季','type': '巡检周期' },
        { 'source': '无变形、缺损、堵塞、污浊、覆盖异物，防盗设施完好、无异常进入特征，井口设施不影响交通，已打开井口有防护及警示措施', 'target': '1次/周','type': '巡检周期' },
        { 'source': '无变形、缺损、堵塞、覆盖异物，通道通畅，无异常进入特征，格栅等金属构配件安装牢固，无受损、锈蚀', 'target': '1次/周','type': '巡检周期' },
    ]
    
    
    # for node in nodesData:
    #     node_name = node['name']
    #     node_type = node['type']
    #     node_att = ''
    #     string_node="'"+node_name+"'"
    #     string_type="'"+node_type+"'"
    #     string_att= "'"+node_att+"'"
    #     string = '('+'null'+','+string_node+','+string_type+','+string_att+')'
    #     db.insert("backend_node",string)
    
    # for link in linksData:
    #     source_node = link['source']
    #     target_node = link['target']
    #     link_type = link['type']

    #     source_node="'"+source_node+"'"
    #     target_node="'"+target_node+"'"
    #     link_type= "'"+link_type+"'"
    #     string = '('+'null'+','+source_node+','+target_node+','+link_type+')'
    #     db.insert("backend_link",string)

    # print(string)
    # 新增一条数据
    # db.insert("backend_node",string)
    
    # 新增多条数据
    # db.insert("user", "(21,'123'),(22,'456')")

    # # 修改一个字段的数据
    db.update("backend_node", {"nodeName": "123"}, "id=17")
    # # 修改多个字段的数据
    # db.update("user", {"id": "23", "name": "12345"}, "id=103")

    # # 删除数据
    # db.delete("backend_node", "nodeName='控制学院'")
