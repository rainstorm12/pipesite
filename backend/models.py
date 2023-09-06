from django.db import models

# Create your models here.
class KnowledgeData(models.Model):
	# 如果没有指定主键的话Django会自动新增一个自增id作为主键 否则应该用primary_key定义主键
    Knowledge = models.CharField(max_length=128, verbose_name='知识')
    # createTime = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    Content = models.CharField(max_length=256, verbose_name='内容')
    
    class Meta:
        db_table = 'backend_knowledge'