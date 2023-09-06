import os
os.system('start cmd /k "ping -n 10 127.0.0.1>nul && start http://localhost:5000/#/home"')
os.system('start cmd /k "neo4j console"')
os.system('manage.exe runserver 0.0.0.0:5000 --noreload')
input()