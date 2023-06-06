import csv
import random
import mysql.connector

# 创建到MySQL的连接
cnx = mysql.connector.connect(user='root', password='citi_brainstorm', host='1.117.207.54', database='brainstorm', port='3306')
cursor = cnx.cursor()

# 打开csv文件
with open('./name_soeid.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过标题行
    for row in reader:
        # 从csv行中分离出名字和soeid
        name_parts = row[1].split(" ")
        name = name_parts[0] + " " + name_parts[1]
        soeid = name_parts[2][1:-1]  # 去掉括号

        # 生成随机年龄
        age = random.randint(25, 65)

        # 创建SQL插入语句
        query = ("INSERT INTO student "
                 "(name, age, soeid) "
                 "VALUES (%s, %s, %s)")

        data_student = (name, age, soeid)

        # 执行插入语句
        cursor.execute(query, data_student)

# 提交事务，如果你不调用这个函数，你的修改不会被保存
cnx.commit()

# 关闭游标和连接
cursor.close()
cnx.close()
