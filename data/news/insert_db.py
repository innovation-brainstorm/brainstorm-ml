import csv
import string

import mysql.connector

# 创建到MySQL的连接
cnx = mysql.connector.connect(user='root', password='citi_brainstorm', host='1.117.207.54', database='other_db_mock', port='3306')
cursor = cnx.cursor()

with open('news.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        id, link, headline, category, short_description, authors, date = row
        print(id, link, headline, category, short_description, authors, date)
        link = link.strip()
        headline = headline.strip()
        category = category.strip()
        short_description = short_description.strip()
        authors = authors.strip()
        date = date.strip()

        # 如果字符串为空或只包含特殊字符，则跳过这个行
        if any(not bool(s) or all(c in string.punctuation for c in s) for s in
               [link, headline, category, short_description, authors, date]):
            continue

        sql = """INSERT INTO other_db_mock.news (link, headline, category, short_description, authors, date) VALUES (%s,%s,%s,%s,%s,%s)"""
        cursor.execute(sql, (link, headline, category, short_description, authors, date))

cnx.commit()
cnx.close()