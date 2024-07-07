import psycopg2

# 建立数据库连接
con = psycopg2.connect(database="TestDataset", user="postgres", password="123456", host="localhost", port="5432")
# 调用游标对象
cur = con.cursor()
# 用cursor中的execute 使用DDL语句创建一个名为 STUDENT 的表,指定表的字段以及字段类型
cur.execute('''CREATE TABLE STUDENT
      (ADMISSION INT PRIMARY KEY     NOT NULL,
      NAME           TEXT            NOT NULL,
      AGE            INT             NOT NULL,
      COURSE        CHAR(50),
      DEPARTMENT        CHAR(50));''')

# 提交更改，增添或者修改数据只会必须要提交才能生效
con.commit()
con.close()
