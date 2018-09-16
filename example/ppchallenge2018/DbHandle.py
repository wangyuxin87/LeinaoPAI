#coding=utf-8

# 数据库操作相关函数

import pymysql
import re

import logging
import logging.config
logging.config.fileConfig("etc/logger.ini")
logger = logging.getLogger("user_activity")

class MysqlUtil():
    
    def __init__(self):
        pass


    '''
        数据库的连接
    '''
    def connectDatabase(self, myHost, myPort, myUser, myPasswd, myDb):
        try:
            # 连接数据库
            conn = pymysql.connect(
                host = myHost,
                port = myPort,
                user = myUser,
                passwd = myPasswd,
                db = myDb,   # 所连接的数据库必须是支持utf-8编码的！！！！
                use_unicode = True,
                charset = "utf8"
                )
            return conn

        except Exception as e:
            logger.error('数据库连接异常！')
            logger.error(e) # 打印异常信息
            quit();  #退出程序
            return None;


    '''
        数据库的配置，使其支持中文操作
        conn：数据库连接
        cursor：数据库操作执行器
    '''
    def configureDatabase(self, conn, cursor):
        try:
            # 设置数据库的字符集处理，使其支持中文
            cursor.execute('SET NAMES utf8;') 
            cursor.execute('SET CHARACTER SET utf8;')
            cursor.execute('SET character_set_connection=utf8;')
            conn.commit()

        except Exception as e:
            logger.error('数据库设置异常！')


    '''
        删除表格
        conn：数据库连接
        cursor：数据库操作执行器
        tableName：数据表名称
    '''
    def dropTable(self, conn, cursor, tableName):

        # 查找已经存在的表
        try:
            conn.autocommit(0)
            sql_show_table = 'SHOW TABLES;'
            cursor.execute(sql_show_table)
            conn.commit()
            tables = [cursor.fetchall()]
            table_list = re.findall('(\'.*?\')', str(tables))
            table_list = [re.sub("'", '', each) for each in table_list]
            
        except Exception as e:
            logger.error('查询已有的表存在异常！')
            logger.error(e)
            conn.rollback()

        # 删除旧表
        try:
            if tableName in table_list:
                cursor.execute("drop table %s" %(tableName))
            conn.commit()
                
        except Exception as e:
            logger.error('旧表删除异常！')
            logger.error(e)
            conn.rollback()


    '''
        建立新的表格
        conn：数据库连接
        cursor：数据库操作执行器
        sql：相应的sql语句
    '''
    def createTable(self, conn, cursor, sql):
        try:
            conn.autocommit(0)
            cursor.execute(sql)
            conn.commit()
            
        except Exception as e:
            logger.error('表格建立异常！')
            logger.error(e)
            conn.rollback()


    '''
        判断表格中数据的个数
        conn：数据库连接
        cursor：数据库操作执行器
        sql：相应的sql语句
    '''
    def numberOfData(self, conn, cursor, sql):
        try:
            lis = cursor.execute(sql)
            conn.commit()
            return lis
        
        except Exception as e:
            logger.error('数据检索异常！')
            logger.error(e)   # 打印异常信息
            conn.rollback()
            return -1
   


    '''
        查询操作
        conn：数据库连接
        cursor：数据库操作执行器
        sql：相应的sql语句
    '''
    def searchData(self, conn, cursor, sql):
        try:
            conn.autocommit(0)
            aa = cursor.execute(sql)
            info = cursor.fetchmany(aa)
            conn.commit()
            return info
        except Exception as e:
            logger.error('数据库查询异常！')
            logger.error(e)   # 打印异常信息
            conn.rollback()
            return ''

    
    '''
        插入操作
        conn：数据库连接
        cursor：数据库操作执行器
        sql：相应的sql语句
    '''
    def insertData(self, conn, cursor, sql):
        try:
            conn.autocommit(0)
            cursor.execute(sql)
            conn.commit()
            return 'success'
        except Exception as e:
            logger.error('数据插入异常！')
            logger.error(e)
            conn.rollback()
            return ''


    '''
        更新操作
        conn：数据库连接
        cursor：数据库操作执行器
        sql：相应的sql语句
    '''
    def updateData(self, conn, cursor, sql):
        try:
            conn.autocommit(0)
            cursor.execute(sql)
            conn.commit()
            return 'success'
        except Exception as e:
            logger.error('数据更新异常！')
            logger.error(e)
            conn.rollback()
            return ''
            

    '''
        删除操作
        conn：数据库连接
        cursor：数据库操作执行器
        sql：相应的sql语句
    '''
    def deleteData(self, conn, cursor, sql):
        try:
            conn.autocommit(0)
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error('数据删除异常！')
            logger.error(e)
            conn.rollback()
            return ''


        
        







    
