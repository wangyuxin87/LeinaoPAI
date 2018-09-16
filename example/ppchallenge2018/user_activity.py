import pymysql
import os
import uuid
import datetime
import time
import sys
import traceback
import logging
import logging.config
from resultEvaluation import calculateHandle
from DbHandle import MysqlUtil

logging.config.fileConfig("etc/logger.ini")
logger = logging.getLogger("user_activity")

operate = MysqlUtil()

connection = operate.connectDatabase(os.getenv('DB_HOST'), os.getenv('DB_PORT'), os.getenv('DB_USER'), os.getenv('DB_PWD'), os.getenv('DB_DATABASE'))

cursor = connection.cursor()

operate.configureDatabase(connection, cursor)

uploadPath = '/gshare/_paicontrolcenter0000000000000001/cmmodels/'
while True:
    #查询等待处理的作品，这一过程可以多线程竞争
    try:
        logger.info('start....')
        sql = "SELECT id,active_id,user_id,random_name,original_name FROM `ai_active_work` WHERE state = '等待' order by create_date desc LIMIT 1 for update"
        works_to_handle = operate.searchData(connection, cursor, sql)
        logger.info('query result %s ' % (works_to_handle if works_to_handle else ''))
    except Exception as e:
        logger.error(traceback.format_exc())
        works_to_handle = []

    #没有需要处理的作品就阻塞60秒
    if not works_to_handle:
        logger.info('no works to handle')
        time.sleep(60)
        continue

    for work in works_to_handle:
        logger.info('work id: %s' %(work[0]))
        model_path = uploadPath + work[1] + '/' + work[2] + '/' + work[3]
        (work_id, user_id, file_name, original_name) = (work[0], work[2], work[3], work[4])

        #修改处理作品的状态
        try:
            if not operate.updateData(connection, cursor,
                                      "update ai_active_work set state='计算中' where id ='%s'" % work_id) :
                raise Exception('更新任务状态时出错')
        except Exception as ex:
            logger.error('error when update database: ' + str(ex))
            logger.error(traceback.format_exc())
            # 当数据库状态更新失败时，不计算
            continue

        try:
            logger.info('modelFile path: %s' %(model_path))
            succeed, acc, cost = calculateHandle(model_path)
        except Exception as e:
            logger.error('error when evaluating : ' + str(e))
            logger.error(traceback.format_exc())
            succeed, acc, cost = (False, 0.0000, 0.00)

        try:
            #将作品结果入库
            logger.info('succeed %s, acc %.4f, cost %.2f' % (succeed, acc, cost))
            dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if succeed:
                insertsql_acc = "insert into ai_active_work_score(id,work_id,score_id,score,create_date,update_date) values('%s', '%s', '%s', %.4f, '%s', '%s')" \
                    % (uuid.uuid1(), work_id, '_6894a00620594e40b6408819e9170bbc', acc, dt, dt)
                insertsql_cost = "insert into ai_active_work_score(id,work_id,score_id,score,create_date,update_date) values('%s', '%s', '%s', %.2f, '%s', '%s')" \
                    % (uuid.uuid1(), work_id, '_f51216713029450297e6c2695d80c081', cost, dt, dt)
                if not operate.insertData(connection, cursor, insertsql_acc):
                    raise Exception('插入acc数值时出错')
                if not operate.insertData(connection, cursor, insertsql_cost):
                    raise Exception('插入cost数值时出错')
                if not operate.updateData(connection, cursor,
                                          "update ai_active_work set state='计算完成' where id ='%s'" % work_id) :
                    raise Exception('更新任务状态时出错')
            else:
                if not operate.updateData(connection, cursor,
                                          "update ai_active_work set state='计算失败' where id ='%s'" % work_id) :
                    raise Exception('更新任务状态时出错')
        except Exception as e:
            logger.error('error when update database: ' + str(e))
            logger.error(traceback.format_exc())

cursor.close()