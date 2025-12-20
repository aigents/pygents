import sqlite3
import uuid

create_users = '''
CREATE TABLE IF NOT EXISTS users (
id INTEGER PRIMARY KEY,
uuid TEXT NOT NULL,
time_first INTEGER COMMENT 'time of the first activity',
time_last INTEGER COMMENT 'time of the last activity',
posts INTEGER COMMENT 'number of posts',
role TEXT COMMENT 'system|admin/specialist|member/client',
onoff TEXT COMMENT 'on|off',
threshold REAL COMMENT '0.0-1.0 - for user and all his/her chats',
destination TEXT COMMENT 'chat|admin - whether metrics are sent to chat or admin'
metrics  TEXT COMMENT 'list of metrics to detect',
CONSTRAINT uuid_unique UNIQUE (uuid))
'''


create_idx_users = 'CREATE INDEX IF NOT EXISTS idx_users_uuid ON users (uuid);'''


class BotMinderDB:

    def __init__(self,name,namespace):
        self.name = name
        self.namespace = namespace if type(namespace) == uuid.UUID else uuid.UUID(namespace)
        
    def create(self):
        with sqlite3.connect(self.name) as conn:
            cursor = conn.cursor()
            cursor.execute(create_users)
            cursor.execute(create_idx_users)
            conn.commit()

    def drop(self):
        with sqlite3.connect(self.name) as conn:
            cursor = conn.cursor()
            cursor.execute('DROP TABLE users')
            conn.commit()
            
    def add_user(self,user_id):
        with sqlite3.connect(self.name) as conn:
            cursor = conn.cursor()
            users_uuid = uuid.uuid5(self.namespace, str(user_id))
            cursor.execute(f'INSERT INTO users (uuid) VALUES ("{str(users_uuid)}")')
            conn.commit()

    def set_user(self,user_id,**values):
        if len(values) == 0:
            return 0
        with sqlite3.connect(self.name) as conn:
            cursor = conn.cursor()
            users_uuid = uuid.uuid5(self.namespace, str(user_id))
            key_values = []
            for key in values:
                key_value = f'"{values[key]}"' if type(values[key])==str else str(values[key])
                key_values.append(f"{key} = {key_value}")
            values = ', '.join(key_values)
            ret = cursor.execute(f'UPDATE users SET {values} WHERE uuid = "{str(users_uuid)}"')
            conn.commit()
        return ret

    def get_user(self,user_id,columns=('role','onoff','threshold','destination')):
        with sqlite3.connect(self.name) as conn:
            cursor = conn.cursor()
            users_uuid = uuid.uuid5(self.namespace, str(user_id))
            #values = ','.join(('id', 'uuid')+columns)
            values = ','.join(columns)
            cursor.execute(f'SELECT {str(values)} from users WHERE uuid = "{str(users_uuid)}"')
            rows = cursor.fetchall()
            for row in rows:
                user = {}
                for col, val in zip(columns,row):
                    user[col] = val
                return user
        return None

    def get_users(self,user_id,columns=('role','onoff','threshold','destination')):
        with sqlite3.connect(self.name) as conn:
            cursor = conn.cursor()
            users_uuid = uuid.uuid5(self.namespace, str(user_id))
            #values = ','.join(('id', 'uuid')+columns)
            values = ','.join(columns)
            cursor.execute(f'SELECT {str(values)} from users WHERE uuid = "{str(users_uuid)}"')
            rows = cursor.fetchall()
            for row in rows:
                user = {}
                for col, val in zip(columns,row):
                    user[col] = val
                return user
        return None
