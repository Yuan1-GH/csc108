import oracledb

# 连接数据库
conn = oracledb.connect(
    user="system",       # 当前登录用户名
    password="0000",   # 密码
    dsn="localhost:1521/FREEPDB1"
)

cursor = conn.cursor()

while True:
    user_input = input("请输入要查看表的用户名（输入 exit 退出）：").strip()
    if user_input.lower() == "exit":
        break

    # 获取用户的表
    cursor.execute("""
        SELECT table_name 
        FROM all_tables 
        WHERE owner = :owner
        ORDER BY table_name
    """, owner=user_input.upper())
    tables = cursor.fetchall()
    if not tables:
        print(f"用户 {user_input.upper()} 没有表或你没有权限查看。")
        continue

    print(f"用户 {user_input.upper()} 的表：")
    for t in tables:
        print(f"- {t[0]}")

    table_name = input("请输入要查看内容的表名（输入 skip 跳过）：").strip()
    if table_name.lower() == "skip":
        continue

    try:
        cursor.execute(f"SELECT * FROM {user_input.upper()}.{table_name.upper()}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        print("列名:", columns)
        print("数据：")
        for row in rows:
            print(row)

    except oracledb.DatabaseError as e:
        print("查询出错:", e)

cursor.close()
conn.close()
