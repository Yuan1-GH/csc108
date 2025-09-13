import oracledb

# -------------------------------
# 配置区
# -------------------------------
DEV_USER = "YUAN"
DEV_PASSWORD = "0000"
DEV_DSN = "//localhost:1521/FREEPDB1"

TABLESPACE_NAME = "YUAN_DATA"  # 直接使用已有表空间
# -------------------------------

def interactive_session():
    """开发用户交互式操作表"""
    conn = oracledb.connect(user=DEV_USER, password=DEV_PASSWORD, dsn=DEV_DSN)
    cur = conn.cursor()

    # 显示已有表
    print("=== 当前用户已有的表 ===")
    cur.execute("SELECT table_name, tablespace_name FROM user_tables ORDER BY table_name")
    tables = cur.fetchall()
    if tables:
        for tname, tsp in tables:
            print(f"- {tname} (表空间: {tsp})")
    else:
        print("当前用户没有创建任何表。")

    print("\n=== Oracle 交互式表操作 ===")
    table_name = input("请输入要操作的表名: ").strip()
    
    # 创建表
    try:
        cur.execute(f"""
            CREATE TABLE {table_name} (
                id NUMBER PRIMARY KEY,
                name VARCHAR2(50)
            ) TABLESPACE {TABLESPACE_NAME}
        """)
        print(f"表 {table_name} 创建成功")
    except oracledb.DatabaseError as e:
        err_obj, = e.args
        if err_obj.code == 955:  # ORA-00955: 名称已被使用
            print(f"表 {table_name} 已存在")
        else:
            raise
    
    while True:
        print("\n选择操作：")
        print("1. 插入数据")
        print("2. 查询数据")
        print("3. 更新数据")
        print("4. 删除数据")
        print("5. 显示表结构")
        print("0. 退出")
        choice = input("输入操作编号: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            try:
                id_val = int(input("输入 ID: "))
                name_val = input("输入 Name: ")
                cur.execute(f"INSERT INTO {table_name} (id, name) VALUES (:1, :2)", (id_val, name_val))
                conn.commit()
                print("数据插入成功")
            except Exception as e:
                print("插入失败:", e)
        elif choice == "2":
            cur.execute(f"SELECT * FROM {table_name}")
            rows = cur.fetchall()
            if rows:
                print("表内容：")
                for row in rows:
                    print(row)
            else:
                print("表为空")
        elif choice == "3":
            try:
                id_val = int(input("输入要修改的 ID: "))
                name_val = input("输入新的 Name: ")
                cur.execute(f"UPDATE {table_name} SET name=:1 WHERE id=:2", (name_val, id_val))
                conn.commit()
                print("数据更新成功")
            except Exception as e:
                print("更新失败:", e)
        elif choice == "4":
            try:
                id_val = int(input("输入要删除的 ID: "))
                cur.execute(f"DELETE FROM {table_name} WHERE id=:1", (id_val,))
                conn.commit()
                print("数据删除成功")
            except Exception as e:
                print("删除失败:", e)
        elif choice == "5":
            cur.execute(f"SELECT column_name, data_type FROM user_tab_columns WHERE table_name=UPPER('{table_name}')")
            print("表结构：")
            for col, dtype in cur.fetchall():
                print(f"{col} | {dtype}")
        else:
            print("无效选择，请重新输入")
    
    cur.close()
    conn.close()


if __name__ == "__main__":
    interactive_session()
