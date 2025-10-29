import os
from pymilvus import connections, Collection
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pymysql
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_manager import load_config

ROOT_DIR = Path(__file__).parent.parent


def expand_metadata_to_columns(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    将 metadata 字段展开为独立的列

    Args:
        data: Milvus 查询结果列表，每条记录包含 id 和 metadata

    Returns:
        pd.DataFrame: 展开后的 DataFrame，metadata 中的每个键都成为独立的列

    Example:
        输入: [{"id": "001", "metadata": '{"lat": 1.0, "lon": 2.0}'}]
        输出: DataFrame with columns: id, lat, lon
    """

    if not data:
        return pd.DataFrame()

    expanded_data = []

    for record in data:
        # 创建新记录，从 id 开始
        expanded_record = {"id": record.get("id")}

        # 解析 metadata JSON 字符串
        metadata_str = record.get("metadata", "{}")
        try:
            metadata_dict = json.loads(metadata_str)
            # 将 metadata 中的所有键值对添加到记录中
            expanded_record.update(metadata_dict)
        except json.JSONDecodeError as e:
            print(f"⚠️  警告: 记录 {record.get('id')} 的 metadata 解析失败: {e}")
            # 如果解析失败，保留原始 metadata 字符串
            expanded_record["metadata"] = metadata_str

        expanded_data.append(expanded_record)

    # 转换为 DataFrame
    df = pd.DataFrame(expanded_data)

    return df


def export_to_csv(
    collection_name: str = "images_vector_target",
    output_file: Optional[str] = None,
    batch_size: int = 1000,
    expand_metadata: bool = True,
    host: str = "localhost",
    port: str = "19530",
) -> str:
    """
    从 Milvus 导出数据到 CSV 文件

    Args:
        collection_name: Milvus 集合名称    [images_vector_history,images_vector_target]
        output_file: 输出文件名（可选，默认自动生成带时间戳的文件名）
        batch_size: 每批次查询的数据量
        expand_metadata: 是否将 metadata 展开为独立的列
        host: Milvus 服务器地址
        port: Milvus 服务器端口

    Returns:
        str: 保存的文件路径

    Example:
        >>> export_to_csv("images_vector_history", expand_metadata=True)
        '✅ 已保存到: exported_data_history_20251004_212530.csv'
    """

    print(f"\n{'='*60}")
    print(f"📦 开始从 Milvus 导出数据到 CSV")
    print(f"{'='*60}")

    # 连接 Milvus
    try:
        connections.connect("default", host=host, port=port)
        print(f"✅ 成功连接到 Milvus ({host}:{port})")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return None

    # 加载集合
    try:
        collection = Collection(collection_name)
        collection.load()
        print(f"✅ 已加载集合: {collection_name}")
    except Exception as e:
        print(f"❌ 加载集合失败: {e}")
        return None

    print(f"📊 集合记录数: {collection.num_entities:,}")

    # 分批查询数据
    offset = 0
    all_data = []

    print(f"\n🔄 开始查询数据 (每批 {batch_size:,} 条)...")

    while True:
        try:
            res = collection.query(
                expr='id != ""',  # VARCHAR 类型主键
                output_fields=["id", "metadata"],  # 不导出 vector 字段
                limit=batch_size,
                offset=offset,
            )
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            break

        if not res:
            print(f"✅ 批次 {offset // batch_size + 1}: 无更多数据")
            break

        print(f"✅ 批次 {offset // batch_size + 1}: 已获取 {len(res):,} 条记录")
        all_data.extend(res)
        offset += batch_size

        if len(res) < batch_size:
            break

    print(f"\n📦 总共查询到 {len(all_data):,} 条记录")

    if not all_data:
        print("⚠️  没有数据可导出")
        return None

    # 处理数据
    if expand_metadata:
        print("\n🔄 正在展开 metadata 字段...")
        df = expand_metadata_to_columns(all_data)
        print(f"✅ metadata 已展开为 {len(df.columns) - 1} 个独立列")
        print(f"📋 列名: {list(df.columns)}")
    else:
        df = pd.DataFrame(all_data)
        print(f"📋 保持原始格式，列名: {list(df.columns)}")

    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(ROOT_DIR, "data/object_milvus_data", f"exported_{collection_name}_{timestamp}.csv")

    # 保存为 CSV
    print(f"\n📄 正在保存为 CSV 格式...")
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"✅ CSV 文件已保存到: {output_file}")
    print(f"📊 CSV 文件: {len(df):,} 行 × {len(df.columns)} 列")

    return output_file


def import_data2mysql(
    csv_file: str,
    table_name: str,
    database: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    if_exists: str = "append",
    batch_size: int = 1000,
) -> bool:

    """
    将CSV文件数据导入到MySQL数据库
    
    功能特性：
        - 自动检查数据库是否存在，如果不存在则自动创建
        - 根据CSV数据自动推断并创建表结构
        - 支持批量插入，提高导入效率
        - 支持事务回滚，保证数据一致性
    
    Args:
        csv_file: CSV文件路径
        table_name: 目标表名
        database: 数据库名（可选，默认从配置文件读取）
            注意：如果指定的数据库不存在，函数会自动创建
        host: MySQL服务器地址（可选，默认从配置文件读取）
        port: MySQL服务器端口（可选，默认从配置文件读取）
        user: MySQL用户名（可选，默认从配置文件读取）
        password: MySQL密码（可选，默认从配置文件读取）
        if_exists: 表存在时的处理方式
            - 'fail': 如果表存在则报错
            - 'replace': 删除原表并创建新表
            - 'append': 追加数据到现有表（默认）
        batch_size: 批量插入的记录数
    
    Returns:
        bool: 导入是否成功
        
    Example:
        >>> # 导入到新数据库（自动创建）
        >>> import_data2mysql(
        ...     csv_file="exported_images_vector_history_20251005_090541.csv",
        ...     table_name="images_history",
        ...     database="new_database",  # 如果不存在会自动创建
        ...     if_exists="replace"
        ... )
        True
    """
    
    print(f"\n{'='*60}")
    print(f"📥 开始导入CSV数据到MySQL数据库")
    print(f"{'='*60}")
    
    # 加载配置
    try:
        config = load_config()
        milvus_config = config.get("milvus", {})
        mysql_config = config.get("mysql_image", {})
        
        # 使用提供的参数或从配置文件读取
        db_host = host or mysql_config.get("host", "localhost")
        db_port = port or mysql_config.get("port", 3306)
        db_user = user or mysql_config.get("user", "root")
        db_password = password or mysql_config.get("password", "123456")
        db_name = database or mysql_config.get("database", "Object_detection_db")
        
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return False
    
    # 读取CSV文件
    try:
        print(f"\n📂 正在读取CSV文件: {csv_file}")
        df = pd.read_csv(csv_file, encoding="utf-8-sig")
        print(f"✅ 成功读取 {len(df):,} 行 × {len(df.columns)} 列数据")
        print(f"📋 列名: {list(df.columns)}")
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        return False
    
    if df.empty:
        print("⚠️  CSV文件为空，无数据可导入")
        return False
    
    # 第一步：连接MySQL服务器（不指定数据库）检查并创建数据库
    connection = None
    try:
        print(f"\n🔗 正在连接MySQL服务器...")
        connection = pymysql.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            charset="utf8mb4",
            autocommit=False,
        )
        print(f"✅ 成功连接到MySQL服务器 ({db_host}:{db_port})")
        
        # 检查数据库是否存在
        cursor = connection.cursor()
        cursor.execute("SHOW DATABASES")
        existing_databases = [db[0] for db in cursor.fetchall()]
        
        if db_name not in existing_databases:
            print(f"\n⚠️  数据库 '{db_name}' 不存在，正在创建...")
            cursor.execute(f"CREATE DATABASE `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            connection.commit()
            print(f"✅ 数据库 '{db_name}' 创建成功")
        else:
            print(f"✅ 数据库 '{db_name}' 已存在")
        
        # 切换到目标数据库
        cursor.execute(f"USE `{db_name}`")
        print(f"✅ 已切换到数据库 '{db_name}'")
        
    except pymysql.Error as e:
        print(f"❌ 连接MySQL或创建数据库失败: {e}")
        if connection:
            connection.close()
        return False
    
    # 第二步：在目标数据库中创建表并导入数据
    try:
        # 检查表是否存在
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            if if_exists == "fail":
                print(f"❌ 表 {table_name} 已存在，且 if_exists='fail'")
                return False
            elif if_exists == "replace":
                print(f"⚠️  表 {table_name} 已存在，正在删除...")
                cursor.execute(f"DROP TABLE `{table_name}`")
                print(f"✅ 已删除表 {table_name}")
                table_exists = False
            elif if_exists == "append":
                print(f"ℹ️  表 {table_name} 已存在，将追加数据")
        
        # 创建表（如果不存在）
        if not table_exists:
            print(f"\n🔨 正在创建表 {table_name}...")
            
            # 推断字段类型
            column_definitions = []
            for col_name in df.columns:
                dtype = df[col_name].dtype
                
                # 根据pandas数据类型映射到MySQL类型
                if col_name == "id":
                    # id字段设置为主键
                    mysql_type = "VARCHAR(255) PRIMARY KEY"
                elif pd.api.types.is_integer_dtype(dtype):
                    mysql_type = "BIGINT"
                elif pd.api.types.is_float_dtype(dtype):
                    mysql_type = "DOUBLE"
                elif pd.api.types.is_bool_dtype(dtype):
                    mysql_type = "BOOLEAN"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    mysql_type = "DATETIME"
                else:
                    # 字符串类型，检查最大长度
                    max_length = df[col_name].astype(str).str.len().max()
                    if pd.isna(max_length) or max_length == 0:
                        max_length = 255
                    elif max_length > 65535:
                        mysql_type = "TEXT"
                    else:
                        mysql_type = f"VARCHAR({min(int(max_length * 1.5), 65535)})"
                
                column_definitions.append(f"`{col_name}` {mysql_type}")
            
            create_table_sql = f"""
            CREATE TABLE `{table_name}` (
                {', '.join(column_definitions)}
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
            
            cursor.execute(create_table_sql)
            print(f"✅ 表 {table_name} 创建成功")
            print(f"📋 字段定义:\n{chr(10).join(f'  - {defn}' for defn in column_definitions)}")
        
        # 批量插入数据
        print(f"\n📤 正在插入数据 (每批 {batch_size:,} 条)...")
        
        # 替换NaN值为None
        df = df.where(pd.notnull(df), None)
        
        total_rows = len(df)
        inserted_rows = 0
        skipped_rows = 0
        
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            # 构建批量插入SQL
            columns = ", ".join([f"`{col}`" for col in df.columns])
            placeholders = ", ".join(["%s"] * len(df.columns))
            
            # 根据if_exists参数决定使用INSERT还是INSERT IGNORE
            if if_exists == "append":
                # 追加模式使用INSERT IGNORE,遇到重复主键时跳过
                insert_sql = f"INSERT IGNORE INTO `{table_name}` ({columns}) VALUES ({placeholders})"
            else:
                insert_sql = f"INSERT INTO `{table_name}` ({columns}) VALUES ({placeholders})"
            
            # 准备数据
            values = [tuple(row) for row in batch_df.values]
            
            # 执行批量插入
            cursor.executemany(insert_sql, values)
            
            # 获取实际插入的行数
            actual_inserted = cursor.rowcount
            batch_skipped = len(batch_df) - actual_inserted
            
            connection.commit()
            
            inserted_rows += actual_inserted
            skipped_rows += batch_skipped
            
            if batch_skipped > 0:
                print(f"✅ 批次 {i // batch_size + 1}: 已插入 {actual_inserted:,} 条,跳过 {batch_skipped:,} 条重复记录 (总进度: {inserted_rows:,}/{total_rows:,})")
            else:
                print(f"✅ 批次 {i // batch_size + 1}: 已插入 {inserted_rows:,}/{total_rows:,} 条记录")
        
        print(f"\n{'='*60}")
        print(f"🎉 数据导入完成！")
        print(f"📊 总共导入 {inserted_rows:,} 条新记录到表 {table_name}")
        if skipped_rows > 0:
            print(f"⚠️  跳过 {skipped_rows:,} 条重复记录（主键已存在）")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 导入数据时发生错误: {e}")
        if connection:
            connection.rollback()
        return False
        
    finally:
        if connection:
            connection.close()
            print(f"🔌 已关闭数据库连接")


def import_sql_file(
    sql_file: str,
    database: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    encoding: str = "utf-8",
) -> bool:
    """
    从SQL文件导入数据到MySQL数据库
    
    功能特性：
        - 支持读取各种SQL文件（建表语句、数据导入、存储过程等）
        - 自动处理SQL注释（-- 和 /* */ 格式）
        - 智能分割SQL语句（基于分号分隔符）
        - 支持多行SQL语句
        - 自动创建不存在的数据库
        - 支持事务处理和错误回滚
    
    Args:
        sql_file: SQL文件路径
        database: 目标数据库名（可选，默认从配置文件读取）
            注意：如果不存在会自动创建
        host: MySQL服务器地址（可选，默认从配置文件读取）
        port: MySQL服务器端口（可选，默认从配置文件读取）
        user: MySQL用户名（可选，默认从配置文件读取）
        password: MySQL密码（可选，默认从配置文件读取）
        encoding: SQL文件编码格式（默认utf-8）
    
    Returns:
        bool: 导入是否成功
        
    Example:
        >>> # 导入SQL文件到指定数据库
        >>> import_sql_file(
        ...     sql_file="backup_database.sql",
        ...     database="restored_db",
        ... )
        True
        
        >>> # 使用自定义连接参数
        >>> import_sql_file(
        ...     sql_file="data/backup.sql",
        ...     database="my_database",
        ...     host="localhost",
        ...     port=3306,
        ...     encoding="utf-8"
        ... )
        True
    """
    
    print(f"\n{'='*60}")
    print(f"📥 开始从SQL文件导入数据到MySQL数据库")
    print(f"{'='*60}")
    
    # 加载配置
    try:
        config = load_config()
        mysql_config = config.get("mysql_image", {})
        
        # 使用提供的参数或从配置文件读取
        db_host = host or mysql_config.get("host", "localhost")
        db_port = port or mysql_config.get("port", 3306)
        db_user = user or mysql_config.get("user", "root")
        db_password = password or mysql_config.get("password", "123456")
        db_name = database or mysql_config.get("database", "Object_detection_db")
        
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return False
    
    # 读取SQL文件
    try:
        print(f"\n📂 正在读取SQL文件: {sql_file}")
        with open(sql_file, 'r', encoding=encoding) as f:
            sql_content = f.read()
        
        if not sql_content.strip():
            print("⚠️  SQL文件为空")
            return False
            
        print(f"✅ 成功读取SQL文件 ({len(sql_content)} 字符)")
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {sql_file}")
        return False
    except UnicodeDecodeError:
        print(f"❌ 文件编码错误，请尝试其他编码格式（如 'gbk', 'latin1'）")
        return False
    except Exception as e:
        print(f"❌ 读取SQL文件失败: {e}")
        return False
    
    # 预处理SQL内容：移除注释
    def remove_comments(sql_text: str) -> str:
        """移除SQL注释"""
        lines = []
        in_multiline_comment = False
        
        for line in sql_text.split('\n'):
            # 处理多行注释 /* */
            if '/*' in line:
                in_multiline_comment = True
                line = line[:line.index('/*')]
            if '*/' in line:
                in_multiline_comment = False
                line = line[line.index('*/') + 2:]
                
            if in_multiline_comment:
                continue
                
            # 处理单行注释 --
            if '--' in line:
                line = line[:line.index('--')]
                
            # 处理单行注释 #
            if '#' in line:
                line = line[:line.index('#')]
                
            line = line.strip()
            if line:
                lines.append(line)
        
        return '\n'.join(lines)
    
    # 移除注释
    sql_content = remove_comments(sql_content)
    
    # 分割SQL语句（基于分号）
    # 注意：这里使用简单的分号分割，可能不适用于包含分号的字符串
    sql_statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    print(f"📋 解析到 {len(sql_statements)} 条SQL语句")
    
    # 连接MySQL服务器
    connection = None
    try:
        print(f"\n🔗 正在连接MySQL服务器...")
        connection = pymysql.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            charset="utf8mb4",
            autocommit=False,
        )
        print(f"✅ 成功连接到MySQL服务器 ({db_host}:{db_port})")
        
        cursor = connection.cursor()
        
        # 检查数据库是否存在，如果不存在则创建
        cursor.execute("SHOW DATABASES")
        existing_databases = [db[0] for db in cursor.fetchall()]
        
        if db_name not in existing_databases:
            print(f"\n⚠️  数据库 '{db_name}' 不存在，正在创建...")
            cursor.execute(f"CREATE DATABASE `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            connection.commit()
            print(f"✅ 数据库 '{db_name}' 创建成功")
        else:
            print(f"✅ 数据库 '{db_name}' 已存在")
        
        # 切换到目标数据库
        cursor.execute(f"USE `{db_name}`")
        print(f"✅ 已切换到数据库 '{db_name}'")
        
    except pymysql.Error as e:
        print(f"❌ 连接MySQL失败: {e}")
        if connection:
            connection.close()
        return False
    
    # 执行SQL语句
    try:
        print(f"\n📤 开始执行SQL语句...")
        
        executed_count = 0
        failed_count = 0
        
        for i, statement in enumerate(sql_statements, 1):
            # 跳过空语句和USE DATABASE语句（因为已经切换了）
            if not statement or statement.upper().startswith('USE '):
                continue
            
            try:
                cursor.execute(statement)
                executed_count += 1
                
                # 每10条语句提交一次
                if executed_count % 10 == 0:
                    connection.commit()
                    print(f"✅ 已执行 {executed_count}/{len(sql_statements)} 条语句")
                    
            except pymysql.Error as e:
                failed_count += 1
                print(f"⚠️  语句 {i} 执行失败: {str(e)[:100]}")
                # 根据需要决定是否继续执行
                # 这里选择记录错误但继续执行
                continue
        
        # 提交剩余的事务
        connection.commit()
        
        print(f"\n{'='*60}")
        print(f"🎉 SQL文件导入完成！")
        print(f"📊 总共执行 {executed_count} 条语句")
        if failed_count > 0:
            print(f"⚠️  失败 {failed_count} 条语句")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 执行SQL语句时发生错误: {e}")
        if connection:
            connection.rollback()
        return False
        
    finally:
        if connection:
            connection.close()
            print(f"🔌 已关闭数据库连接")


def expor2csv_and_import2mysql(
    milvus_collection_name: str="images_vector_history",
    mysql_database_name: str="test_db",
    mysql_table_name: str="test_milvus_data",
    expand_metadata: bool=True,
    batch_size: int=1000,
    if_exists: str ="append",
    ):
   
    output_file = export_to_csv(
        collection_name=milvus_collection_name,
        expand_metadata=expand_metadata,
        batch_size=batch_size,
    )

    if output_file:
        print(f"\n{'='*60}")
        print(f"🎉 导出完成！")
        print(f"📁 文件: {output_file}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"❌ 导出失败")
        print(f"{'='*60}")
        output_file = None
    
    # ========== 第二步：将CSV数据导入MySQL（可选）==========
    
    # 如果需要将导出的CSV数据导入到MySQL，取消下面的注释
    if output_file:
        print("\n" + "="*60)
        print("🚀 开始导入数据到MySQL")
        print("="*60)
        
        success = import_data2mysql(
            csv_file=output_file,
            table_name=mysql_table_name,  # 目标表名
            database=mysql_database_name,  # 目标数据库（可选，默认从配置读取）
            # if_exists="replace",  # 'fail'/'replace'/'append'
            batch_size=1000,
        )
        
        if success:
            print(f"\n{'='*60}")
            print(f"🎉 MySQL导入完成！")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"❌ MySQL导入失败")
            print(f"{'='*60}")

if __name__ == "__main__":
    
    # 导出Milvus数据到CSV并导入MySQL
    # expor2csv_and_import2mysql(
    #     milvus_collection_name="images_vector_history", 
    #     mysql_table_name="test_milvus_data", 
    #     mysql_database_name="test1",
    # )

    expor2csv_and_import2mysql(
        milvus_collection_name="TW_0829_history", 
        mysql_table_name="history_object_images", 
        mysql_database_name="Object_detection_db",
    )

    expor2csv_and_import2mysql(
        milvus_collection_name="TW_0829_target", 
        mysql_table_name="all_objects_images", 
        mysql_database_name="Object_detection_db",
    )

    # expor2csv_and_import2mysql(
    #     milvus_collection_name="fleet_target", 
    #     mysql_table_name="all_objects_images", 
    #     mysql_database_name="Object_detection_db",
    # )

    # expor2csv_and_import2mysql(
    #     milvus_collection_name="fleet_history", 
    #     mysql_table_name="fleet_history", 
    #     mysql_database_name="Object_detection_db",
    # )


    # 导入SQL文件到MySQL
    # import_sql_file(
    #     sql_file=r"C:\Users\LJ\Desktop\桌面文件夹\sql数据\Equipments_db.sql",
    #     database="test2",
    #     host="localhost",
    #     port=3306,
    #     user="root",
    #     password="123456",
    #     encoding="utf-8",
    # )
    