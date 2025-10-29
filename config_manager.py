import yaml
import os
from pathlib import Path
from typing import Optional
from pprint import pprint   
  
def load_config(config_file: Optional[str] = None):
    """加载 YAML 配置文件"""
    # 默认config路径
    if config_file is None:
        # 从当前文件位置计算项目根目录
        current_file_dir = os.path.dirname(__file__)  
        config_file = os.path.join(current_file_dir, 'configs', 'config.yaml') 
        
    # 转换为绝对路径
    config_file = Path(config_file).resolve()
    
    # print(f"🔍 尝试加载配置文件: {config_file}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 成功加载配置文件: {config_file}")
        return config
    
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_file}")
        print(f"📁 当前工作目录: {os.getcwd()}")

    except yaml.YAMLError as e:
        print(f"❌ YAML 配置文件格式错误: {e}")

    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")



if __name__ == '__main__':

    config = load_config()

    pprint(config)