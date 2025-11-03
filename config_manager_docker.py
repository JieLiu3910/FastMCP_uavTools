import yaml
import os
from pathlib import Path
from typing import Optional
from pprint import pprint   
  
def load_config(config_file: Optional[str] = None) -> dict:
    """加载 YAML 配置文件"""

    if config_file is None:
        # 从当前文件位置计算项目根目录
        custom_config = '/app/configs/config.yaml'
    else:
        custom_config = config_file

    # 先加载默认配置
    default_config = '/app/configs/config.default.yaml'
    # custom_config = '/app/configs/config.yaml'
    # default_config = '/mnt/docker/FastMCP_uavTools_v1.3/configs/config.default.yaml'
    # custom_config = '/mnt/docker/FastMCP_uavTools_v1.3/configs/config.yaml'

    # 检查配置文件是否存在
    with open(default_config, 'r') as f:
        default_config = yaml.safe_load(f)
    print(f"✅ 成功加载默认配置文件: {default_config}")

    # 再尝试加载用户自定义配置
    try:
        with open(custom_config, 'r') as f:
            custom_config = yaml.safe_load(f)
        
        # 合并配置，自定义配置覆盖默认配置
        def merge_configs(default, custom):
            for key, value in custom.items():
                if isinstance(value, dict) and key in default and isinstance(default[key], dict):
                    merge_configs(default[key], value)
                else:
                    default[key] = value
            return default
        
        final_config = merge_configs(default_config, custom_config)
    except FileNotFoundError:
        print(f"自定义配置文件不存在，使用默认配置!")
        
        # 如果没有自定义配置，则使用默认配置
        final_config = default_config

        print(f"⚙ 工具配置信息如下：!")
        pprint(default_config)

    return final_config


if __name__ == '__main__':

    config = load_config()

    pprint(config)