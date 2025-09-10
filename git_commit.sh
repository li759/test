#!/bin/bash
# 设置你的 Obsidian 笔记库路径
VAULT_PATH="/home/qilin/文档/rl/test"
cd "$VAULT_PATH"
# 获取当前日期和时间
current_date=$(date +"%Y-%m-%d %H:%M:%S")
git add .
git commit -m "Auto commit on closing Obsidian - $current_date"
git push -u origin master 
