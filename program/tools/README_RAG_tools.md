# RAG 管理与测试工具

在 **Reorganize** 目录下执行以下命令（与主程序共用 `memory_db`）。

## 1. RAG 人为管理（写入 / 删除）

- **写入一条 RAG 知识**（不经过 importance 过滤，直接入库）：
  ```bash
  python tools/rag_manage.py add "用户问题或关键词" "助手回答或知识内容"
  python tools/rag_manage.py add "用户问题或关键词" "助手回答或知识内容" --importance 0.8
  ```
- **按 ID 删除一条**：
  ```bash
  python tools/rag_manage.py delete <memory_id>
  ```
- **列出 RAG 记忆**（分页）：
  ```bash
  python tools/rag_manage.py list
  python tools/rag_manage.py list --limit 20 --offset 0
  ```
- **清空所有 RAG 记忆**（谨慎使用）：
  ```bash
  python tools/rag_manage.py clear
  ```

## 2. RAG 测试 / 对比（独立于 Qwen）

不加载 Qwen，仅做检索与上下文对比，用于查看「使用 RAG」与「未使用 RAG」的差异以及 RAG 检索情况。

- **单次查询**：
  ```bash
  python tools/rag_test_compare.py "你的查询"
  python tools/rag_test_compare.py --query "你的查询" --top-k 5 --threshold 0.7
  ```
- **交互式多轮查询**：
  ```bash
  python tools/rag_test_compare.py --interactive
  ```

输出包括：
- **RAG 检索情况**：每条检索到的记忆的 id、相似度、user_input、assistant_response；
- **对比**：未使用 RAG 时传给模型的上下文 vs 使用 RAG 时拼接到 system prompt 的「相关记忆」片段。
