# iWiki 课程资源库智能检索系统

后端笔试题（方向 B）实现：文件解析 + OCR + MySQL 关键词检索 + Qdrant 语义检索 + 加权 RRF 融合排序。

## 1. 架构

```
上传/导入 ──> extractor.py（PDF 文本层优先 / 扫描页 OCR / 图片 OCR）──> 文本
                │
                ├─> 分块（~320 字，80 字重叠）
                │
        MySQL  <─┤─> resources（课程/标题/正文/search_text + FULLTEXT ngram 索引）
                │   resource_chunks（块正文 + point_id，作为与 Qdrant 的映射）
                │   search_logs（检索日志，热词统计）
                │
        Qdrant <── embedding.py（DashScope / 离线哈希向量）──> point（id=uuid5(uid,chunk_idx)，
                     payload 内含 resource_id / resource_uid / chunk_index / text）

查询 ──┬─> 模式① 关键词：MySQL FULLTEXT(ngram) BOOLEAN MODE + 精确命中加成
       └─> 模式② 智能：MySQL 召回 ⨁ Qdrant 召回 ──> 资源级聚合 ──> 加权 RRF ──> 统一排序
```

目录结构：

```
backend/
  config.py        配置（.env）
  db.py            MySQL 建表/写入/关键词检索（FULLTEXT ngram 方案）
  extractor.py     PDF/图片解析、OCR、分块
  embedding.py     Embedding 提供方（dashscope / hash）
  vector_store.py  Qdrant 集合管理、写入、删除、检索
  search.py        两种检索模式 + 加权 RRF 融合 + 热词缓存
  pipeline.py      入库流水线（解析→MySQL→Qdrant），更新/删除同步
  demo_data.py     生成演示资料（文本 PDF、扫描 PDF、图片）
  app.py           Flask API + 静态前端
web/index.html     简易前端（上传、两种模式检索、对比、资源列表、热词）
tests/e2e_test.py  端到端测试与对比报告（输出 tests/report.json）
tests/offline_check.py  离线自检（不依赖 MySQL/Qdrant，验证解析与融合逻辑）
```

## 2. 部署配置

### 2.1 MySQL（8.0+）

```sql
-- 容器方式
docker run -d --name iwiki-mysql -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=root -e MYSQL_DATABASE=iwiki mysql:8.0 \
  --character-set-server=utf8mb4 --collation-server=utf8mb4_general_ci
```

本机 Windows 免安装方式（已验证，MySQL 8.4.9）：

```powershell
winget install Oracle.MySQL -e --silent
# 数据目录、配置文件放在纯英文路径下（mysqld 不支持 --defaults-file 含中文路径）
mysqld --initialize-insecure --basedir="C:\Program Files\MySQL\MySQL Server 8.4" --datadir="D:\mysql-data"
# D:\mysql-data\my.ini 中设置 port / datadir / utf8mb4 / ngram_token_size=2，再注册服务
mysqld --install MySQL84 --defaults-file="D:\mysql-data\my.ini"
Start-Service MySQL84
```

关键点：

- 库/表由程序自动创建（`db.init_db()`），**无需手工执行 SQL**。
- 中文检索使用 `FULLTEXT ... WITH PARSER ngram`；`ngram_token_size` 默认 2，程序会自动读取它来构造查询短语。若自行修改该变量，需重建全文索引并重启服务。
- `search_text` 列冗余存放「标题×3 + 课程名×2 + 正文」，让标题/课程命中天然获得更高 TF，从而排在前面。

### 2.2 Qdrant

两种模式（`QDRANT_MODE`）：

- `server`（推荐交付/演示）：`docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant`，然后 `QDRANT_MODE=server QDRANT_URL=http://127.0.0.1:6333`。
- `local`（默认）：`qdrant-client` 本地文件模式（`data/qdrant/`），不需要服务端即可跑通全流程，便于阅卷时零配置启动。

集合由程序自动创建（COSINE 距离，维度取自 Embedding）。**更换 Embedding 维度时请同时更换集合名**，否则会提示维度不一致。

> local 模式下若需彻底清空索引，请先停止服务进程再删除 `data/qdrant/` 目录：进程持有本地库句柄时，删除集合/文件不会立即生效。

### 2.3 OCR

```bash
# Windows 安装 Tesseract，并下载 chi_sim.traineddata 放入 tessdata
# https://github.com/UB-Mannheim/tesseract/wiki
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
OCR_LANG=chi_sim+eng
```

- 判定逻辑：PDF 单页文本少于 `OCR_MIN_TEXT_CHARS`（默认 20）视为扫描页，渲染成图片后 OCR；纯图片直接 OCR。
- 若本机没有 Tesseract：入库不会崩溃，资源状态记为 `empty` 并写明原因（前端与 `/api/resources` 均可见）；也可把 `OCR_ENGINE=pymupdf` 改走 PyMuPDF 内置 OCR（同样依赖 tessdata）。

### 2.3.1 PDF 处理后端

一键安装的依赖即可跑通，`PDF_BACKEND` 控制后端选择：

| 取值 | 文本抽取 | 扫描页渲染 |
| --- | --- | --- |
| `auto`（默认） | pypdf | 优先 PyMuPDF(fitz)，其次 pypdfium2，最后用 pypdf 提取页内内嵌图片 |
| `pypdf` | pypdf | pypdfium2 → pypdf 内嵌图片 |
| `fitz` | pypdf | 仅 PyMuPDF |

说明：`pypdf` 是纯 Python，无需编译、不依赖 DLL，作为文本抽取主力；`pymupdf` 提供 `_mupdf.pyd` 原生模块，若被系统应用控制策略拦截（本机 `ImportError: DLL load failed`），程序会自动跳过并改用 `pypdfium2` 渲染，不影响入库与 OCR 流程。

### 2.4 Embedding

- `EMBEDDING_PROVIDER=dashscope`：阿里云百炼，`DASHSCOPE_API_KEY=sk-xxx`，模型 `text-embedding-v3`（OpenAI 兼容接口，免费额度即可）。
- `EMBEDDING_PROVIDER=hash`（默认）：离线 Hashing Trick 向量（中文按字+二元组），**无需任何 Key**，用于无网络/无 Key 场景跑通全流程，语义能力弱于真实模型，但能直观体现“词面不同也能召回”的差异。

### 2.5 启动

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # 修改 MySQL 密码等
cd backend
..\.venv\Scripts\python app.py
# 打开 http://127.0.0.1:8000 ，点“加载演示数据”即可一键体验
```

## 3. 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/resources/upload` | form-data：`files`(多文件)、`course`，入库（文本 PDF/扫描 PDF/图片） |
| GET | `/api/resources` | 资源列表（含解析状态、块数、OCR 页数） |
| GET | `/api/resources/<id>` | 资源详情 + 文本块 |
| DELETE | `/api/resources/<id>` | 删除资源并同步清理 MySQL chunk 与 Qdrant 向量 |
| GET | `/api/search?mode=keyword\|smart&q=...` | 两种检索模式 |
| GET | `/api/search/compare?q=...` | 同一条查询两种模式并排对比 + 排名差异表 |
| GET | `/api/hot` | 热词排行（带 TTL 缓存） |
| POST | `/api/demo/sample` | 生成并入库演示资料 |
| GET | `/api/health` | MySQL / Qdrant / Embedding 状态与数据量 |

## 4. 检索与融合设计（答题要点）

### 4.1 中文关键词检索方案

MySQL 默认 FULLTEXT 不切中文，整段中文会成为单一 token，召回几乎为 0。因此：

1. 建 `FULLTEXT KEY ft_search_text (search_text) WITH PARSER ngram`，`ngram_token_size=2`（官方推荐的二元组切分）；
2. 查询侧把用户输入切成「英文整词 + 中文短语」，用 `AGAINST('"事务 隔离"' IN BOOLEAN MODE)` 做短语匹配，英文加 `word*` 前缀通配；
3. 过滤“的/了/怎么/如何”等停用词，避免噪声词把相关度带偏；
4. 用 `IN NATURAL LANGUAGE MODE` 取原始相关度 `ft_score`，再叠加精确命中加成（标题 +1.0、课程 +0.6、正文 +0.3，分别 ×10）与标题命中权重；
5. 短查询（单字、ngram 失效）用 `LIKE` 兜底，保证“能搜到”。

返回字段：资源 ID、资源 uid、标题、课程、相关片段（关键词上下文窗口）、`ft_score`、`score`（归一化）。

### 4.2 为什么同时用 MySQL 与 Qdrant

- MySQL 擅长**精确、可解释、强约束**的匹配：课程名、年份、代号、专有名词（如“2021”“innodb_flush_log_at_trx_commit”），词面不命中就是 0 分，不会“幻觉”；
- Qdrant 擅长**语义泛化**：用户说“为什么要避免长时间运行的大事务”，原文写的是“长事务会导致 undo 日志膨胀”，两者词面不重合，但语义一致；
- 两者互补：只用关键词会漏掉同义表达；只用向量在面对精确编号/年份时又会因相似度普遍偏高而排序不准。所以做**混合召回 + 统一融合排序**。

### 4.3 分数融合：加权 RRF

两路分数尺度不同：MySQL 的 `ft_score` 是无上界的正数（受文档长度、TF 影响），Qdrant 是 -1~1 的 cosine 相似度且集中在窄区间。**直接相加会被 MySQL 的量级主导**，所以采用 RRF（只用名次，不用分值）：

```
score(d) = w_kw / (k + rank_kw(d)) + w_sem / (k + rank_sem(d))
k = 60（原论文经验值，弱化头部名次绝对优势，让两路 Top10 都能贡献）
```

权重依据（并实现动态权重加分项）：

- 默认 `w_kw = w_sem = 1.0`；
- 用 `analyze_query()` 判断查询意图：含年份（`\d{4}`）、英文代号、引号短语、或长度 ≤8 且无标点 → 精确倾向，提高 `w_kw`；含疑问词/长句/标点 → 自然语言倾向，提高 `w_sem`；
- 用 `tanh(Δ/8)` 平滑后，权重区间为 `[0.5, 1.5]` 再归一化，避免单条启发式规则让权重剧烈跳变；
- 最终 `score` 做最大归一化只为了展示，不改变排序。

### 4.4 资源级聚合

- 入库时一个文件切成多个块，每块一个 Qdrant point；
- 检索时先在**资源内部**聚合：取该资源最好的一块的名次参与 RRF，其余命中块作为附加上下文，`matched_chunks` 记录命中块数；
- 因此同一文件的多个块只会出现一条结果（题目 B4 要求），不会重复刷分。

### 4.5 索引同步（加分项）

- 重新上传同名同内容文件：`resource_uid = md5(文件名+内容哈希)` 稳定，`update_resource` 覆盖元数据，同时删除旧 chunk 与旧 Qdrant point 再写入新的；
- 删除资源：`DELETE /api/resources/<id>` 在同一流程里删除 MySQL `resource_chunks` 与 Qdrant point（按 `resource_id` payload 过滤 + point_id 双保险）。

### 4.6 热词排行与缓存（加分项）

`search_logs` 记录每次查询与模式，`hot_queries()` 聚合近 30 天 TopN；结果按 `HOT_CACHE_TTL`（默认 60s）缓存，入库/删除时主动失效，避免每次请求都扫日志表。

## 5. 测试与结果

先跑离线自检（无需 MySQL / Qdrant，验证解析、OCR 降级与融合逻辑）：

```bash
cd d:\后端\project
.\.venv\Scripts\python tests\offline_check.py
```

再跑端到端测试（需 MySQL 8 已启动；OCR 需 Tesseract）：

```bash
cd backend
..\.venv\Scripts\python ..\tests\e2e_test.py
```

脚本会：生成 6 份演示资料（4 份文本型 PDF、2 份扫描件，含 1 张 JPG 与 1 份纯图片 PDF）→ 校验解析/OCR 结果 → 入库 → 执行 4 组查询（2 组精确关键词 + 2 组自然语言）→ 打印两种模式 Top5 对比 → 校验删除同步 → 输出 `tests/report.json`。

覆盖点：

| 类型 | 查询 |
| --- | --- |
| 精确课程/年份 | `2021 线性回归` |
| 精确课程关键词 | `事务 隔离级别` |
| 自然语言描述 | `为什么生产环境要尽量避免长时间运行的大事务` |
| 自然语言描述 | `怎样让模型训练收敛更快又不容易震荡` |

OCR 验证：`扫描件-数据结构-考试重点-2023.jpg` 与 `扫描件-计算机组成原理-总线与存储层次.pdf` 走 OCR 通道，入库结果中 `source_type=image-ocr / pdf-ocr`、`ocr_pages>0`，并能被上述检索命中。

> 注：`tests/report.json` 是脚本运行产物（已加入 .gitignore），README 不预填任何虚构的检索结果，请以本地实际运行输出为准。

## 6. 常见问答（面试自述）

- **为什么同时用 MySQL 与 Qdrant？** MySQL 保证精确召回与数据一致性（资源、课程、片段、配额、状态都在一处），Qdrant 负责语义泛化；两者做混合召回，再用 RRF 统一排序，兼顾“能查到”和“查得准”。
- **为什么用 RRF 而不是分数加权和？** 两路分数不可比（无上界的 TF 分 vs 有界的 cosine），归一化依赖单次结果的 min/max，波动大；RRF 只用名次，天然尺度无关，且对异常分值鲁棒。
- **Qdrant 智能检索架构？** `MySQL 先做 ngram 关键词召回 Top30` + `query 向量化后 Qdrant Top30` → 按 `resource_id` 聚合去重（资源内取最佳块）→ 加权 RRF 合并 → 返回统一结构（含 `keyword_rank` / `semantic_rank` / `match_type` 便于解释）→ 写检索日志、刷新热词缓存。
