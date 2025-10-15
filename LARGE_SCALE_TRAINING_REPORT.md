# 大规模训练流程运行报告

## 执行日期
2025年10月15日

## 数据准备阶段 ✅ 完成

### Step 1: GitHub仓库挖掘
- **状态**: ✅ 成功
- **仓库数量**: 5个
  - psf/requests
  - pallets/flask  
  - django/django
  - scikit-learn/scikit-learn
  - pytorch/pytorch
- **克隆时间**: 11秒
- **结果**: 5/5 成功克隆

### Step 2: Python函数提取
- **状态**: ✅ 成功
- **Python文件数**: 4,187个
- **提取的函数数**: 54,297个（原始）
- **过滤后**: 37,160个（5-400行）
- **去重后**: 36,803个唯一函数
- **包含if语句的函数**: 22,451个
- **平均行数**: 26.7行/函数
- **平均if语句数**: 1.88个/函数

### Step 3: 预训练语料库构建
- **状态**: ✅ 成功
- **函数数量**: 36,803个
- **输出文件**: `data/pretrain_corpus.txt`

### Step 4: Fine-tuning数据集构建
- **状态**: ✅ 成功
- **包含if语句的函数**: 22,451个
- **生成的masked样本**: 30,789个
- **数据集划分**:
  - 训练集: 24,633个样本 (80%)
  - 验证集: 3,078个样本 (10%)
  - 测试集: 3,078个样本 (10%)

### Step 5: Tokenizer训练
- **状态**: ✅ 成功
- **词汇表大小**: 8,000
- **特殊Token**: `<pad>`, `<s>`, `</s>`, `<unk>`, `<mask>`, `<IFMASK>`, `<answer>`
- **输出目录**: `artifacts/tokenizer/`
- **训练时间**: ~11秒

## 模型训练阶段

### Step 6: 预训练 (Pre-training)  
- **状态**: ✅ 成功完成
- **模型架构**: GPT-2 (4层, 4头, 256维embedding)
- **总参数量**: 5,469,696 (5.47M)
- **训练配置**:
  - 样本数: 5,000 (限制以加快训练)
  - Block size: 512 tokens
  - Batch size: 4
  - Epochs: 1
  - Learning rate: 5e-4
  - 最终数据集: 95 blocks (90 train, 5 eval)
- **训练结果**:
  - 训练时间: 31.6秒
  - 训练步数: 23步
  - 训练损失: 8.745
  - 速度: 2.844 samples/sec, 0.727 steps/sec
- **输出目录**: `artifacts/pretrain_gpt2/`

### Step 7: Fine-tuning
- **状态**: 🔄 进行中
- **配置**:
  - 预训练模型: `artifacts/pretrain_gpt2`
  - 训练样本: 24,633
  - 验证样本: 3,078
  - Batch size: 4
  - Epochs: 1
  - Learning rate: 3e-5
  - 预计步数: ~6,158步
- **进度**: 数据集tokenization完成，模型训练已启动
- **后台进程PID**: 54901
- **日志文件**: `finetune_output.log`

## 数据统计总结

| 指标 | 数值 |
|------|------|
| GitHub仓库 | 5个 |
| Python文件 | 4,187个 |
| 提取函数数 | 36,803个 |
| 带if语句的函数 | 22,451个 |
| Masked样本总数 | 30,789个 |
| 训练样本 | 24,633个 |
| 验证样本 | 3,078个 |
| 测试样本 | 3,078个 |
| Tokenizer词汇量 | 8,000 |
| 模型参数量 | 5.47M |

## 模型架构

```
GPT-2 Decoder-only Transformer
- Layers: 4
- Attention Heads: 4  
- Embedding Dimension: 256
- FFN Dimension: 1024
- Max Sequence Length: 1024
- Total Parameters: 5,469,696
```

## 预期最终结果

基于当前数据规模（~25k训练样本），预期模型性能：
- **准确率**: 15-30% (较小规模，训练不充分)
- **如需更好效果**:
  - 增加预训练样本（目前仅5,000，建议使用全部36,803）
  - 增加预训练epochs（目前1，建议3-5）
  - 增加fine-tuning epochs（目前1，建议3-5）
  - 增加模型层数（目前4层，建议6-8层）

## 待完成步骤

- [ ] Fine-tuning完成（进行中）
- [ ] 生成预测（Step 8）
- [ ] 评估结果（Step 9）

## 文件输出位置

- 数据文件: `data/`
- Tokenizer: `artifacts/tokenizer/`
- 预训练模型: `artifacts/pretrain_gpt2/`
- Fine-tuned模型: `artifacts/ifrec_finetuned/` (待完成)
- 预测结果: `predictions.csv` (待完成)
- 日志文件:
  - `pretrain_output.log`
  - `finetune_output.log`

## 命令记录

### 完整的pipeline命令（已执行）

```bash
# 1. 挖掘GitHub仓库
python src/data/mine_github.py --repos-file repos.txt --out-dir data/raw_repos

# 2. 提取函数
python src/data/extract_functions.py --repos-root data/raw_repos --out data/functions.jsonl

# 3. 构建预训练语料库
python src/data/build_pretrain_corpus.py --functions data/functions.jsonl --out data/pretrain_corpus.txt

# 4. 构建fine-tuning数据集
python src/data/build_finetune_dataset.py --functions data/functions.jsonl --out-prefix data/finetune

# 5. 训练tokenizer
python src/tokenizer/train_tokenizer.py --corpus data/pretrain_corpus.txt --vocab-size 8000 --out-dir artifacts/tokenizer

# 6. 预训练GPT-2模型
python src/modeling/pretrain_clm.py --tokenizer artifacts/tokenizer --corpus data/pretrain_corpus.txt --out-dir artifacts/pretrain_gpt2 --n-layer 4 --n-head 4 --n-embd 256 --batch-size 4 --epochs 1 --learning-rate 5e-4 --eval-steps 500 --save-steps 2000 --max-samples 5000

# 7. Fine-tuning (进行中)
python src/modeling/finetune_if_condition.py --tokenizer artifacts/tokenizer --pretrained artifacts/pretrain_gpt2 --train data/finetune_train.jsonl --val data/finetune_val.jsonl --out-dir artifacts/ifrec_finetuned --batch-size 4 --epochs 1 --learning-rate 3e-5 --eval-steps 1500 --save-steps 3000

# 8. 生成预测 (待执行)
python src/modeling/predict.py --tokenizer artifacts/tokenizer --model artifacts/ifrec_finetuned --test data/finetune_test.jsonl --out predictions.csv

# 9. 评估 (待执行)
python src/evaluation/score_predictions.py --csv predictions.csv
```

## 监控Fine-tuning进度

```bash
# 检查进程状态
ps -p $(cat finetune.pid)

# 查看实时日志
tail -f finetune_output.log

# 查看最新进度
tail -30 finetune_output.log
```

## 下一步行动

1. **等待Fine-tuning完成** (可能需要10-30分钟，取决于硬件)
2. **运行预测生成**:
   ```bash
   python src/modeling/predict.py --tokenizer artifacts/tokenizer --model artifacts/ifrec_finetuned --test data/finetune_test.jsonl --out predictions.csv
   ```
3. **评估结果**:
   ```bash
   python src/evaluation/score_predictions.py --csv predictions.csv
   ```

## 优化建议

为了获得更好的结果（35-60%准确率），建议：

1. **增加预训练规模**:
   ```bash
   python src/modeling/pretrain_clm.py \
     --tokenizer artifacts/tokenizer \
     --corpus data/pretrain_corpus.txt \
     --out-dir artifacts/pretrain_gpt2_full \
     --n-layer 6 \
     --n-head 8 \
     --n-embd 512 \
     --batch-size 8 \
     --epochs 5 \
     --learning-rate 5e-4 \
     --max-samples 36803  # 使用全部数据
   ```

2. **增加Fine-tuning训练**:
   ```bash
   python src/modeling/finetune_if_condition.py \
     --tokenizer artifacts/tokenizer \
     --pretrained artifacts/pretrain_gpt2_full \
     --train data/finetune_train.jsonl \
     --val data/finetune_val.jsonl \
     --out-dir artifacts/ifrec_finetuned_full \
     --batch-size 8 \
     --epochs 5 \
     --learning-rate 3e-5
   ```

3. **增加更多GitHub仓库** (在repos.txt中添加10-20个仓库)

---

**报告生成时间**: 2025-10-15 17:15 UTC
**Pipeline状态**: Pre-training完成 ✅ | Fine-tuning进行中 🔄
