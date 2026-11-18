"""Regenerate a concise evidence report from completed feasibility JSON files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
lines = [
    '# LSNet 可行性检查记录', '',
    '本记录仅汇总真实检查输出，不是最终模型比较、正式消融结论或测试集结果。', '',
    '## 已验证的结构与环境', '',
    '- 官方源代码固定到 `cbe737c92b7c43ecf02d08545a07f03f1010177c`；上游文件未修改。',
    '- 本地 CPU 使用已有 PyTorch 2.5.1 和独立目录中的 timm 0.5.4。',
    '- HPC 已验证导入 PyTorch 2.7.1+cu126、timm 0.5.4、torchvision 0.22.1+cu126、Triton 3.3.1。',
    '- 保持 28×28、单通道、像素除以 255；只使用原训练集和验证集。',
    '- 修改前端 stride 后，阶段特征图为 28→14→7→4；原下采样在该输入下为 4→2→1→1。',
    '- CPU 动态聚合参考实现已与独立实现比较前向结果和输入/权重梯度，三个不同尺寸检查通过。',
    '- 核心适配保留官方 LSNet-T 宽度和深度；compact 版本另行标注，不冒充原模型。', '',
    '## 实际小样本训练记录', '',
    '训练损失为该轮在线训练平均值，验证准确率为同一轮结束后的 eval 模式结果。',
    '这些数据只用于发现实现、优化和泛化方面的问题，不能用来评选最终赢家。', '',
    '| 运行 | 版本 | 参数量 | 训练样本 | 轮数 | 首轮→末轮训练损失 | 末轮验证准确率 |',
    '|---|---|---:|---:|---:|---|---:|',
]
gpu_reports = []
statuses = []
for file in sorted((ROOT/'results').glob('*/result.json')):
    result = json.loads(file.read_text())
    statuses.append(f'- `{file.parent.name}`：`{result["status"]}`。')
    if result['args']['device'] == 'cuda':
        gpu_reports.append((file, result))
    for row in result.get('variants', []):
        epochs = row['epochs']
        if not epochs:
            continue
        lines.append('| {} | {} | {:,} | {} | {} | {:.4f}→{:.4f} | {:.1%} |'.format(
            file.parent.name, row['variant'], row['parameters'], result['args']['train_per_class']*10,
            len(epochs), epochs[0]['train_loss'], epochs[-1]['train_loss'], epochs[-1]['validation']['accuracy']))
lines += ['', '### 各检查的原始状态', ''] + statuses
lines += ['', '## 结果解释与边界', '',
    '- 训练损失下降且梯度有限说明这些路径可以优化，不等于具有良好泛化能力。',
    '- `learning_check_failed` 比较的是训练子集在 eval 模式下的初始/最终损失；原宽度 CPU 三轮、320 张图试跑的两个消融未通过此项。不能只引用下降的 train 模式损失而隐藏该结果。',
    '- 三轮、320 张图的小样本检查尤其容易受过拟合和 BatchNorm 运行统计尚未稳定影响；这两种解释尚不能仅凭曲线分开确认。',
    '- 完整适配约 1100 万参数，明显大于原作业 CNN；未来跨模型的准确率比较必须同时报告容量和计算成本。',
    '- 本地为参考算子 CPU 运行，其速度不能代表官方 Triton GPU 实现。',
    '- 移除大核的版本保留相应 BN，只删除深度卷积；3×3 版本只改变该核大小。两者都保留动态聚合。', '',
    '## GPU 检查', '']
if not gpu_reports:
    lines += ['官方 CUDA/Triton 前向与反向一致性、显存、训练速度：**尚未完成**。',
              '已提交 UAEUHPCLB 单 GPU 任务 **614431**（初始 20 分钟，后缩短为 5 分钟时限），最近查询仍在排队。',
              '调度器曾给出 2026-11-04T09:18:26 的预计开始时间；这不是承诺，且未独立确认调度器时区。']
else:
    for file, result in gpu_reports:
        lines += [f'- `{file.relative_to(ROOT)}`：状态 `{result["status"]}`。']
        lines += [f'- 设备：{result["environment"].get("gpu", "unknown")}；配置 `{result["args"]["scale"]}`，算子 `{result["args"]["backend"]}`。']
        checks = result.get('ska_checks', [])
        if checks and all(c.get('official_forward_backward') == 'pass' for c in checks):
            lines += [f'- 官方算子三个形状的前向和输入/权重梯度均通过；最大前向绝对误差 {max(c["max_output_error"] for c in checks):.3g}。']
        lines += ['', '| GPU 版本 | 每轮训练秒数（平均） | PyTorch 分配峰值 MiB | batch 推理 ms |',
                  '|---|---:|---:|---:|']
        for row in result.get('variants', []):
            mean_seconds = sum(e['train_seconds'] for e in row['epochs']) / len(row['epochs'])
            lines += [f'| {row["variant"]} | {mean_seconds:.3f} | {row["peak_allocated_mb"]:.1f} | {row["inference_ms_per_batch"]:.2f} |']
        lines += ['', '上述 GPU 时间对应 1,280 张训练图、batch 64、FP32，排除预热；分配峰值并非 GPU 总占用。',
                  'GPU 与先前 CPU 试跑的数据量不同，不能把准确率差异归因于设备或算子。']
        if 'error' in result:
            lines += ['```', result['error'], '```']
gpu_passed = any(r.get('status') == 'passed' and r['args'].get('backend') == 'official'
                 and len(r.get('ska_checks', [])) == 3
                 and all(c.get('official_forward_backward') == 'pass' for c in r['ska_checks'])
                 and len(r.get('variants', [])) == 3
                 and all(v['finite_gradients'] and v['loss_reduced'] for v in r['variants'])
                 for _, r in gpu_reports)
lines += ['', '## 当前判断', '']
if gpu_passed:
    lines += [
        '**核心 GPU 可行性检查通过，可以继续以 LSNet 为主论文推进正式 reality check。**',
        '保留官方 LSNet-T 宽度和深度的小图像适配已在 GPU 上运行成功，当前没有因运行资源而必须缩小模型的证据。',
        '原 CPU 小样本失败记录仍保留。通过 GPU 检查不代表正式实验准确率有保证，也不等于所有历史试跑均通过。',
        '本轮只有 seed 42、1,280 张训练图、500 张验证图和三轮训练；不能据此宣称某个消融显著更优或 LSNet 优于原 CNN。',
        '全训练集每轮约 45 秒仅是按子集吞吐线性外推，尚未实测；验证、调参、保存和排队时间另计。',
        '下一步应沿用完整划分和预处理，固定调参预算，运行 seeds 42/43/44，之后再进行冻结模型的测试集评价。',
    ]
else:
    lines += [
    '结构适配与 CPU 执行已有实测证据；GPU 可用性和正式实验资源需求需要实际 GPU 输出支持。',
    '当前不应把本次检查称为“全部通过”，也不能据此宣称 LSNet 优于原 CNN 或保证作业满分。',
    '正式选题锁定前，应读取 GPU 检查，并确认足量训练后的验证曲线具有解释价值。']
lines += ['',
    '## 文件与复现', '',
    '- `README.md`：适配原因、操作定义、运行方法、局限。',
    '- `adapter.py`：官方模型适配与 CPU 参考算子。',
    '- `probe.py`：梯度检查、输入检查及训练/验证子集试跑。',
    '- `results/*/result.json`：原始数字与状态；每次运行保留自己的目录。',
    '- `results/*/subset_indices.npz`：准确的子集索引。',
    '- `upstream.json`、`provenance.json`：源代码版本及哈希。',
    '- 远端目录：`/home/700053286/advanced AI/experiments/lsnet_feasibility`。',
    '- 未发布、未向公共服务上传项目或数据。',
]
(ROOT/'FEASIBILITY_REPORT.md').write_text('\n'.join(lines)+'\n')
print(ROOT/'FEASIBILITY_REPORT.md')
