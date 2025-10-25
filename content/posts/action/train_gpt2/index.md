---
title: '预训练 GPT2'
description: 'aaa'
summary: '动手训练一个 GPT-2'
date: 2025-10-11T13:25:22+08:00
lastmod: 2025-10-11T13:25:22+08:00 #更新时间
draft: true
tags: ['aaa', 'bbbs']
---
## 环境准备

自己试着玩一下预训练 124M 的 GPT-2。

不得不感慨，世界变化真的很快，没想到有一天我也能预训练 GPT-2 了。

这次体验基于 Keller Jordan 的 [moded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) 库。

背景简介：Keller Jordan 就是那位开发了 Muon 优化器而声名大噪的天才少年，这个库也是他做 GPT-2 竞速的库。
这个库基于 [Andrej Karpathy](https://karpathy.ai/) 的 [nanogpt](https://github.com/karpathy/nanoGPT) 修改而来。

首先，下载代码

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git
```

然后安装相应的依赖。在下载数据的时候可能会有问题：

```bash
Retrying in 8s [Retry 5/5].
'(MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_val_000000.bin (Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x7f2bacd99310>: Failed to establish a new connection: [Errno 101] Network is unreachable'))"), '(Request ID: 626b00f7-29c2-4a2b-bcea-3f785c50639a)')' thrown while requesting HEAD https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/fineweb_val_000000.bin
```

这是因为 Hugging Face 的服务器在国外，国内访问不了。
解决的办法很简单，改成国内镜像地址即可：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后就能正常下载了，试下一个还挺快。

```bash
python data/cached_fineweb10B.py 1
fineweb_val_000000.bin: 100%|██████████████████████████████████████████████████████████████████████████████████| 200M/200M [00:44<00:00, 4.48MB/s]
fineweb_train_000001.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 200M/200M [00:43<00:00, 4.61MB/s]
```

每一个 chunk 会有 100M token，容量是 200 M。实际上我们只需要 8 个 chunk 共 1.6G 的数据就够了。
这些数据会放到 `./data/fineweb10B/` 下。
补充一句：fineweb 是 Hugging Face 清洗的一个数据集。

下好了之后直接 `./run.sh` 就可以开始训练了。

```bash
./run.sh 
W1012 12:47:52.381000 285932 site-packages/torch/distributed/run.py:811] 
W1012 12:47:52.381000 285932 site-packages/torch/distributed/run.py:811] *****************************************
W1012 12:47:52.381000 285932 site-packages/torch/distributed/run.py:811] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W1012 12:47:52.381000 285932 site-packages/torch/distributed/run.py:811] *****************************************
logs/c618f880-8b4a-4312-91aa-7b6c8af00890.txt
```

等待结果即可。

## 代码细节

这里学一些他们的代码细节:

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

设置PyTorch使用**可扩展内存段（expandable segments）**的内存分配策略。

---

```python
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
```

针对某些系统上PyTorch CUDA初始化bug的预防性措施。

1. 提前初始化CUDA上下文：在训练开始前强制创建CUDA张量并进行反向传播，确保CUDA运行时环境完全初始化。

2. 避免延迟初始化问题：某些系统上，第一次CUDA操作可能会有较大的延迟，这个操作可以提前"预热"CUDA。

3. 防止内存分配异常：确保PyTorch的CUDA内存分配器在正式训练开始前就处于稳定状态。

这个操作出现在文件开头的导入部分，在设置环境变量 `PYTORCH_CUDA_ALLOC_CONF` 之后，但在导入其他torch模块之前，确保在模型训练开始前CUDA环境已经完全就绪。

这种做法在分布式训练中尤其重要，可以避免不同进程间因CUDA初始化时机不一致导致的同步问题。

### 优化器 DistAdam

没仔细看过 Muon，只知道是 Shampoo 的变种，用了 Newton-Shulz 迭代来近似 Hessian，然后直接优化整个正交矩阵。
（实际上我并不是很能明白这在说什么）
所以我直接跳过，看 DistAdam。

这个 `DistAdam` 是一个分布式（data-parallel 风格，但把通信换成 reduce-scatter + all-gather）实现的 Adam 优化器。核心思想是把每个参数张量在第 0 维上拆成 `world_size` 片：先用 `reduce_scatter` 把各卡梯度平均并把对应片发到每个 rank，上面局部更新该片（节省通信量和显存峰值），然后用 `all_gather_into_tensor` 把更新后的片汇回到完整参数张量。

```py
class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal
```

`DistAdam` 继承自 `torch.optim.Optimizer` ，是 pytorch 的自定义优化器类。

`defaults = dict(...)` : 把默认超参放到 optimizer 的 defaults（这是 `Optimizer` 要的格式）

然后为**每种参数形状**创建一组参数（以便复用通信缓冲区 / batch 相同形状的参数一起操作），减少内存/通信开销。这是常见的优化（减少不同尺寸造成的碎片化）。

`param_groups = []` 和 for 循环：按形状把参数分组，`param_groups` 中每个 dict 只有 `params` 字段（其余超参由 `defaults` 补全）。实际上 `Optimizer` 支持 param groups，每组都可以有不同 lr / betas。如果你想按形状共享通信缓冲区或将来按组定制超参，这种分组很有用。

---

```py
    @torch.compile
    @torch.no_grad()
    def step(self):
```

`@torch.no_grad()`：是因为在 `step` 中对参数做的是 in-place 更新（`p_slice.add_(...)` 等），所以如果不加这条会被 autograd 记录。
但是这条计算其实并不在计算图之中，它只是优化器中的计算，所以我们不希望这些操作被 autograd 记录。

`@torch.compile`：是把 `step` 编译为更快的内核（TorchDynamo / Inductor 等），以获得更高性能。

---

```py
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
```

* `rank` / `world_size`：当前进程的分布式 rank 与总进程数，用于后续把张量按第0维切片（`p.shape[0] // world_size`）并知道当前进程处理哪一段。
* `reduce_scatter_futures` / `all_reduce_futures`：存放异步通信返回的 `Future`，先发起异步通信（`async_op=True`）再按需要等待，以便重叠通信与计算（性能优化），提升吞吐。
* `grad_slices`：保存每个参数对应的本地梯度 slice（post `reduce_scatter`）供后续更新使用。

---

```py
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)
```
总的来说是先为每个参数发起并行的 `reduce_scatter`，把全梯度的“平均片”并行获取到本地。
后面再做本地 Adam 更新（只在该片上），这样通信和计算可以交错。

为每个 param_group 的参数发起 reduce_scatter（把梯度平均并把某一片放到每个 rank）：

`grad = torch.empty_like(params[-1])`：这里先创建一个与最后一个参数形状一样的占位梯度（chat 老师说这行实际上没必要，会被下面 `grad = params[base_i].grad` 覆盖，可删除该行，避免不必要的 alloc。但是太底层了，不确定会不会有预热等优化问题，我也不确定对不对）

大体是把梯度进行分片，然后放到 `reduce_scatter_futures` 里。
这样每个 rank 仅获取自己要更新的那一片的“平均梯度片”。
从而避免广播/收集整个梯度张量，通信量从 `O(N)` -> `O(N/world_size)`（理论上降低通信带宽与显存峰值）。

* `grad_slice = torch.empty_like(grad[:rank_size])`：创建一个空张量来接收 reduce_scatter 的结果（类型/设备/形状匹配该参数的片）。

* `reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())`：发起异步 `reduce_scatter_tensor` 操作：

  * `reduce_scatter_tensor(grad_slice, grad, op=AVG)`：在所有 rank 上把各自 `grad` 张量按第0维分片并把对应片做 `AVG`（平均），最后把平均后的那一片写到每个 rank 的 `grad_slice`（不同 rank 会得到不同片的平均结果）。
  * `async_op=True`：以异步方式发起，返回一个 `Work`，通过 `.get_future()` 转成 `torch.Future` 并加入 `reduce_scatter_futures` 以便之后等待。

---

```py
        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_reduce_futures).wait()
```

等待对应的 reduce_scatter 完成，执行局部 Adam 更新，然后发起 all_gather 把更新片写回全参数

* `reduce_scatter_futures[idx].wait()`：等待对应的 `reduce_scatter` 完成，确保 `grad_slices[idx]` 中有合法数据。
`reduce_scatter_futures` 中存的是 `torch.Future`。
而 `grad_slices` 存的是对应的结果。
所以当 `reduce_scatter_futures[idx].wait()` 完成时，`grad_slices[idx]` 就准备好了。

* `lr = group['lr'] * getattr(p, "lr_mul", 1.0)` 这里是设计支持给单个参数设置 `p.lr_mul` 来放大/缩放学习率（方便例如 bias 或某些层使用不同 lr）。

接下来就是一个 AdamW 的更新方式。

* `all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())` 这行代码是在发起异步 `all_gather_into_tensor`，把每个 rank 的 `p_slice` 汇集到 `p`，即把更新后的片放回到完整参数。

因为刚刚是异步的，所以最后 `torch.futures.collect_all(all_reduce_futures).wait()` 等待所有的 `all_gather` 完成。

### 网络层

`CastedLinear` 就是一个把参数使用 fp8 计算前向的 linear 层

`nn.Buffer` 是用来存储网络的固定参数的。不会计算梯度，且可以设置是否加载到模型文件。语义更明确。
